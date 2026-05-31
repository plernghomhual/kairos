"""Live paper trading simulator for Kairos signals.

Tracks a mock per-asset account through forward signals without placing
orders or touching real capital.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import duckdb

from kairos.db import create_schema, get_connection
from kairos.models.signal_event import SignalEvent


@dataclass
class PaperPosition:
    asset: str
    direction: str
    entry_price: float
    entry_time: datetime
    size: float
    signal_id: str
    exit_price: float | None = None
    exit_time: datetime | None = None
    pnl_pct: float | None = None
    closed: bool = False


@dataclass
class PaperAccount:
    asset: str
    initial_capital: float
    current_capital: float
    positions: list[PaperPosition]
    open_position: PaperPosition | None
    trades: list[PaperPosition]
    equity_curve: list[float]
    timestamps: list[datetime]


class PaperTradingEngine:
    """Signal-driven paper portfolio with one open position per asset."""

    def __init__(self, initial_capital: float = 10_000.0, db_path: str | None = None):
        self.initial_capital = float(initial_capital)
        self._accounts: dict[str, PaperAccount] = {}
        self._conn: duckdb.DuckDBPyConnection | None = None
        if db_path is not None:
            self._conn = get_connection(db_path)
            create_schema(self._conn)
            self._load_trades()

    def process_signal(self, event: SignalEvent, current_price: float) -> PaperPosition | None:
        """Open, close, hold, or flip a paper position from a Kairos signal."""
        account = self._get_or_create_account(event.asset)

        if event.direction == "neutral":
            if account.open_position is None:
                self._append_equity(account, account.current_capital)
                return None
            return self._close_position(account, current_price, event)

        next_direction = _signal_direction(event.direction)
        if next_direction is None:
            self._append_equity(account, account.current_capital)
            return None

        if account.open_position is None:
            return self._open_position(account, event, current_price, next_direction)

        if account.open_position.direction == next_direction:
            self.update_price(event.asset, current_price)
            return None

        try:
            self._close_position(account, current_price, event)
        except Exception:
            return None
        if account.open_position is not None:
            return None
        return self._open_position(account, event, current_price, next_direction)

    def update_price(self, asset: str, price: float) -> None:
        """Update mark-to-market P&L and equity for an open position."""
        account = self._get_or_create_account(asset)
        position = account.open_position
        if position is None:
            self._append_equity(account, account.current_capital)
            return

        position.pnl_pct = _position_return(position, price)
        equity = account.current_capital * (1 + position.size * position.pnl_pct)
        self._append_equity(account, equity)

    def get_account(self, asset: str) -> PaperAccount:
        """Return account state for an asset."""
        return self._get_or_create_account(asset)

    def get_summary(self) -> dict:
        """Return aggregate paper trading state across all tracked assets."""
        total_initial = sum(account.initial_capital for account in self._accounts.values())
        total_equity = sum(_latest_equity(account) for account in self._accounts.values())
        trade_count = sum(len(account.trades) for account in self._accounts.values())
        winning = sum(
            1
            for account in self._accounts.values()
            for trade in account.trades
            if trade.pnl_pct is not None and trade.pnl_pct > 0
        )
        open_pnl = sum(_latest_equity(account) - account.current_capital for account in self._accounts.values())
        return {
            "total_equity": round(total_equity, 2),
            "total_return_pct": round((total_equity / total_initial - 1) * 100, 4) if total_initial else 0.0,
            "open_pnl": round(open_pnl, 2),
            "trade_count": trade_count,
            "win_rate": round(winning / trade_count, 4) if trade_count else 0.0,
        }

    def get_daily_pnl(self, asset: str) -> float:
        """Return today's account P&L, including realized and unrealized moves."""
        account = self._get_or_create_account(asset)
        today = datetime.now(timezone.utc).date()
        todays_points = [
            equity for equity, ts in zip(account.equity_curve, account.timestamps) if _as_utc(ts).date() == today
        ]
        if not todays_points:
            return 0.0
        return round(todays_points[-1] - todays_points[0], 2)

    def close(self) -> None:
        """Close the optional persistence connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _get_or_create_account(self, asset: str) -> PaperAccount:
        if asset not in self._accounts:
            now = datetime.now(timezone.utc)
            self._accounts[asset] = PaperAccount(
                asset=asset,
                initial_capital=self.initial_capital,
                current_capital=self.initial_capital,
                positions=[],
                open_position=None,
                trades=[],
                equity_curve=[self.initial_capital],
                timestamps=[now],
            )
        return self._accounts[asset]

    def _open_position(
        self,
        account: PaperAccount,
        event: SignalEvent,
        current_price: float,
        direction: str,
    ) -> PaperPosition:
        slippage = _slippage(event.regime, direction="buy" if direction == "long" else "sell")
        entry_price = current_price * (1 + slippage if direction == "long" else 1 - slippage)
        position = PaperPosition(
            asset=event.asset,
            direction=direction,
            entry_price=entry_price,
            entry_time=event.triggered_at,
            size=_kelly(event.confidence),
            signal_id=event.id,
        )
        account.open_position = position
        account.positions.append(position)
        self._append_equity(account, account.current_capital)
        self._persist_open(position, event)
        return position

    def _close_position(
        self,
        account: PaperAccount,
        current_price: float,
        event: SignalEvent,
    ) -> PaperPosition:
        position = account.open_position
        if position is None:
            raise ValueError("cannot close a missing paper position")

        slippage = _slippage(event.regime, direction="sell" if position.direction == "long" else "buy")
        exit_price = current_price * (1 - slippage if position.direction == "long" else 1 + slippage)
        position.exit_price = exit_price
        position.exit_time = event.triggered_at
        position.pnl_pct = _position_return(position, exit_price)
        position.closed = True

        account.current_capital *= 1 + position.size * position.pnl_pct
        account.open_position = None
        account.trades.append(position)
        self._append_equity(account, account.current_capital)
        self._persist_close(position)
        return position

    def _append_equity(self, account: PaperAccount, equity: float) -> None:
        account.equity_curve.append(float(equity))
        account.timestamps.append(datetime.now(timezone.utc))

    def _persist_open(self, position: PaperPosition, event: SignalEvent) -> None:
        if self._conn is None:
            return
        existing = self._conn.execute(
            "SELECT COUNT(*) FROM paper_trades WHERE signal_id = ? AND asset = ? AND closed = FALSE",
            [position.signal_id, position.asset],
        ).fetchone()
        if existing and existing[0] > 0:
            return
        self._conn.execute(
            """
            INSERT INTO paper_trades (
                asset, signal_id, direction, entry_price, entry_time, size,
                exit_price, exit_time, pnl_pct, closed,
                signal_confidence, signal_regime
            ) VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, NULL, FALSE, ?, ?)
            """,
            [
                position.asset,
                position.signal_id,
                position.direction,
                position.entry_price,
                position.entry_time,
                position.size,
                event.confidence,
                event.regime,
            ],
        )

    def _persist_close(self, position: PaperPosition) -> None:
        if self._conn is None:
            return
        self._conn.execute(
            """
            UPDATE paper_trades
            SET exit_price = ?, exit_time = ?, pnl_pct = ?, closed = TRUE
            WHERE signal_id = ? AND asset = ? AND closed = FALSE
            """,
            [
                position.exit_price,
                position.exit_time,
                position.pnl_pct,
                position.signal_id,
                position.asset,
            ],
        )

    def _load_trades(self) -> None:
        if self._conn is None:
            return
        rows = self._conn.execute(
            """
            SELECT asset, signal_id, direction, entry_price, entry_time, size,
                   exit_price, exit_time, pnl_pct, closed
            FROM paper_trades
            ORDER BY entry_time ASC
            """
        ).fetchall()
        for row in rows:
            position = _position_from_row(row)
            account = self._get_or_create_account(position.asset)
            account.positions.append(position)
            if position.closed:
                account.trades.append(position)
                if position.pnl_pct is not None:
                    account.current_capital *= 1 + position.size * position.pnl_pct
                    self._append_equity(account, account.current_capital)
            else:
                account.open_position = position


def format_paper_summary(engine: PaperTradingEngine, asset: str) -> str:
    """Return a rich-formatted summary of paper trading state."""
    account = engine.get_account(asset)
    equity = _latest_equity(account)
    total_return = (equity / account.initial_capital - 1) * 100 if account.initial_capital else 0.0
    open_position = account.open_position
    if open_position is None:
        position_line = "[dim]Open:[/dim] none"
    else:
        pnl = open_position.pnl_pct or 0.0
        position_line = (
            f"[dim]Open:[/dim] {open_position.direction} "
            f"{open_position.size:.1%} @ ${open_position.entry_price:,.2f} "
            f"({pnl:+.2%})"
        )
    return "\n".join(
        [
            f"[bold]{asset} Paper Account[/bold]",
            f"[dim]Equity:[/dim] ${equity:,.2f} ({total_return:+.2f}%)",
            f"[dim]Closed trades:[/dim] {len(account.trades)}",
            position_line,
        ]
    )


def _kelly(confidence: float) -> float:
    from kairos.backtest.engine import _kelly_fraction

    return _kelly_fraction(confidence)


def _slippage(regime: str, direction: str = "buy") -> float:
    from kairos.backtest.engine import _compute_slippage

    return _compute_slippage(regime, 0.0, direction=direction)


def _signal_direction(direction: str) -> str | None:
    if direction == "bullish":
        return "long"
    if direction == "bearish":
        return "short"
    return None


def _position_return(position: PaperPosition, price: float) -> float:
    side = 1.0 if position.direction == "long" else -1.0
    return (price / position.entry_price - 1) * side


def _latest_equity(account: PaperAccount) -> float:
    return account.equity_curve[-1] if account.equity_curve else account.current_capital


def _position_from_row(row: tuple[Any, ...]) -> PaperPosition:
    (
        asset,
        signal_id,
        direction,
        entry_price,
        entry_time,
        size,
        exit_price,
        exit_time,
        pnl_pct,
        closed,
    ) = row
    return PaperPosition(
        asset=asset,
        direction=direction,
        entry_price=float(entry_price),
        entry_time=_as_utc(entry_time),
        size=float(size),
        signal_id=signal_id,
        exit_price=float(exit_price) if exit_price is not None else None,
        exit_time=_as_utc(exit_time) if exit_time is not None else None,
        pnl_pct=float(pnl_pct) if pnl_pct is not None else None,
        closed=bool(closed),
    )


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
