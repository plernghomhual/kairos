"""Backtest report generation — text summary + optional HTML.

Outputs a structured performance report suitable for CLI display
or export to a file.
"""

from __future__ import annotations

import csv
import json
import math
from io import StringIO
from statistics import mean
from typing import Any

from kairos.backtest.engine import BacktestResult, BacktestTrade
from kairos.models.signal_event import SignalEvent


def _pct_str(v: float) -> str:
    return f"{v:+.2f}%" if v >= 0 else f"{v:.2f}%"


def _iso(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _clean_field(value: Any) -> str:
    return " ".join(str(value if value is not None else "").split())


def _finite_or_none(value: Any) -> float | int | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _fmt_float(value: Any, digits: int = 6) -> str:
    numeric = _finite_or_none(value)
    return "" if numeric is None else f"{numeric:.{digits}f}"


def _closed_trades(result: BacktestResult) -> list[BacktestTrade]:
    return [trade for trade in result.trades if trade.closed and trade.pnl_pct is not None]


def _trade_direction(trade: BacktestTrade) -> str:
    direction = (trade.direction or "").lower()
    if direction in {"bullish", "long", "buy"}:
        return "long"
    if direction in {"bearish", "short", "sell"}:
        return "short"
    return direction


def _trade_position(trade: BacktestTrade) -> float:
    size = float(getattr(trade, "position_size", 0.0) or 0.0)
    if _trade_direction(trade) == "short":
        return -size
    return size


def _holding_bars(trade: BacktestTrade) -> int | None:
    if trade.exit_idx is None:
        return None
    return int(trade.exit_idx - trade.entry_idx)


def _trade_capital_after(
    result: BacktestResult,
    trade: BacktestTrade,
    running_capital: float,
) -> float:
    explicit = getattr(trade, "cumulative_capital", None)
    numeric = _finite_or_none(explicit)
    if numeric is not None:
        return numeric

    pnl = _finite_or_none(trade.pnl_pct)
    if pnl is None:
        return running_capital
    size = float(getattr(trade, "position_size", 1.0) or 1.0)
    return running_capital * (1.0 + pnl * abs(size))


def _benchmark_return_pct(result: BacktestResult) -> float:
    if not result.benchmark_curve:
        return 0.0
    start = result.initial_capital or result.benchmark_curve[0]
    if not start:
        return 0.0
    return (result.benchmark_curve[-1] / start - 1.0) * 100.0


def _series_for_rows(values: list[Any], row_count: int) -> list[Any]:
    if row_count <= 0:
        return []
    if not values:
        return [""] * row_count
    if len(values) == row_count + 1:
        values = values[1:]
    elif len(values) > row_count:
        values = values[:row_count]
    if len(values) < row_count:
        values = values + [values[-1]] * (row_count - len(values))
    return values


def _active_trade_at(result: BacktestResult, ts: Any, row_idx: int) -> BacktestTrade | None:
    for trade in reversed(result.trades):
        if hasattr(trade.entry_time, "timestamp") and hasattr(ts, "timestamp"):
            exit_time = trade.exit_time
            if trade.entry_time <= ts and (exit_time is None or ts <= exit_time):
                return trade
        elif trade.entry_idx <= row_idx and (trade.exit_idx is None or row_idx <= trade.exit_idx):
            return trade
    return None


def _regime_breakdown(result: BacktestResult) -> dict[str, dict[str, float | int]]:
    breakdown: dict[str, dict[str, float | int]] = {}
    for regime in sorted({trade.regime for trade in _closed_trades(result)}):
        trades = [trade for trade in _closed_trades(result) if trade.regime == regime]
        wins = [trade for trade in trades if float(trade.pnl_pct or 0.0) > 0.0]
        returns = [float(trade.pnl_pct or 0.0) * 100.0 for trade in trades]
        breakdown[regime] = {
            "trade_count": len(trades),
            "win_rate": round(len(wins) / len(trades) * 100.0, 4) if trades else 0.0,
            "avg_return": round(mean(returns), 6) if returns else 0.0,
        }
    return breakdown


def format_compact_track_record(result: BacktestResult) -> str:
    """Return a compact one-line track record summary for the live dashboard."""
    from io import StringIO

    from rich.console import Console
    from rich.table import Table

    buf = StringIO()
    console = Console(file=buf, force_terminal=False)

    t = Table(show_header=False, box=None, padding=(0, 2), title_style="bold dim")
    t.add_column("Metric", style="dim")
    t.add_column("Value", style="white")

    bench_ret = (result.benchmark_curve[-1] / result.initial_capital - 1) * 100
    color_sharpe = "green" if result.sharpe > 0.5 else "yellow" if result.sharpe > 0 else "red"
    color_dd = "green" if result.max_drawdown_pct < 20 else "yellow" if result.max_drawdown_pct < 35 else "red"

    conflict_pct = (result.conflict_days / result.total_trading_days * 100) if result.total_trading_days > 0 else 0

    t.add_row("Total Return", _pct_str(result.total_return_pct))
    t.add_row("Benchmark (Buy & Hold)", _pct_str(bench_ret))
    t.add_row("Sharpe", f"[{color_sharpe}]{result.sharpe}[/]")
    t.add_row("Max DD", f"[{color_dd}]{result.max_drawdown_pct:.1f}%[/]")
    t.add_row(
        "Win Rate",
        f"{result.win_rate:.1f}%  ({result.winning_trades}W/{result.losing_trades}L)",
    )
    t.add_row("Trades", str(result.total_trades))
    if result.capitulation_trades > 0:
        t.add_row("Capitulation Trades", str(result.capitulation_trades))
    t.add_row("Avg Kelly", f"{result.avg_kelly_pct:.1f}%")
    t.add_row(
        "Conflict Days",
        f"{result.conflict_days}/{result.total_trading_days}  ({conflict_pct:.0f}%)",
    )

    from rich.panel import Panel

    panel = Panel(
        t,
        title="[bold]Historical Track Record (365d walk-forward)[/bold]",
        border_style="dim",
        padding=(0, 1),
    )
    console.print("\n", panel)
    return buf.getvalue()


def _color(val: float, good_high: bool = True) -> str:
    """Return rich color tag based on sign."""
    if good_high:
        return "green" if val >= 0 else "red"
    return "red" if val >= 0 else "green"


def format_text_report(result: BacktestResult) -> str:
    """Return a formatted text summary of backtest results."""
    from io import StringIO

    from rich.console import Console
    from rich.table import Table

    buf = StringIO()
    console = Console(file=buf, force_terminal=False)

    # ── Header ──
    console.print("[bold white]Kairos Backtest Report[/bold white]")
    console.print(
        f"[dim]{result.asset}  ·  {result.total_trades} trades  ·  " f"${result.initial_capital:,.0f} initial[/dim]\n"
    )

    # ── Performance Table ──
    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_column("Metric", style="dim")
    t.add_column("Value", style="white")

    t.add_row("Total Return", _pct_str(result.total_return_pct))
    t.add_row("Annualized Return", _pct_str(result.annualized_return_pct))
    t.add_row("Final Capital", f"${result.final_capital:,.2f}")
    t.add_row(
        "Benchmark Return (Buy & Hold)",
        f"[{_color(result.total_return_pct)}]"
        f"{_pct_str((result.benchmark_curve[-1] / result.initial_capital - 1) * 100)}[/]",
    )
    t.add_row("")
    t.add_row("Sharpe Ratio", f"[{_color(result.sharpe)}]{result.sharpe}[/]")
    t.add_row("Sortino Ratio", f"[{_color(result.sortino)}]{result.sortino}[/]")
    t.add_row(
        "Max Drawdown",
        f"[{_color(-result.max_drawdown_pct, False)}]{result.max_drawdown_pct:.2f}%[/]",
    )
    t.add_row("")
    t.add_row("Win Rate", f"{result.win_rate:.1f}%")
    t.add_row("Winning / Losing", f"{result.winning_trades} / {result.losing_trades}")
    t.add_row("Total Trades", str(result.total_trades))
    t.add_row("Avg Holding", f"{result.avg_holding_bars:.1f} days")
    t.add_row("Profit Factor", f"{result.profit_factor:.2f}")
    t.add_row("Avg Kelly", f"{result.avg_kelly_pct:.1f}%")

    console.print(t)

    # ── Trade Journal ──
    if result.trades:
        console.print(f"\n[bold dim]Recent Trades (last {min(len(result.trades), 10)} shown)[/bold dim]")
        jt = Table(show_header=True, header_style="dim", box=None, padding=(0, 1))
        jt.add_column("Entry", style="dim", no_wrap=True)
        jt.add_column("Dir", style="bold", width=6)
        jt.add_column("Conf", justify="right")
        jt.add_column("Regime", width=14)
        jt.add_column("Holding", justify="right")
        jt.add_column("P&L", justify="right")

        for trade in result.trades[-10:]:
            direction = trade.direction or ""
            dir_style = "green" if direction == "bullish" else "red"
            entry_str = (
                trade.entry_time.strftime("%m-%d")
                if hasattr(trade.entry_time, "strftime")
                else str(trade.entry_time)[:10]
            )
            conf_str = f"{trade.confidence:.0%}" if trade.confidence else "—"
            holding = f"{trade.exit_idx - trade.entry_idx}d" if trade.exit_idx is not None else "open"
            pnl_str = ""
            if trade.pnl_pct is not None:
                pnl_color = "green" if trade.pnl_pct >= 0 else "red"
                pnl_str = f"[{pnl_color}]{trade.pnl_pct:+.1%}[/{pnl_color}]"
            jt.add_row(
                entry_str,
                f"[{dir_style}]{direction}[/{dir_style}]",
                conf_str,
                trade.regime or "—",
                holding,
                pnl_str,
            )

        console.print(jt)

    # ── Interpretation ──
    console.print("\n[bold dim]Interpretation[/bold dim]")
    lines = []
    if result.sharpe > 1.0:
        lines.append("✓ Sharpe > 1.0 — risk-adjusted returns are attractive")
    elif result.sharpe > 0.5:
        lines.append(f"~ Sharpe {result.sharpe:.2f} — moderate risk-adjusted returns")
    else:
        lines.append(f"⚠ Sharpe {result.sharpe:.2f} — low risk-adjusted returns vs. risk-free")

    if result.max_drawdown_pct > 30:
        lines.append(f"⚠ Max drawdown {result.max_drawdown_pct:.1f}% — high peak-to-trough loss")
    elif result.max_drawdown_pct > 15:
        lines.append(f"~ Max drawdown {result.max_drawdown_pct:.1f}% — moderate peak-to-trough loss")
    else:
        lines.append(f"✓ Max drawdown {result.max_drawdown_pct:.1f}% — well-controlled")

    if result.win_rate > 60:
        lines.append(f"✓ Win rate {result.win_rate:.0f}% — majority of trades profitable")
    elif result.win_rate > 45:
        lines.append(f"~ Win rate {result.win_rate:.0f}% — near break-even")
    else:
        lines.append(f"⚠ Win rate {result.win_rate:.0f}% — most trades lose; check if winners are larger")

    for l in lines:
        console.print(f"  {l}")

    console.print(
        "\n[dim]Note: past performance does not guarantee future results. "
        "This backtest is for research purposes only.[/dim]"
    )

    return buf.getvalue()


def format_csv_trades(result: BacktestResult) -> str:
    """Return closed trades as a machine-parseable CSV string."""
    fields = [
        "entry_idx",
        "entry_time",
        "entry_price",
        "direction",
        "confidence",
        "regime_at_entry",
        "mechanism",
        "estimated_hours",
        "is_capitulation",
        "exit_idx",
        "exit_time",
        "exit_price",
        "pnl_pct",
        "position_size",
        "holding_bars",
        "cumulative_capital",
    ]
    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    writer.writeheader()

    running_capital = float(result.initial_capital)
    for trade in result.trades:
        if not trade.closed:
            continue
        running_capital = _trade_capital_after(result, trade, running_capital)
        writer.writerow(
            {
                "entry_idx": trade.entry_idx,
                "entry_time": _iso(trade.entry_time),
                "entry_price": _fmt_float(trade.entry_price, 2),
                "direction": _trade_direction(trade),
                "confidence": _fmt_float(trade.confidence, 4),
                "regime_at_entry": trade.regime,
                "mechanism": trade.mechanism,
                "estimated_hours": _fmt_float(trade.estimated_hours, 2),
                "is_capitulation": str(bool(getattr(trade, "is_capitulation", False))).lower(),
                "exit_idx": "" if trade.exit_idx is None else trade.exit_idx,
                "exit_time": _iso(trade.exit_time),
                "exit_price": _fmt_float(trade.exit_price, 2),
                "pnl_pct": _fmt_float(trade.pnl_pct, 6),
                "position_size": _fmt_float(getattr(trade, "position_size", 0.0), 6),
                "holding_bars": "" if _holding_bars(trade) is None else _holding_bars(trade),
                "cumulative_capital": _fmt_float(running_capital, 2),
            }
        )

    return buf.getvalue().rstrip("\n")


def format_equity_csv(result: BacktestResult) -> str:
    """Export daily equity curve with benchmark comparison."""
    fields = [
        "timestamp",
        "portfolio_value",
        "benchmark_value",
        "daily_return",
        "cumulative_return",
        "drawdown_pct",
        "position",
        "regime",
        "confidence",
    ]
    row_count = len(result.timestamps)
    portfolio_values = _series_for_rows(list(result.equity_curve), row_count)
    benchmark_values = _series_for_rows(list(result.benchmark_curve), row_count)
    confidence_values = _series_for_rows(list(result.confidence_series), row_count)

    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    writer.writeheader()

    peak = float(result.initial_capital)
    previous = None
    for idx, ts in enumerate(result.timestamps):
        portfolio = float(portfolio_values[idx] or 0.0)
        benchmark = benchmark_values[idx]
        peak = max(peak, portfolio)
        daily_return = 0.0 if previous is None or previous == 0 else portfolio / previous - 1.0
        cumulative_return = 0.0 if result.initial_capital == 0 else portfolio / result.initial_capital - 1.0
        drawdown_pct = 0.0 if peak == 0 else (portfolio / peak - 1.0) * 100.0
        active_trade = _active_trade_at(result, ts, idx)

        writer.writerow(
            {
                "timestamp": _iso(ts),
                "portfolio_value": _fmt_float(portfolio, 2),
                "benchmark_value": _fmt_float(benchmark, 2),
                "daily_return": _fmt_float(daily_return, 8),
                "cumulative_return": _fmt_float(cumulative_return, 8),
                "drawdown_pct": _fmt_float(drawdown_pct, 4),
                "position": _fmt_float(_trade_position(active_trade), 6) if active_trade else "0.000000",
                "regime": active_trade.regime if active_trade else "",
                "confidence": _fmt_float(confidence_values[idx], 4),
            }
        )
        previous = portfolio

    return buf.getvalue().rstrip("\n")


def format_metrics_json(result: BacktestResult) -> str:
    """Return a JSON string with all performance metrics."""
    closed_trades = _closed_trades(result)
    kelly_values = [abs(float(getattr(trade, "position_size", 0.0) or 0.0)) * 100.0 for trade in closed_trades]
    max_drawdown = float(result.max_drawdown_pct)
    calmar = result.annualized_return_pct / max_drawdown if max_drawdown > 0 else 0.0

    payload = {
        "metadata": {
            "asset": result.asset,
            "strategy": result.strategy,
            "initial_capital": result.initial_capital,
            "date_range": {
                "start": _iso(result.timestamps[0]) if result.timestamps else "",
                "end": _iso(result.timestamps[-1]) if result.timestamps else "",
            },
        },
        "returns": {
            "total_return_pct": result.total_return_pct,
            "annualized_return_pct": result.annualized_return_pct,
            "benchmark_return_pct": round(_benchmark_return_pct(result), 6),
        },
        "risk": {
            "sharpe": _finite_or_none(result.sharpe),
            "sortino": _finite_or_none(result.sortino),
            "max_drawdown_pct": result.max_drawdown_pct,
            "calmar_ratio": round(calmar, 6),
        },
        "trades": {
            "total": result.total_trades,
            "winning": result.winning_trades,
            "losing": result.losing_trades,
            "win_rate": result.win_rate,
            "profit_factor": _finite_or_none(result.profit_factor),
            "avg_holding_bars": result.avg_holding_bars,
        },
        "kelly": {
            "avg_kelly_pct": result.avg_kelly_pct,
            "max_kelly_pct": round(max(kelly_values), 6) if kelly_values else 0.0,
            "min_kelly_pct": round(min(kelly_values), 6) if kelly_values else 0.0,
        },
        "regime_breakdown": _regime_breakdown(result),
    }
    return json.dumps(payload, sort_keys=True, allow_nan=False)


def format_comparison_csv(results: dict[str, BacktestResult]) -> str:
    """Return CSV comparing multiple strategy results side-by-side."""
    strategy_names = list(results)
    metrics = [
        ("total_return_pct", lambda r: r.total_return_pct),
        ("annualized_return_pct", lambda r: r.annualized_return_pct),
        ("benchmark_return_pct", _benchmark_return_pct),
        ("sharpe", lambda r: r.sharpe),
        ("sortino", lambda r: r.sortino),
        ("max_drawdown_pct", lambda r: r.max_drawdown_pct),
        (
            "calmar_ratio",
            lambda r: r.annualized_return_pct / r.max_drawdown_pct if r.max_drawdown_pct > 0 else 0.0,
        ),
        ("win_rate", lambda r: r.win_rate),
        ("total_trades", lambda r: r.total_trades),
        ("profit_factor", lambda r: _finite_or_none(r.profit_factor)),
        ("avg_holding_bars", lambda r: r.avg_holding_bars),
        ("avg_kelly_pct", lambda r: r.avg_kelly_pct),
        ("conflict_days", lambda r: r.conflict_days),
        ("capitulation_trades", lambda r: r.capitulation_trades),
    ]

    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=["metric", *strategy_names], lineterminator="\n")
    writer.writeheader()
    for metric_name, getter in metrics:
        row = {"metric": metric_name}
        for strategy_name, result in results.items():
            value = getter(result)
            row[strategy_name] = "" if value is None else value
        writer.writerow(row)
    return buf.getvalue().rstrip("\n")


def format_signal_log(event: SignalEvent, current_price: float, ctx: dict) -> str:
    """Return a structured tab-separated log line for a signal event."""
    mechanism = ctx.get("mechanism", event.mechanism)
    estimated_hours = ctx.get("estimated_hours", event.estimated_hours)
    try:
        hours = f"{float(estimated_hours):.0f}h"
    except (TypeError, ValueError):
        hours = _clean_field(estimated_hours)
    return "\t".join(
        [
            _iso(event.triggered_at),
            _clean_field(ctx.get("asset", event.asset)),
            _clean_field(event.direction),
            _fmt_float(event.confidence, 4),
            _clean_field(event.regime),
            f"${current_price:,.2f}",
            _clean_field(mechanism),
            hours,
        ]
    )


def format_regime_report(result: BacktestResult) -> str:
    """Return a rich-formatted per-regime trading statistics table."""
    from rich.console import Console
    from rich.table import Table

    buf = StringIO()
    console = Console(file=buf, force_terminal=False)

    table = Table(title="Regime Performance", show_header=True, header_style="dim", box=None)
    table.add_column("Regime", style="white")
    table.add_column("Trades", justify="right")
    table.add_column("Win Rate", justify="right")
    table.add_column("Avg Return", justify="right")
    table.add_column("Avg Confidence", justify="right")
    table.add_column("Total P&L", justify="right")

    for regime in sorted({trade.regime for trade in _closed_trades(result)}):
        trades = [trade for trade in _closed_trades(result) if trade.regime == regime]
        wins = [trade for trade in trades if float(trade.pnl_pct or 0.0) > 0.0]
        returns = [float(trade.pnl_pct or 0.0) for trade in trades]
        confidences = [float(trade.confidence or 0.0) for trade in trades]
        contribution = sum(
            float(trade.pnl_pct or 0.0) * abs(float(getattr(trade, "position_size", 1.0) or 1.0)) for trade in trades
        )
        table.add_row(
            regime,
            str(len(trades)),
            f"{len(wins) / len(trades) * 100.0:.1f}%" if trades else "0.0%",
            f"{mean(returns) * 100.0:+.2f}%" if returns else "+0.00%",
            f"{mean(confidences):.2f}" if confidences else "0.00",
            f"{contribution * 100.0:+.2f}%",
        )

    console.print(table)
    return buf.getvalue()
