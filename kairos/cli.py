"""
Kairos TUI - single terminal dashboard for all engine views.

Default: Live signal dashboard.
Shortcuts: [b] backtest  [c] compare  [h] history  [s] serve
           [p] paper     [1/2/3] asset [tab] next  [q] quit
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
import select
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import StringIO
from typing import Any

import typer
from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

if os.name == "nt":
    try:
        from colorama import just_fix_windows_console

        just_fix_windows_console()
    except ImportError:
        pass


# Kairos Terminal Theme
THEME = {
    "bg": "rgb(10,10,15)",
    "accent": "rgb(0,200,255)",  # cyan accent
    "bullish": "rgb(0,230,118)",  # green
    "bearish": "rgb(255,82,82)",  # red
    "neutral": "rgb(180,180,180)",  # grey
    "warning": "rgb(255,200,50)",  # amber
    "dim": "rgb(100,100,120)",
    "panel_border": "rgb(40,40,60)",
    "highlight": "rgb(255,255,255)",
}


def _detect_color_system() -> str | None:
    """Return a Rich color system that fits the active terminal."""
    if os.name == "nt":
        return "auto"
    colorterm = os.environ.get("COLORTERM", "").lower()
    term = os.environ.get("TERM", "").lower()
    if "truecolor" in colorterm or "24bit" in colorterm:
        return "truecolor"
    if "256color" in term:
        return "256"
    if term and term != "dumb":
        return "standard"
    return None


_COLOR_SYSTEM = _detect_color_system()
console = Console(color_system=_COLOR_SYSTEM)

app = typer.Typer(
    help="Kairos - sees what moves before price does.",
    no_args_is_help=False,
    invoke_without_command=True,
)

_BASIC_THEME = {
    "bg": "black",
    "accent": "cyan",
    "bullish": "green",
    "bearish": "red",
    "neutral": "white",
    "warning": "yellow",
    "dim": "bright_black",
    "panel_border": "blue",
    "highlight": "white",
}

_ASSETS = ("BTC", "ETH", "SOL")
_ASSET_KEYS = {"1": "BTC", "2": "ETH", "3": "SOL"}
_VIEW_ORDER = ("signal", "backtest", "compare", "history", "serve", "paper")
_VIEW_KEYS = {
    "b": "backtest",
    "c": "compare",
    "h": "history",
    "p": "paper",
}
_REFRESH_INTERVAL = 10.0
_STATIC_REFRESH_INTERVAL = 60.0
_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_RESIZE_DEBOUNCE_SECONDS = 0.10
_TOAST_SECONDS = 5.0

_STRATEGIES = {
    "A": "Strict Defense",
    "B": "Capitulation Offense",
    "C": "Exhaustion Offense",
}


@app.callback(invoke_without_command=True)
def _default(ctx: typer.Context) -> None:
    """Run the interactive dashboard when invoked through the Typer app."""
    if ctx.invoked_subcommand is None:
        run_cli()


@app.command()
def run(
    asset: str = typer.Option("BTC", "--asset", "-a", help="Asset to analyze (BTC, ETH, SOL)"),
    no_tui: bool = typer.Option(False, "--no-tui", help="One-shot mode, no interactive TUI"),
) -> None:
    """Fetch live CoinGecko/Fear & Greed data and show the Kairos dashboard."""
    args = ["kairos", "--asset", asset]
    if no_tui:
        args.append("--no-tui")
    original = sys.argv
    try:
        sys.argv = args
        run_cli()
    finally:
        sys.argv = original


def _style(name: str, *, bold: bool = False) -> str:
    color = THEME.get(name, THEME["neutral"])
    if _COLOR_SYSTEM in (None, "standard"):
        color = _BASIC_THEME.get(name, "white")
    return f"{'bold ' if bold else ''}{color}"


def _panel_bg() -> str:
    if _COLOR_SYSTEM in (None, "standard"):
        return ""
    return f"on {THEME['bg']}"


def _markup(text: str) -> str:
    return text.replace("[", "\\[").replace("]", "\\]")


@dataclass
class _ViewRender:
    body: Any
    metric: str = "warming up"
    updated_at: float = field(default_factory=time.time)


@dataclass
class _Toast:
    message: str
    style: str
    expires_at: float


@dataclass
class _DashboardState:
    asset: str = "BTC"
    current_view: str = "signal"
    last_view_before_errors: str = "signal"
    running: bool = True
    force_refresh: bool = True
    show_error_log: bool = False
    server_requested: bool = False
    server_running: bool = False
    server_host: str = "127.0.0.1"
    server_port: int = 8000
    server_process: subprocess.Popen[Any] | None = None
    last_metrics: dict[str, str] = field(default_factory=dict)
    error_log: list[str] = field(default_factory=list)
    toast: _Toast | None = None
    resize_version: int = 0

    def cache_key(self, view: str | None = None) -> tuple[str, str]:
        return (view or self.current_view, self.asset)


def _toast(state: _DashboardState, message: str, style: str = "warning") -> None:
    state.toast = _Toast(message, _style(style, bold=True), time.monotonic() + _TOAST_SECONDS)


def _record_error(state: _DashboardState, view: str, exc: BaseException) -> None:
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    message = f"{stamp} [{view}] {type(exc).__name__}: {exc}"
    state.error_log.append(message)
    state.error_log[:] = state.error_log[-12:]
    _toast(state, message, "warning")


def _cycle_view(state: _DashboardState, step: int) -> None:
    state.show_error_log = False
    try:
        idx = _VIEW_ORDER.index(state.current_view)
    except ValueError:
        idx = 0
    state.current_view = _VIEW_ORDER[(idx + step) % len(_VIEW_ORDER)]
    state.force_refresh = True


def _key_handler(key: str, state: _DashboardState) -> None:
    """Mutate dashboard state for one keyboard event."""
    if not key:
        return

    if key == "\x03":
        state.running = False
        return

    if key in ("\t", "tab"):
        _cycle_view(state, 1)
        return

    if key in ("\x1b[Z", "shift+tab"):
        _cycle_view(state, -1)
        return

    if key == "E":
        if state.show_error_log:
            state.show_error_log = False
            state.current_view = state.last_view_before_errors
        else:
            state.last_view_before_errors = state.current_view if state.current_view != "errors" else "signal"
            state.show_error_log = True
            state.current_view = "errors"
        return

    low = key.lower()
    if low == "q":
        state.running = False
    elif low == "r":
        state.force_refresh = True
    elif low == "s":
        state.show_error_log = False
        state.current_view = "serve"
        state.server_requested = True
    elif low in _VIEW_KEYS:
        state.current_view = _VIEW_KEYS[low]
        state.show_error_log = False
        state.force_refresh = True
    elif key in _ASSET_KEYS:
        state.asset = _ASSET_KEYS[key]
        state.force_refresh = True


def _header_help(small: bool = False) -> Text:
    help_text = Text()
    pairs = [
        ("b", "Backtest"),
        ("c", "Compare"),
        ("h", "History"),
        ("s", "API"),
        ("p", "Paper"),
        ("Tab", "Next"),
        ("E", "Errors"),
        ("q", "Quit"),
    ]
    for key, label in pairs:
        help_text.append(f" {key}", style=_style("highlight", bold=True))
        if not small:
            help_text.append(f" {label}", style=_style("accent", bold=True))
    help_text.append("  |  ", style=_style("dim"))
    help_text.append("1 BTC  2 ETH  3 SOL", style=_style("dim"))
    return help_text


def _make_layout(
    current_view: str,
    body: Any,
    footer: Any | None = None,
    *,
    asset: str = "BTC",
    server_running: bool = False,
    status: Any | None = None,
    error_count: int = 0,
    terminal_size: tuple[int, int] | None = None,
) -> Layout:
    """Build the stable TUI shell."""
    width, height = terminal_size or (console.size.width, console.size.height)
    small = width < 80 or height < 24
    layout = Layout(name="root")

    if small:
        layout.split_column(
            Layout(name="header", size=1),
            Layout(name="body", ratio=1),
            Layout(name="status", size=1),
        )
        api = "API:on" if server_running else "API:off"
        header = Text(
            f"KAIROS {asset} [{current_view.upper()}] {api}  <80x24",
            style=_style("accent", bold=True),
            overflow="ellipsis",
            no_wrap=True,
        )
        layout["header"].update(header)
        layout["body"].update(
            Group(
                Text(
                    "Small terminal: content may wrap. Scroll if needed.",
                    style=_style("warning"),
                ),
                body,
            )
        )
        layout["status"].update(status or footer or Text(_status_clock(), style=_style("dim")))
        return layout

    api_style = "bullish" if server_running else "dim"
    api_label = "API RUNNING" if server_running else "API STOPPED"
    error_label = f"  Errors:{error_count}" if error_count else ""
    header = Panel(
        Text.assemble(
            (
                " KAIROS ",
                f"{_style('highlight', bold=True)} on "
                f"{_BASIC_THEME['accent'] if _COLOR_SYSTEM in (None, 'standard') else THEME['accent']}",
            ),
            ("  ", _style("dim")),
            (asset, _style("highlight", bold=True)),
            ("  ", _style("dim")),
            (f"[{current_view.upper()}]", _style("accent", bold=True)),
            ("  |  ", _style("dim")),
            (api_label, _style(api_style, bold=server_running)),
            (error_label, _style("warning", bold=True)),
            ("  | ", _style("dim")),
            _header_help(),
        ),
        box=box.SQUARE,
        border_style=_style("panel_border"),
        padding=(0, 1),
        style=_panel_bg(),
    )

    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body", ratio=1),
        Layout(name="status", size=1),
    )
    layout["header"].update(header)
    layout["body"].update(body)
    layout["status"].update(status or footer or Text(_status_clock(), style=_style("dim")))
    return layout


def _status_clock() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S UTC")


def _status_bar(state: _DashboardState) -> Text:
    signal = state.last_metrics.get("signal", "signal pending")
    backtest = state.last_metrics.get("backtest", "Sharpe --")
    paper = state.last_metrics.get("paper", "Paper --")
    text = Text(style=f"{_style('highlight')} {_panel_bg()}".strip())
    text.append(f"[{state.current_view.upper()}] ", style=_style("accent", bold=True))
    text.append(f"{state.asset}: {signal}", style=_style("highlight"))
    text.append("  |  ", style=_style("dim"))
    text.append(f"Backtest: {backtest}", style=_style("neutral"))
    text.append("  |  ", style=_style("dim"))
    text.append(f"Paper: {paper}", style=_style("neutral"))
    text.append("  |  ", style=_style("dim"))
    text.append(_status_clock(), style=_style("dim"))
    return text


def _spinner(message: str) -> Panel:
    progress = Progress(
        SpinnerColumn(style=_style("accent", bold=True)),
        TextColumn(f"[{_style('accent', bold=True)}]{_markup(message)}"),
        console=console,
        transient=True,
    )
    progress.add_task(message, total=None)
    return Panel(
        Align.center(progress, vertical="middle"),
        title=f"[{_style('accent', bold=True)}]Refreshing",
        border_style=_style("panel_border"),
        style=_panel_bg(),
    )


def _with_overlay(body: Any, state: _DashboardState, loading_message: str | None = None) -> Any:
    parts: list[Any] = []
    now = time.monotonic()
    if state.toast and state.toast.expires_at > now:
        toast = Panel(
            Text(state.toast.message, style=state.toast.style, overflow="ellipsis"),
            box=box.SQUARE,
            border_style=state.toast.style,
            padding=(0, 1),
        )
        parts.append(Align.right(toast))
    elif state.toast:
        state.toast = None

    if loading_message:
        frame = _SPINNER_FRAMES[int(now * 12) % len(_SPINNER_FRAMES)]
        parts.append(Text(f"{frame} {loading_message}", style=_style("accent", bold=True)))

    parts.append(body)
    return Group(*parts) if len(parts) > 1 else body


def _error_panel(title: str, exc: BaseException) -> Panel:
    return Panel(
        Text(f"{type(exc).__name__}: {exc}", style=_style("warning", bold=True)),
        title=f"[{_style('warning', bold=True)}]{title}",
        border_style=_style("warning"),
        style=_panel_bg(),
    )


# -- View renderers -----------------------------------------------------------


def _capture_signal_panel(event: Any, current_price: float, fng_score: int, fng_ok: bool, ctx: dict[str, Any]) -> str:
    import kairos.live as live_module
    from kairos.live import display_signal as _display_signal

    buf = StringIO()
    capture_console = Console(file=buf, force_terminal=False, width=max(80, console.size.width - 6))
    original_console = live_module.console
    live_module.console = capture_console
    try:
        _display_signal(
            event,
            current_price,
            fng_score=fng_score,
            fng_available=fng_ok,
            price_context=ctx,
        )
    finally:
        live_module.console = original_console
    return buf.getvalue()


def _signal_metric(event: Any) -> str:
    if event.direction == "bullish":
        arrow = "↑"
    elif event.direction == "bearish":
        arrow = "↓"
    else:
        arrow = "→"
    return f"{arrow} {event.confidence:.0%} confidence"


def _render_signal(asset: str = "BTC") -> _ViewRender:
    from kairos.live import fetch_live_data, run_pipeline_with_context

    prices, current_price, fng_scores, fng_ok, volumes = asyncio.run(fetch_live_data(asset=asset))
    event, ctx = run_pipeline_with_context(prices, fng_scores, asset=asset)
    text = _capture_signal_panel(
        event,
        current_price,
        fng_scores[-1] if fng_ok else 50,
        fng_ok,
        ctx,
    )
    body = Panel(
        text,
        title=f"[{_style('bullish', bold=True)}]Live Signal - {asset}",
        border_style=_style(
            "bullish" if event.direction == "bullish" else "bearish" if event.direction == "bearish" else "neutral"
        ),
        style=_panel_bg(),
    )
    return _ViewRender(body=body, metric=_signal_metric(event))


def _build_signal_view(asset: str = "BTC") -> Panel:
    try:
        return _render_signal(asset).body
    except Exception as exc:
        return _error_panel("Signal", exc)


def _render_backtest(asset: str = "BTC", days: int = 365) -> _ViewRender:
    from kairos.backtest.engine import run_backtest
    from kairos.backtest.report import format_text_report

    result = run_backtest(asset=asset, days=days)
    report = format_text_report(result)
    body = Panel(
        report,
        title=f"[{_style('accent', bold=True)}]Backtest - {asset} ({days}d)",
        border_style=_style("accent"),
        style=_panel_bg(),
    )
    return _ViewRender(body=body, metric=f"Sharpe {result.sharpe:.2f}")


def _build_backtest_view(asset: str = "BTC", days: int = 365) -> Panel:
    try:
        return _render_backtest(asset, days).body
    except Exception as exc:
        return _error_panel("Backtest", exc)


def _render_compare(asset: str = "BTC", days: int = 365, capital: float = 10000.0) -> _ViewRender:
    from kairos.backtest.engine import run_backtest
    from kairos.live import _FNG_FALLBACK, fetch_live_data

    prices, _, fng_scores, fng_ok, volumes = asyncio.run(fetch_live_data(asset=asset))
    if not fng_ok:
        fng_scores = list(_FNG_FALLBACK)

    results = {}
    for sid in ("A", "B", "C"):
        try:
            results[sid] = run_backtest(
                asset=asset,
                days=days,
                initial_capital=capital,
                prices=prices,
                fng_scores=fng_scores,
                volumes=volumes,
                strategy=sid,
            )
        except Exception:
            continue

    if not results:
        raise RuntimeError("No strategies completed")

    bench_ret = (
        (results["A"].benchmark_curve[-1] / results["A"].initial_capital - 1) * 100
        if "A" in results and results["A"].benchmark_curve
        else 0.0
    )

    def sharpe_style(v: float) -> str:
        return _style("bullish") if v > 0.5 else _style("warning") if v > 0 else _style("bearish")

    def dd_style(v: float) -> str:
        return _style("bullish") if v < 20 else _style("warning") if v < 35 else _style("bearish")

    table = Table(
        show_header=True,
        header_style=_style("accent", bold=True),
        box=None,
        padding=(0, 3),
    )
    table.add_column("Metric", style=_style("dim"), no_wrap=True)
    for sid in ("A", "B", "C"):
        table.add_column(sid, justify="right", style=_style("highlight"))
    table.add_column("Benchmark", justify="right", style=_style("dim"))

    table.add_row(
        "Total Return",
        *[(f"{results[s].total_return_pct:+.2f}%" if s in results else "-") for s in "ABC"],
        f"{bench_ret:+.2f}%",
    )
    table.add_row(
        "Sharpe",
        *[(f"[{sharpe_style(results[s].sharpe)}]{results[s].sharpe:.2f}[/]" if s in results else "-") for s in "ABC"],
        "-",
    )
    table.add_row(
        "Max DD",
        *[
            (f"[{dd_style(results[s].max_drawdown_pct)}]{results[s].max_drawdown_pct:.1f}%[/]" if s in results else "-")
            for s in "ABC"
        ],
        "-",
    )
    table.add_row(
        "Win Rate",
        *[(f"{results[s].win_rate:.1f}%" if s in results else "-") for s in "ABC"],
        "-",
    )
    table.add_row(
        "Trades",
        *[(str(results[s].total_trades) if s in results else "-") for s in "ABC"],
        "-",
    )
    table.add_row(
        "Avg Kelly",
        *[(f"{results[s].avg_kelly_pct:.1f}%" if s in results else "-") for s in "ABC"],
        "-",
    )
    table.add_row(
        "Conflict Days",
        *[(f"{results[s].conflict_days}/{results[s].total_trading_days}" if s in results else "-") for s in "ABC"],
        "-",
    )
    capitulation = [results[s].capitulation_trades if s in results else 0 for s in "ABC"]
    if any(capitulation):
        table.add_row("Capitulation", *[str(c) for c in capitulation], "-")

    best = max(results, key=lambda s: results[s].total_return_pct)
    legend = Text()
    legend.append("\nA - Strict Defense  ", style=_style("dim"))
    legend.append("B - Capitulation  ", style=_style("dim"))
    legend.append("C - Exhaustion", style=_style("dim"))
    legend.append(
        f"\nBest: Strategy {best} ({_STRATEGIES[best]})",
        style=_style("bullish", bold=True),
    )

    body = Panel(
        Group(table, legend),
        title=f"[{_style('warning', bold=True)}]Strategy Comparison - {asset} ({days}d)",
        border_style=_style("warning"),
        style=_panel_bg(),
    )
    return _ViewRender(body=body, metric=f"Best {best} {results[best].total_return_pct:+.1f}%")


def _build_compare_view(asset: str = "BTC", days: int = 365, capital: float = 10000.0) -> Panel:
    try:
        return _render_compare(asset, days, capital).body
    except Exception as exc:
        return _error_panel("Compare", exc)


def _render_history(asset: str = "BTC", limit: int = 20) -> _ViewRender:
    from kairos.db import (
        create_schema,
        get_connection,
        get_hit_rate,
        get_signal_history,
    )

    conn = get_connection()
    try:
        create_schema(conn)
        signals = get_signal_history(conn, asset=asset, limit=limit)
        stats = get_hit_rate(conn, asset=asset)
    finally:
        conn.close()

    if not signals:
        body = Panel(
            Text("No signal history yet.", style=_style("dim")),
            title=f"[{_style('highlight', bold=True)}]Signal History - {asset}",
            border_style=_style("panel_border"),
            style=_panel_bg(),
        )
        return _ViewRender(body=body, metric="0 signals")

    table = Table(
        show_header=True,
        header_style=_style("dim", bold=True),
        box=None,
        padding=(0, 2),
    )
    table.add_column("Time", style=_style("dim"), no_wrap=True)
    table.add_column("Dir", style=_style("highlight", bold=True))
    table.add_column("Conf", justify="right")
    table.add_column("Regime")
    table.add_column("Price", justify="right")
    table.add_column("Outcome", justify="center")

    for sig in signals:
        triggered = sig["triggered_at"]
        ts = triggered.strftime("%m-%d %H:%M") if hasattr(triggered, "strftime") else str(triggered)[:16]
        direction = sig["direction"] or ""
        direction_style = "bullish" if direction == "bullish" else "bearish" if direction == "bearish" else "neutral"
        conf = f"{sig['confidence']:.0%}" if sig["confidence"] is not None else "-"
        price = f"${sig['price_at_signal']:,.0f}" if sig["price_at_signal"] is not None else "-"
        outcome = sig["outcome"] or "pending"
        outcome_style = "bullish" if outcome == "correct" else "bearish" if outcome == "incorrect" else "dim"
        table.add_row(
            ts,
            f"[{_style(direction_style, bold=True)}]{direction}[/]",
            conf,
            sig["regime"] or "-",
            price,
            f"[{_style(outcome_style)}]{outcome}[/]",
        )

    footer = Text()
    if stats["total_resolved"] > 0:
        hit = f"{stats['hit_rate']:.1%}" if stats["hit_rate"] is not None else "-"
        footer.append(
            f"Hit rate: {hit} ({stats['correct']}/{stats['total_resolved']})",
            style=_style("highlight", bold=True),
        )
        metric = f"Hit {hit}"
    else:
        footer.append("No resolved signals yet.", style=_style("dim"))
        metric = f"{len(signals)} pending"

    body = Panel(
        Group(table, footer),
        title=f"[{_style('highlight', bold=True)}]Signal History - {asset}",
        border_style=_style("panel_border"),
        style=_panel_bg(),
    )
    return _ViewRender(body=body, metric=metric)


def _build_history_view(asset: str = "BTC", limit: int = 20) -> Panel:
    try:
        return _render_history(asset, limit).body
    except Exception as exc:
        return _error_panel("History", exc)


def _sparkline(values: list[float], width: int = 42) -> str:
    if not values:
        return "-" * width
    blocks = "▁▂▃▄▅▆▇█"
    if len(values) > width:
        step = len(values) / width
        values = [values[int(i * step)] for i in range(width)]
    lo = min(values)
    hi = max(values)
    if hi <= lo:
        return blocks[0] * len(values)
    return "".join(blocks[int((value - lo) / (hi - lo) * (len(blocks) - 1))] for value in values)


def _ensure_paper_trading_engine(initial_capital: float = 10000.0):
    from kairos import live
    from kairos.papertrade import PaperTradingEngine

    if live._PAPER_TRADING_ENGINE is None:
        live._PAPER_TRADING_ENGINE = PaperTradingEngine(initial_capital=initial_capital)
    return live._PAPER_TRADING_ENGINE


def _render_paper(asset: str = "BTC", days: int = 90, capital: float = 10000.0) -> _ViewRender:
    engine = _ensure_paper_trading_engine(initial_capital=capital)
    account = engine.get_account(asset)
    equity = account.equity_curve[-1] if account.equity_curve else account.current_capital
    pnl = equity - account.initial_capital
    pnl_pct = (equity / account.initial_capital - 1) * 100 if account.initial_capital else 0.0
    pnl_style = "bullish" if pnl >= 0 else "bearish"
    open_trade = account.open_position
    wins = [trade for trade in account.trades if trade.pnl_pct is not None and trade.pnl_pct > 0]
    win_rate = len(wins) / len(account.trades) * 100 if account.trades else 0.0
    open_risk = open_trade.size * 100 if open_trade is not None else 0.0

    summary = Table.grid(padding=(0, 3))
    summary.add_column(style=_style("dim"))
    summary.add_column(style=_style("highlight", bold=True), justify="right")
    summary.add_row("Account Equity", f"${equity:,.2f}")
    summary.add_row(
        "Session P&L",
        f"[{_style(pnl_style, bold=True)}]{pnl:+,.2f} ({pnl_pct:+.2f}%)[/]",
    )
    summary.add_row("Open Risk", f"{open_risk:.1f}% live Kelly")
    summary.add_row("Trades", f"{len(account.trades)} total / {win_rate:.1f}% win")

    positions = Table(
        show_header=True,
        header_style=_style("accent", bold=True),
        box=None,
        padding=(0, 2),
    )
    positions.add_column("Asset")
    positions.add_column("Side")
    positions.add_column("Size", justify="right")
    positions.add_column("Entry", justify="right")
    positions.add_column("Status")
    if open_trade:
        side_style = "bullish" if open_trade.direction == "long" else "bearish"
        pnl_text = f"{(open_trade.pnl_pct or 0.0):+.2%}"
        positions.add_row(
            asset,
            f"[{_style(side_style, bold=True)}]{open_trade.direction.upper()}[/]",
            f"{open_trade.size:.1%}",
            f"${open_trade.entry_price:,.0f}",
            f"open {pnl_text}",
        )
    else:
        positions.add_row(asset, "FLAT", "0.0%", "-", "no open positions")

    curve = Text()
    curve.append("Equity curve  ", style=_style("dim"))
    curve.append(
        _sparkline([float(v) for v in account.equity_curve]),
        style=_style(pnl_style, bold=True),
    )

    body = Panel(
        Group(summary, Text(""), positions, Text(""), curve),
        title=f"[{_style('accent', bold=True)}]Paper Trading - {asset}",
        border_style=_style("accent"),
        style=_panel_bg(),
    )
    return _ViewRender(body=body, metric=f"{pnl_pct:+.1f}%")


def _build_paper_view(asset: str = "BTC") -> Panel:
    try:
        return _render_paper(asset).body
    except Exception as exc:
        return _error_panel("Paper Trading", exc)


def _render_serve(state: _DashboardState) -> _ViewRender:
    url = f"http://{state.server_host}:{state.server_port}"
    status = "running" if state.server_running else "stopped"
    status_style = "bullish" if state.server_running else "neutral"
    table = Table.grid(padding=(0, 3))
    table.add_column(style=_style("dim"))
    table.add_column(style=_style("highlight", bold=True))
    table.add_row("Status", f"[{_style(status_style, bold=True)}]{status.upper()}[/]")
    table.add_row("Endpoint", url if state.server_running else "-")
    table.add_row("Controls", "s toggle  |  E errors  |  q quit")
    body = Panel(
        table,
        title=f"[{_style('accent', bold=True)}]API Server",
        border_style=_style("bullish" if state.server_running else "panel_border"),
        style=_panel_bg(),
    )
    return _ViewRender(body=body, metric=f"API {status}")


def _render_error_log(state: _DashboardState) -> _ViewRender:
    if not state.error_log:
        content = Text("No TUI render errors recorded.", style=_style("dim"))
    else:
        content = Text()
        for row in state.error_log[-12:]:
            content.append(row + "\n", style=_style("warning"))
    body = Panel(
        content,
        title=f"[{_style('warning', bold=True)}]Error Log",
        border_style=_style("warning"),
        style=_panel_bg(),
    )
    return _ViewRender(body=body, metric=f"{len(state.error_log)} errors")


def _render_view(view: str, asset: str, state: _DashboardState) -> _ViewRender:
    if view == "signal":
        return _render_signal(asset)
    if view == "backtest":
        return _render_backtest(asset)
    if view == "compare":
        return _render_compare(asset)
    if view == "history":
        return _render_history(asset)
    if view == "paper":
        return _render_paper(asset)
    if view == "serve":
        return _render_serve(state)
    if view == "errors":
        return _render_error_log(state)
    raise ValueError(f"Unknown view: {view}")


# -- Server lifecycle ---------------------------------------------------------


def _start_server(state: _DashboardState) -> None:
    if state.server_process and state.server_process.poll() is None:
        state.server_running = True
        return
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "kairos.api.server:create_app",
        "--factory",
        "--host",
        state.server_host,
        "--port",
        str(state.server_port),
        "--log-level",
        "warning",
    ]
    state.server_process = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    state.server_running = True
    _toast(
        state,
        f"API server started at http://{state.server_host}:{state.server_port}",
        "bullish",
    )


def _stop_server(state: _DashboardState) -> None:
    proc = state.server_process
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2)
    state.server_running = False
    state.server_process = None
    _toast(state, "API server stopped", "neutral")


def _toggle_server(state: _DashboardState) -> None:
    if state.server_running:
        _stop_server(state)
    else:
        try:
            _start_server(state)
        except Exception as exc:
            state.server_running = False
            _record_error(state, "serve", exc)
    state.server_requested = False
    state.force_refresh = True


def _sync_server_state(state: _DashboardState) -> None:
    proc = state.server_process
    if not proc:
        state.server_running = False
        return
    if proc.poll() is None:
        state.server_running = True
        return
    state.server_running = False
    state.server_process = None
    _toast(state, "API server exited; check port 8000 or dependencies.", "warning")


# -- Terminal loop ------------------------------------------------------------


def _read_key(timeout: float = 0.1) -> str | None:
    if os.name == "nt":
        try:
            import msvcrt

            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if msvcrt.kbhit():
                    ch = msvcrt.getwch()
                    return ch
                time.sleep(0.01)
            return None
        except ImportError:
            return None

    try:
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
    except (OSError, ValueError):
        return None
    if not ready:
        return None
    key = sys.stdin.read(1)
    if key == "\x1b":
        seq = [key]
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 0.01)
            if not ready:
                break
            seq.append(sys.stdin.read(1))
            if len(seq) >= 4:
                break
        return "".join(seq)
    return key


def _ttl_for(view: str) -> float:
    if view in {"signal", "paper"}:
        return _REFRESH_INTERVAL
    if view in {"serve", "errors"}:
        return 1.0
    return _STATIC_REFRESH_INTERVAL


def _loading_message(view: str, asset: str) -> str:
    if view == "signal":
        return f"Refreshing {asset} signal..."
    if view == "paper":
        return f"Refreshing {asset} paper account..."
    if view == "backtest":
        return f"Refreshing {asset} backtest..."
    if view == "compare":
        return f"Refreshing {asset} strategy comparison..."
    if view == "history":
        return f"Refreshing {asset} signal history..."
    return "Refreshing dashboard..."


def _cache_stale(cache: dict[tuple[str, str], _ViewRender], key: tuple[str, str], view: str) -> bool:
    item = cache.get(key)
    if item is None:
        return True
    return (time.time() - item.updated_at) >= _ttl_for(view)


def main_tui(asset: str = "BTC") -> None:
    import termios
    import tty

    try:
        from kairos.live import install_fetch_signal_handler

        install_fetch_signal_handler()
    except Exception:
        pass
    _ensure_paper_trading_engine()

    state = _DashboardState(asset=asset)
    cache: dict[tuple[str, str], _ViewRender] = {}
    pending: tuple[concurrent.futures.Future[_ViewRender], tuple[str, str], str] | None = None
    last_frame: tuple[Any, ...] | None = None
    resize_due_at: float | None = None
    old_resize_handler: Any = None

    def _on_resize(_signum: int, _frame: Any) -> None:
        nonlocal resize_due_at
        resize_due_at = time.monotonic() + _RESIZE_DEBOUNCE_SECONDS

    if hasattr(signal, "SIGWINCH"):
        old_resize_handler = signal.getsignal(signal.SIGWINCH)
        signal.signal(signal.SIGWINCH, _on_resize)

    fd = sys.stdin.fileno()
    old_settings = None
    try:
        old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)
    except Exception:
        old_settings = None

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            with Live(
                auto_refresh=False,
                screen=True,
                redirect_stdout=False,
                console=console,
            ) as live:
                while state.running:
                    now = time.monotonic()
                    if resize_due_at and now >= resize_due_at:
                        state.resize_version += 1
                        resize_due_at = None

                    if state.server_requested:
                        _toggle_server(state)
                    _sync_server_state(state)

                    active_view = "errors" if state.show_error_log else state.current_view
                    if active_view == "errors":
                        render = _render_error_log(state)
                        cache[state.cache_key("errors")] = render
                        state.last_metrics["errors"] = render.metric
                    else:
                        key = state.cache_key(active_view)
                        should_refresh = state.force_refresh or _cache_stale(cache, key, active_view)
                        if should_refresh and pending is None:
                            state.force_refresh = False
                            pending = (
                                executor.submit(_render_view, active_view, state.asset, state),
                                key,
                                active_view,
                            )

                    if pending and pending[0].done():
                        future, key, view = pending
                        pending = None
                        try:
                            render = future.result()
                        except Exception as exc:
                            _record_error(state, view, exc)
                            if key in cache:
                                cache[key].updated_at = time.time()
                            else:
                                cache[key] = _ViewRender(
                                    body=_error_panel(view.title(), exc),
                                    metric=f"{view} error",
                                )
                        else:
                            cache[key] = render
                            state.last_metrics[view] = render.metric

                    active_view = "errors" if state.show_error_log else state.current_view
                    key = state.cache_key(active_view)
                    cached = cache.get(key)
                    loading = pending is not None and pending[1] == key
                    if cached:
                        body = cached.body
                    elif active_view == "serve":
                        body = _render_serve(state).body
                    elif active_view == "errors":
                        body = _render_error_log(state).body
                    else:
                        body = _spinner(_loading_message(active_view, state.asset))

                    body = _with_overlay(
                        body,
                        state,
                        _loading_message(active_view, state.asset) if loading and cached else None,
                    )
                    status = _status_bar(state)
                    layout = _make_layout(
                        active_view,
                        body,
                        asset=state.asset,
                        server_running=state.server_running,
                        status=status,
                        error_count=len(state.error_log),
                    )

                    frame = (
                        active_view,
                        state.asset,
                        state.server_running,
                        len(state.error_log),
                        state.resize_version,
                        status.plain,
                        id(cached.body) if cached else None,
                        pending[1] if pending else None,
                        int(now * 12) if loading else None,
                        state.toast.message if state.toast else None,
                    )
                    if frame != last_frame:
                        live.update(layout, refresh=True)
                        last_frame = frame

                    keypress = _read_key(0.1)
                    if keypress:
                        _key_handler(keypress, state)
                        last_frame = None

    finally:
        _stop_server(state)
        if old_settings is not None:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        if hasattr(signal, "SIGWINCH") and old_resize_handler is not None:
            signal.signal(signal.SIGWINCH, old_resize_handler)


def _check_tui() -> bool:
    """Check if we're in a real terminal with TUI support."""
    try:
        import termios  # noqa: F401
        import tty  # noqa: F401

        sys.stdin.fileno()
        return sys.stdout.isatty() and sys.stdin.isatty()
    except Exception:
        return False


# -- Entry point --------------------------------------------------------------


def run_cli() -> None:
    """CLI entry point. Starts TUI if in a terminal, falls back to one-shot."""
    import argparse

    parser = argparse.ArgumentParser(description="Kairos - sees what moves before price does.")
    parser.add_argument("--asset", "-a", default="BTC", help="Asset (BTC, ETH, SOL)")
    parser.add_argument("--no-tui", action="store_true", help="Force one-shot mode (no interactive TUI)")
    args = parser.parse_args()

    asset = args.asset.upper()
    from kairos.live import _ASSET_IDS

    if asset not in _ASSET_IDS:
        console.print(f"[{_style('bearish', bold=True)}]Unknown asset '{asset}'. Supported: {', '.join(_ASSET_IDS)}[/]")
        sys.exit(1)

    if args.no_tui or not _check_tui():
        _one_shot(asset)
    else:
        try:
            main_tui(asset=asset)
        except KeyboardInterrupt:
            pass
        except Exception as exc:
            console.print(f"[{_style('bearish', bold=True)}]TUI error:[/] {exc}")
            console.print(f"[{_style('dim')}]Falling back to one-shot mode...[/]")
            _one_shot(asset)


def _one_shot(asset: str) -> None:
    """Print signal once and exit (fallback when TUI unavailable or --no-tui)."""
    console.print(f"[{_style('dim')}]Kairos - {asset} one-shot signal[/]\n")
    try:
        from kairos.live import (
            display_signal,
            fetch_live_data,
            run_pipeline_with_context,
        )

        prices, current_price, fng_scores, fng_ok, volumes = asyncio.run(fetch_live_data(asset=asset))
        event, ctx = run_pipeline_with_context(prices, fng_scores, asset=asset)
        display_signal(
            event,
            current_price,
            fng_score=fng_scores[-1] if fng_ok else 50,
            fng_available=fng_ok,
            price_context=ctx,
        )
    except Exception as exc:
        console.print(f"[{_style('bearish', bold=True)}]Error:[/] {exc}")


if __name__ == "__main__":
    run_cli()
