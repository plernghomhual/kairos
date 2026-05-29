import asyncio
import time

import typer
import uvicorn
from rich.console import Console

app = typer.Typer(
    help="Kairos — sees what moves before price does.\n\nStart here:  kairos run",
    no_args_is_help=True,
)
console = Console()


@app.command()
def run(
    watch: bool = typer.Option(False, "--watch", "-w", help="Keep refreshing every N seconds"),
    interval: int = typer.Option(300, "--interval", "-i", help="Seconds between refreshes (default 5 min)"),
    asset: str = typer.Option("BTC", "--asset", "-a", help="Asset to analyze (BTC, ETH, SOL)"),
):
    """
    Fetch live market data and show current signal.

    Uses CoinGecko (price) + Fear & Greed Index (sentiment). Supports BTC, ETH, SOL.

    Examples:
      kairos run              # one-shot signal (BTC)
      kairos run --asset eth  # one-shot signal (ETH)
      kairos run --asset sol  # one-shot signal (SOL)
      kairos run --watch      # refresh every 5 minutes
      kairos run -w -i 60     # refresh every 60 seconds
    """
    from kairos.live import fetch_live_data, run_pipeline_with_context, display_signal, _ASSET_IDS

    asset = asset.upper()
    if asset not in _ASSET_IDS:
        console.print(f"[red bold]Error:[/red bold] Unknown asset '{asset}'. Supported: {', '.join(_ASSET_IDS.keys())}")
        raise typer.Exit(code=1)

    def _once() -> None:
        console.print(f"[dim]Fetching live data for {asset} from CoinGecko + Fear & Greed Index...[/dim]")
        try:
            prices, current_price, fng_scores, fng_ok = asyncio.run(fetch_live_data(asset=asset))
            fng_note = f"Fear & Greed: {fng_scores[-1]}/100" if fng_ok else "sentiment unavailable"
            console.print(f"[dim]Got {len(prices)} price candles, {fng_note}[/dim]\n")
            event, ctx = run_pipeline_with_context(prices, fng_scores, asset=asset)
            display_signal(
                event, current_price,
                fng_score=fng_scores[-1], fng_available=fng_ok,
                price_context=ctx,
            )
            try:
                from kairos.db import get_connection, create_schema, save_signal, resolve_expired_signals
                conn = get_connection()
                create_schema(conn)
                resolve_expired_signals(conn, asset, current_price)
                save_signal(conn, event, current_price)
                conn.close()
            except Exception as db_err:
                console.print(f"[dim yellow]DB warning:[/dim yellow] {db_err}")
        except Exception as e:
            console.print(f"[red bold]Error:[/red bold] {e}")
            console.print("[dim]Check your internet connection, or CoinGecko may be rate-limiting.[/dim]")

    _once()

    if watch:
        console.print(f"[dim]Auto-refreshing every {interval}s — Ctrl+C to stop[/dim]")
        try:
            while True:
                time.sleep(interval)
                _once()
        except KeyboardInterrupt:
            console.print("\n[dim]Stopped.[/dim]")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Host to bind to"),
    port: int = typer.Option(8000, help="Port to listen on"),
):
    """
    Start the local REST API server.

    Endpoints:
      GET /health           — server status
      GET /signals          — last 20 signals
      GET /signals/latest   — most recent signal
    """
    from kairos.api.server import create_app
    console.print(f"[bold green]Kairos API[/bold green] running at http://{host}:{port}")
    console.print("[dim]Try: curl http://127.0.0.1:8000/health[/dim]\n")
    api = create_app()
    uvicorn.run(api, host=host, port=port)


@app.command()
def history(
    asset: str = typer.Option("BTC", "--asset", "-a", help="Asset to show history for"),
    limit: int = typer.Option(20, "--limit", "-n", help="Number of recent signals to show"),
):
    """
    Show recent signal history and hit-rate statistics.

    Displays a table of saved signals with their outcomes (if resolved),
    followed by overall hit-rate stats for the asset.

    Examples:
      kairos history              # last 20 BTC signals
      kairos history --asset eth  # last 20 ETH signals
      kairos history -n 10        # last 10 signals
    """
    from rich.table import Table
    from kairos.db import get_connection, create_schema, get_signal_history, get_hit_rate

    asset = asset.upper()
    conn = get_connection()
    create_schema(conn)

    signals = get_signal_history(conn, asset=asset, limit=limit)
    stats = get_hit_rate(conn, asset=asset)
    conn.close()

    if not signals:
        console.print(f"[dim]No signal history found for {asset}. Run [cyan]kairos run[/cyan] to generate signals.[/dim]")
        return

    table = Table(title=f"{asset} Signal History (last {len(signals)})", show_lines=False)
    table.add_column("Time (UTC)", style="dim", no_wrap=True)
    table.add_column("Dir", style="bold")
    table.add_column("Conf", justify="right")
    table.add_column("Regime")
    table.add_column("Hrs", justify="right")
    table.add_column("Price@Signal", justify="right")
    table.add_column("Price@Expiry", justify="right")
    table.add_column("Outcome", justify="center")

    for sig in signals:
        triggered = sig["triggered_at"]
        ts_str = triggered.strftime("%Y-%m-%d %H:%M") if hasattr(triggered, "strftime") else str(triggered)[:16]
        direction = sig["direction"] or ""
        dir_style = "green" if direction == "bullish" else "red" if direction == "bearish" else ""
        conf_str = f"{sig['confidence']:.0%}" if sig["confidence"] is not None else "—"
        hrs_str = f"{sig['estimated_hours']:.0f}h" if sig["estimated_hours"] is not None else "—"
        p_signal = f"${sig['price_at_signal']:,.0f}" if sig["price_at_signal"] is not None else "—"
        p_expiry = f"${sig['price_at_expiry']:,.0f}" if sig["price_at_expiry"] is not None else "—"
        outcome = sig["outcome"] or "pending"
        outcome_style = "green" if outcome == "correct" else "red" if outcome == "incorrect" else "dim"

        table.add_row(
            ts_str,
            f"[{dir_style}]{direction}[/{dir_style}]" if dir_style else direction,
            conf_str,
            sig["regime"] or "—",
            hrs_str,
            p_signal,
            p_expiry,
            f"[{outcome_style}]{outcome}[/{outcome_style}]",
        )

    console.print(table)

    # Hit-rate summary
    console.print()
    if stats["total_resolved"] == 0:
        console.print(f"[dim]No resolved signals yet for {asset}. Outcomes update after estimated_hours elapses.[/dim]")
    else:
        hit_pct = f"{stats['hit_rate']:.1%}" if stats["hit_rate"] is not None else "—"
        console.print(
            f"[bold]Hit rate:[/bold] {hit_pct}  "
            f"({stats['correct']} correct / {stats['total_resolved']} resolved)"
        )


@app.command()
def explain():
    """Show what the last signal means in plain English."""
    console.print("[bold]How to see the last signal:[/bold]")
    console.print("  1. Run [cyan]kairos run[/cyan] — prints it right here")
    console.print("  2. Or start [cyan]kairos serve[/cyan] then:")
    console.print("     [dim]curl http://127.0.0.1:8000/signals/latest[/dim]")


if __name__ == "__main__":
    app()
