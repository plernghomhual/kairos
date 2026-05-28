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
):
    """
    Fetch live BTC data and show current market signal.

    Uses CoinGecko (price) + Reddit (sentiment). No API keys needed.

    Examples:
      kairos run              # one-shot signal
      kairos run --watch      # refresh every 5 minutes
      kairos run -w -i 60     # refresh every 60 seconds
    """
    from kairos.live import fetch_live_data, run_pipeline, display_signal

    def _once() -> None:
        console.print("[dim]Fetching live data from CoinGecko + Reddit...[/dim]")
        try:
            prices, current_price, reddit_counts = asyncio.run(fetch_live_data())
            console.print(f"[dim]Got {len(prices)} price candles, {sum(reddit_counts)} Reddit posts[/dim]\n")
            event = run_pipeline(prices, reddit_counts)
            display_signal(event, current_price)
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
def explain():
    """Show what the last signal means in plain English."""
    console.print("[bold]How to see the last signal:[/bold]")
    console.print("  1. Run [cyan]kairos run[/cyan] — prints it right here")
    console.print("  2. Or start [cyan]kairos serve[/cyan] then:")
    console.print("     [dim]curl http://127.0.0.1:8000/signals/latest[/dim]")


if __name__ == "__main__":
    app()
