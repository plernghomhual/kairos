import typer
import uvicorn
from rich.console import Console

app = typer.Typer(help="Kairos — Causal Economic Signal Engine")
console = Console()


@app.command()
def watch(
    asset: str = typer.Argument("BTC", help="Asset symbol to watch"),
):
    """Stream live signals for an asset to the terminal."""
    console.print(f"[bold green]Kairos[/bold green] watching [cyan]{asset}[/cyan]")
    console.print("Ingesting data... (run `kairos serve` to start the pipeline first)")


@app.command()
def backtest(
    asset: str = typer.Argument("BTC"),
    year: str = typer.Argument("2022"),
):
    """Run backtest against historical data and print hit rate."""
    console.print(f"[bold]Backtesting[/bold] {asset} for {year}...")
    console.print("[yellow]Backtest harness requires historical data in kairos.db[/yellow]")


@app.command()
def explain():
    """Explain the most recent signal — mechanism and citations."""
    console.print("[bold]Last signal explanation:[/bold]")
    console.print("Run `kairos serve` and query GET /signals/latest for details.")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
):
    """Start the local FastAPI signal server."""
    from kairos.api.server import create_app
    api = create_app()
    uvicorn.run(api, host=host, port=port)


if __name__ == "__main__":
    app()
