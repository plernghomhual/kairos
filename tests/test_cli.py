from datetime import datetime, timezone

from typer.testing import CliRunner

from kairos.cli import app
from kairos.models.signal_event import SignalEvent
from kairos.papertrade import PaperTradingEngine

runner = CliRunner()


def test_help_exits_cleanly():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.output


def test_run_help_shows_no_api_keys():
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "API" in result.output or "CoinGecko" in result.output


def test_unknown_command_exits_nonzero():
    result = runner.invoke(app, ["nonexistent"])
    assert result.exit_code != 0


def test_paper_view_uses_live_paper_engine(monkeypatch):
    from kairos import cli, live

    engine = PaperTradingEngine()
    engine.process_signal(
        SignalEvent(
            asset="BTC",
            direction="bullish",
            confidence=0.75,
            regime="lv_up",
            narrative_velocity=0.0,
            narrative_tipping_point=False,
            mechanism="test",
            estimated_hours=72.0,
            citations=[],
            triggered_at=datetime.now(timezone.utc),
        ),
        current_price=100.0,
    )
    monkeypatch.setattr(live, "_PAPER_TRADING_ENGINE", engine)

    def fail_backtest(*args, **kwargs):
        raise AssertionError

    monkeypatch.setattr(
        "kairos.backtest.engine.run_backtest",
        fail_backtest,
    )

    render = cli._render_paper("BTC")

    with cli.console.capture() as capture:
        cli.console.print(render.body)

    assert "LONG" in capture.get()
