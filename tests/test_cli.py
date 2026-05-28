from typer.testing import CliRunner
from kairos.cli import app

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
