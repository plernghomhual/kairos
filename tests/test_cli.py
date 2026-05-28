from typer.testing import CliRunner
from kairos.cli import app

runner = CliRunner()


def test_help_exits_cleanly():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "watch" in result.output


def test_unknown_command_exits_nonzero():
    result = runner.invoke(app, ["nonexistent"])
    assert result.exit_code != 0
