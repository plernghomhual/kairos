"""Tests for live paper trading state and persistence."""

from datetime import datetime, timedelta, timezone

import pytest

from kairos.backtest.engine import _kelly_fraction
from kairos.models.signal_event import SignalEvent
from kairos.papertrade import PaperPosition, PaperTradingEngine, _should_exit


def _event(
    direction: str,
    asset: str = "BTC",
    confidence: float = 0.75,
    regime: str = "lv_up",
    event_id: str | None = None,
) -> SignalEvent:
    return SignalEvent(
        asset=asset,
        direction=direction,
        confidence=confidence,
        regime=regime,
        narrative_velocity=0.1,
        narrative_tipping_point=False,
        mechanism="test signal",
        estimated_hours=24.0,
        citations=[],
        id=event_id or f"{asset}-{direction}-{confidence}",
        triggered_at=datetime.now(timezone.utc),
    )


def test_process_signal_opens_position():
    engine = PaperTradingEngine(initial_capital=10_000.0)

    position = engine.process_signal(_event("bullish"), current_price=100.0)

    assert position is not None
    assert position.direction == "long"
    assert position.entry_price == pytest.approx(100.10)
    account = engine.get_account("BTC")
    assert account.open_position == position
    assert account.trades == []


def test_process_signal_closes_position():
    engine = PaperTradingEngine(initial_capital=10_000.0)
    engine.process_signal(_event("bullish", event_id="open-long"), current_price=100.0)

    closed = engine.process_signal(_event("neutral", event_id="close-long"), current_price=110.0)

    assert closed is not None
    assert closed.closed is True
    assert closed.exit_price == pytest.approx(109.89)
    assert closed.pnl_pct is not None and closed.pnl_pct > 0
    account = engine.get_account("BTC")
    assert account.open_position is None
    assert account.trades == [closed]
    assert account.current_capital > account.initial_capital


def test_process_signal_flips_direction():
    """Same-regime direction flip → exit gate holds the position."""
    engine = PaperTradingEngine(initial_capital=10_000.0)
    engine.process_signal(_event("bullish", event_id="long"), current_price=100.0)

    # Same regime (lv_up → lv_up), no ATR breach → gate blocks flip
    opened = engine.process_signal(_event("bearish", event_id="short"), current_price=95.0)

    account = engine.get_account("BTC")
    assert opened is None
    assert account.open_position is not None
    assert account.open_position.direction == "long"
    assert len(account.trades) == 0


def test_process_signal_flips_on_regime_change():
    """Direction flip WITH regime change → exit gate allows flip."""
    engine = PaperTradingEngine(initial_capital=10_000.0)
    engine.process_signal(_event("bullish", event_id="long", regime="lv_up"), current_price=100.0)

    opened = engine.process_signal(_event("bearish", event_id="short", regime="lv_down"), current_price=95.0)

    account = engine.get_account("BTC")
    assert opened is not None
    assert opened.direction == "short"
    assert account.open_position == opened
    assert len(account.trades) == 1
    assert account.trades[0].direction == "long"
    assert account.trades[0].closed is True


def test_update_price_mtm():
    engine = PaperTradingEngine(initial_capital=10_000.0)
    position = engine.process_signal(_event("bullish"), current_price=100.0)
    assert position is not None

    engine.update_price("BTC", position.entry_price * 1.10)

    account = engine.get_account("BTC")
    assert account.open_position is position
    assert position.pnl_pct == pytest.approx(0.10)
    assert account.equity_curve[-1] == pytest.approx(account.current_capital * (1 + position.size * 0.10))


def test_kelly_sizing():
    engine = PaperTradingEngine(initial_capital=10_000.0)

    position = engine.process_signal(_event("bullish", confidence=0.75), current_price=100.0)

    assert position is not None
    assert position.size == pytest.approx(_kelly_fraction(0.75))


def test_multiple_assets():
    engine = PaperTradingEngine(initial_capital=10_000.0)

    btc = engine.process_signal(_event("bullish", asset="BTC"), current_price=100.0)
    eth = engine.process_signal(_event("bearish", asset="ETH", regime="lv_down"), current_price=2_000.0)

    assert btc is not None and btc.direction == "long"
    assert eth is not None and eth.direction == "short"
    assert engine.get_account("BTC").open_position == btc
    assert engine.get_account("ETH").open_position == eth


def test_persistence_save_load(tmp_path):
    db_path = tmp_path / "paper.duckdb"
    engine = PaperTradingEngine(initial_capital=10_000.0, db_path=str(db_path))
    engine.process_signal(_event("bullish", event_id="persist-long"), current_price=100.0)
    engine.process_signal(_event("neutral", event_id="persist-close"), current_price=110.0)
    engine.close()

    reloaded = PaperTradingEngine(initial_capital=10_000.0, db_path=str(db_path))
    account = reloaded.get_account("BTC")

    assert account.open_position is None
    assert len(account.trades) == 1
    assert account.trades[0].signal_id == "persist-long"
    assert account.trades[0].closed is True
    assert account.current_capital > account.initial_capital
    reloaded.close()


def test_reload_preserves_open_position_entry_regime(tmp_path):
    db_path = tmp_path / "paper.duckdb"
    engine = PaperTradingEngine(initial_capital=10_000.0, db_path=str(db_path))
    engine.process_signal(_event("bullish", event_id="persist-open-hv", regime="hv_down"), current_price=100.0)
    engine.close()

    reloaded = PaperTradingEngine(initial_capital=10_000.0, db_path=str(db_path))
    account = reloaded.get_account("BTC")

    assert account.open_position is not None
    assert account.open_position.entry_regime == "hv_down"
    reloaded.close()


def test_failed_close_persist_does_not_corrupt_memory(monkeypatch):
    engine = PaperTradingEngine(initial_capital=10_000.0)
    position = engine.process_signal(_event("bullish", event_id="rollback-open", regime="lv_up"), current_price=100.0)
    assert position is not None

    def fail_persist_close(_position):
        raise RuntimeError("database write failed")

    monkeypatch.setattr(engine, "_persist_close", fail_persist_close)

    with pytest.raises(RuntimeError, match="database write failed"):
        engine.process_signal(_event("bearish", event_id="rollback-flip", regime="lv_down"), current_price=95.0)

    account = engine.get_account("BTC")
    assert account.open_position is position
    assert account.trades == []
    assert position.closed is False
    assert position.exit_price is None
    assert position.exit_time is None
    assert position.pnl_pct is None


def test_live_context_includes_paper_account(monkeypatch):
    from kairos import live

    engine = PaperTradingEngine(initial_capital=10_000.0)
    monkeypatch.setattr(live, "_PAPER_TRADING_ENGINE", engine)
    monkeypatch.setattr(
        live,
        "run_pipeline",
        lambda prices, fng_scores, asset="BTC", volumes=None: _event(
            "bullish", asset=asset, regime="lv_up", event_id="live-long"
        ),
    )

    prices = [100.0 + i for i in range(200)]
    event, ctx = live.run_pipeline_with_context(prices, [30] * 200, asset="BTC")

    assert event.direction == "bullish"
    assert "paper_account" in ctx
    assert ctx["paper_account"].open_position is not None


def test_capital_floor_prevents_negative_capital():
    """A total loss must not produce negative capital."""
    engine = PaperTradingEngine(initial_capital=10_000.0)
    engine.process_signal(_event("bullish", event_id="loss-open"), current_price=100.0)
    # Exit at near-zero price to force maximum loss
    closed = engine.process_signal(_event("neutral", event_id="loss-close"), current_price=0.01)
    assert closed is not None
    account = engine.get_account("BTC")
    assert account.current_capital >= 0.0


def test_persist_close_idempotent(tmp_path):
    """Calling _persist_close twice on the same position must not raise or duplicate rows."""
    db_path = str(tmp_path / "paper.db")
    engine = PaperTradingEngine(initial_capital=10_000.0, db_path=db_path)
    engine.process_signal(_event("bullish", event_id="idem-open"), current_price=100.0)
    engine.process_signal(_event("neutral", event_id="idem-close"), current_price=110.0)

    account = engine.get_account("BTC")
    assert len(account.trades) == 1
    closed_pos = account.trades[0]

    engine._persist_close(closed_pos)  # second call — must not raise

    rows = engine._conn.execute(
        "SELECT COUNT(*) FROM paper_trades WHERE signal_id = ? AND closed = TRUE",
        ["idem-open"],
    ).fetchone()
    assert rows[0] == 1
    engine.close()


def test_hold_hours_uses_wall_clock():
    """Exit gate must fire based on wall-clock elapsed time, not signal timestamp."""
    old_entry = datetime.now(timezone.utc) - timedelta(hours=100)
    position = PaperPosition(
        asset="BTC",
        direction="long",
        entry_price=100.0,
        entry_time=old_entry,
        size=0.1,
        signal_id="hold-test",
        entry_regime="lv_up",
    )
    event = SignalEvent(
        asset="BTC",
        direction="bullish",
        confidence=0.7,
        regime="lv_up",
        narrative_velocity=0.1,
        narrative_tipping_point=False,
        mechanism="test",
        estimated_hours=24.0,
        citations=[],
    )
    # 100h elapsed >> 24h max_hold → exit gate must trigger
    assert _should_exit(position, event, current_price=100.0, price_history=[]) is True
