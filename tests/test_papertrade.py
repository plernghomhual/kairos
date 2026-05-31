"""Tests for live paper trading state and persistence."""

from datetime import datetime, timezone

import pytest

from kairos.backtest.engine import _kelly_fraction
from kairos.models.signal_event import SignalEvent
from kairos.papertrade import PaperTradingEngine


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
    engine = PaperTradingEngine(initial_capital=10_000.0)
    engine.process_signal(_event("bullish", event_id="long"), current_price=100.0)

    opened = engine.process_signal(_event("bearish", event_id="short"), current_price=95.0)

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
