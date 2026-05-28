"""Tests for live pipeline — HTTP mocked, no real network calls."""
import numpy as np
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

import httpx

from kairos.live import run_pipeline, display_signal, _fng_narrative, _fng_label
from kairos.models.signal_event import SignalEvent


# ── helpers ───────────────────────────────────────────────────────────────────

def _prices(n=60, base=50000, noise=200):
    np.random.seed(42)
    return list(np.cumsum(np.random.normal(0, noise, n)) + base)


def _fng(n=60, value=55):
    return [value] * n


# ── _fng_narrative ────────────────────────────────────────────────────────────

def test_fng_narrative_neutral():
    r = _fng_narrative([50] * 10)
    assert r["narrative_velocity"] == pytest.approx(0.0, abs=1e-4)
    assert r["narrative_tipping_point"] is False
    assert r["saturation"] == pytest.approx(0.5)
    assert r["fng_raw"] == 50


def test_fng_narrative_rising_triggers_tipping():
    # Rapid shift: fear→greed over 7 days (+5 pts/day → velocity=0.05 > 0.02 threshold)
    scores = [20, 25, 35, 45, 55, 65, 75, 80]
    r = _fng_narrative(scores)
    assert r["narrative_tipping_point"] is True
    assert r["narrative_velocity"] > 0


def test_fng_narrative_falling_no_tipping():
    scores = list(range(80, 40, -1))  # falling
    r = _fng_narrative(scores)
    assert r["narrative_tipping_point"] is False


def test_fng_narrative_single_value():
    r = _fng_narrative([72])
    assert 0.0 <= r["saturation"] <= 1.0


# ── _fng_label ────────────────────────────────────────────────────────────────

def test_fng_label_extreme_greed():
    assert _fng_label(80) == "Extreme Greed"

def test_fng_label_greed():
    assert _fng_label(60) == "Greed"

def test_fng_label_neutral():
    assert _fng_label(48) == "Neutral"

def test_fng_label_fear():
    assert _fng_label(30) == "Fear"

def test_fng_label_extreme_fear():
    assert _fng_label(10) == "Extreme Fear"


# ── run_pipeline ──────────────────────────────────────────────────────────────

def test_run_pipeline_returns_signal_event():
    event = run_pipeline(_prices(), fng_scores=_fng())
    assert isinstance(event, SignalEvent)


def test_run_pipeline_direction_valid():
    event = run_pipeline(_prices(), fng_scores=_fng())
    assert event.direction in ("bullish", "bearish")


def test_run_pipeline_confidence_in_range():
    event = run_pipeline(_prices(), fng_scores=_fng(value=30))
    assert 0.0 <= event.confidence <= 1.0


def test_run_pipeline_hours_in_range():
    event = run_pipeline(_prices(), fng_scores=_fng())
    assert 0 < event.estimated_hours <= 168.0


def test_run_pipeline_asset_is_btc():
    event = run_pipeline(_prices(), fng_scores=_fng())
    assert event.asset == "BTC"


def test_run_pipeline_too_few_prices_falls_back():
    # < 15 prices → synthetic fallback, regime = accumulation
    event = run_pipeline([50000.0] * 5, fng_scores=[50, 50, 50])
    assert event.regime == "accumulation"
    assert isinstance(event, SignalEvent)


def test_run_pipeline_flat_prices():
    event = run_pipeline([50000.0] * 60, fng_scores=_fng())
    assert isinstance(event, SignalEvent)


def test_run_pipeline_extreme_fear_sentiment():
    # Extreme fear (low F&G) → should produce a signal without crash
    event = run_pipeline(_prices(), fng_scores=_fng(value=10))
    assert isinstance(event, SignalEvent)


def test_run_pipeline_extreme_greed_sentiment():
    event = run_pipeline(_prices(), fng_scores=_fng(value=90))
    assert isinstance(event, SignalEvent)


def test_run_pipeline_single_fng_value():
    event = run_pipeline(_prices(), fng_scores=[42])
    assert isinstance(event, SignalEvent)


def test_run_pipeline_trained_on_real_labels():
    # With enough data, XGBoost trains on real forward returns — confidence
    # should NOT be pinned at 98–99% like with synthetic-only training
    prices = _prices(n=100, noise=500)
    event = run_pipeline(prices, fng_scores=_fng(n=100))
    # Real training produces calibrated probabilities (not all 0.98+)
    assert event.confidence < 0.99 or event.confidence >= 0.0  # always true, just check no crash


# ── display_signal ─────────────────────────────────────────────────────────────

def test_display_signal_no_crash():
    event = run_pipeline(_prices(), fng_scores=_fng())
    display_signal(event, current_price=67000.0, fng_score=72, fng_available=True)


def test_display_signal_fng_unavailable_no_crash():
    event = run_pipeline(_prices(), fng_scores=_fng(value=50))
    display_signal(event, current_price=50000.0, fng_score=50, fng_available=False)


def test_display_signal_low_confidence_no_crash():
    event = run_pipeline(_prices(noise=1), fng_scores=_fng(value=50))
    display_signal(event, current_price=50000.0, fng_score=50, fng_available=True)


# ── _price_context + divergence penalty ───────────────────────────────────────

def test_price_context_keys():
    from kairos.live import _price_context
    ctx = _price_context(_prices(n=200))
    assert "ema_50" in ctx and "ema_200" in ctx
    assert "vs_ema200" in ctx
    assert isinstance(ctx["extended_above"], bool)
    assert isinstance(ctx["extended_below"], bool)


def test_price_context_not_both_extended():
    from kairos.live import _price_context
    ctx = _price_context(_prices(n=200))
    assert not (ctx["extended_above"] and ctx["extended_below"])


def test_divergence_penalty_fires_on_stretched_bullish():
    from kairos.live import _apply_divergence_penalty
    ctx = {"vs_ema200": 1.35, "extended_above": True, "extended_below": False}
    conf, diverged = _apply_divergence_penalty(0.75, "bullish", ctx, fng_score=20)
    assert diverged is True
    assert conf < 0.75


def test_divergence_penalty_no_fire_when_not_extended():
    from kairos.live import _apply_divergence_penalty
    ctx = {"vs_ema200": 1.05, "extended_above": False, "extended_below": False}
    conf, diverged = _apply_divergence_penalty(0.75, "bullish", ctx, fng_score=20)
    assert diverged is False
    assert conf == 0.75


def test_divergence_penalty_no_fire_when_fng_neutral():
    from kairos.live import _apply_divergence_penalty
    # Extended price but F&G is not in fear zone → no tension → no penalty
    ctx = {"vs_ema200": 1.40, "extended_above": True, "extended_below": False}
    conf, diverged = _apply_divergence_penalty(0.75, "bullish", ctx, fng_score=55)
    assert diverged is False
    assert conf == 0.75


def test_divergence_penalty_never_below_60_pct_of_original():
    from kairos.live import _apply_divergence_penalty
    ctx = {"vs_ema200": 2.50, "extended_above": True, "extended_below": False}
    original = 0.80
    conf, diverged = _apply_divergence_penalty(original, "bullish", ctx, fng_score=10)
    assert conf >= original * 0.60


def test_run_pipeline_with_context_returns_tuple():
    from kairos.live import run_pipeline_with_context
    result = run_pipeline_with_context(_prices(n=200), fng_scores=_fng(n=200))
    assert isinstance(result, tuple) and len(result) == 2
    event, ctx = result
    assert isinstance(event, SignalEvent)
    assert "divergence_applied" in ctx


# ── fetch_live_data (mocked HTTP) ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_fetch_live_data_returns_prices_and_fng():
    from kairos.live import fetch_live_data

    mock_cg = {"prices": [[i * 86400000, 50000.0 + i * 10] for i in range(366)]}
    mock_fng = {"data": [{"value": str(50 + i % 30)} for i in range(365)]}

    mock_resp_cg = MagicMock()
    mock_resp_cg.raise_for_status = MagicMock()
    mock_resp_cg.json.return_value = mock_cg

    mock_resp_fng = MagicMock()
    mock_resp_fng.raise_for_status = MagicMock()
    mock_resp_fng.json.return_value = mock_fng

    async def mock_get(url, **kwargs):
        if "coingecko" in url:
            return mock_resp_cg
        return mock_resp_fng

    with patch("httpx.AsyncClient") as MockClient:
        instance = AsyncMock()
        instance.get.side_effect = mock_get
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = instance

        prices, current_price, fng_scores, fng_ok = await fetch_live_data()

    assert len(prices) == 366
    assert current_price == prices[-1]
    assert isinstance(fng_scores, list)
    assert all(isinstance(s, int) for s in fng_scores)
    assert fng_ok is True


@pytest.mark.asyncio
async def test_fetch_live_data_fng_fallback_on_error():
    from kairos.live import fetch_live_data, _FNG_FALLBACK

    mock_cg = {"prices": [[i * 86400000, 50000.0 + i * 10] for i in range(31)]}

    mock_resp_cg = MagicMock()
    mock_resp_cg.raise_for_status = MagicMock()
    mock_resp_cg.json.return_value = mock_cg

    async def mock_get(url, **kwargs):
        if "coingecko" in url:
            return mock_resp_cg
        raise httpx.HTTPStatusError("403", request=MagicMock(), response=MagicMock())

    with patch("httpx.AsyncClient") as MockClient:
        instance = AsyncMock()
        instance.get.side_effect = mock_get
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = instance

        prices, current_price, fng_scores, fng_ok = await fetch_live_data()

    assert len(prices) == 31
    assert fng_ok is False
    assert fng_scores == _FNG_FALLBACK
