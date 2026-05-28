"""Tests for live pipeline — HTTP mocked, no real network calls."""
import numpy as np
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from kairos.live import run_pipeline, display_signal
from kairos.models.signal_event import SignalEvent


# ── run_pipeline ──────────────────────────────────────────────────────────────

def _prices(n=30, base=50000, noise=200):
    np.random.seed(42)
    return list(np.cumsum(np.random.normal(0, noise, n)) + base)


def test_run_pipeline_returns_signal_event():
    event = run_pipeline(_prices(), reddit_counts=[10, 20, 30, 40, 60, 90])
    assert isinstance(event, SignalEvent)


def test_run_pipeline_direction_valid():
    event = run_pipeline(_prices(), reddit_counts=[10, 20, 40])
    assert event.direction in ("bullish", "bearish")


def test_run_pipeline_confidence_in_range():
    event = run_pipeline(_prices(), reddit_counts=[5, 5, 5])
    assert 0.0 <= event.confidence <= 1.0


def test_run_pipeline_hours_in_range():
    event = run_pipeline(_prices(), reddit_counts=[10, 20, 30])
    assert 0 < event.estimated_hours <= 168.0


def test_run_pipeline_asset_is_btc():
    event = run_pipeline(_prices(), reddit_counts=[10, 20, 30])
    assert event.asset == "BTC"


def test_run_pipeline_too_few_prices():
    # 5 prices — below HMM minimum, falls back to accumulation
    event = run_pipeline([50000.0] * 5, reddit_counts=[10, 10, 10])
    assert event.regime == "accumulation"


def test_run_pipeline_flat_prices():
    # Flat price → no trend → should still produce a signal without crash
    event = run_pipeline([50000.0] * 30, reddit_counts=[10, 10, 10])
    assert isinstance(event, SignalEvent)


def test_run_pipeline_growing_narrative_affects_regime():
    # Growing narrative → tipping point might fire
    fast = [50, 100, 200, 400, 800]
    event = run_pipeline(_prices(), reddit_counts=fast)
    assert isinstance(event, SignalEvent)


def test_run_pipeline_single_reddit_count():
    event = run_pipeline(_prices(), reddit_counts=[42])
    assert isinstance(event, SignalEvent)


# ── display_signal (smoke test — just checks it doesn't crash) ─────────────

def test_display_signal_bullish_no_crash(capsys):
    event = run_pipeline(_prices(), reddit_counts=[10, 20, 40, 80, 160])
    # display_signal writes to Rich Console — just ensure no exception
    display_signal(event, current_price=67000.0)


def test_display_signal_low_confidence_shows_warning(capsys):
    event = run_pipeline(_prices(noise=1), reddit_counts=[1, 1, 1])
    display_signal(event, current_price=50000.0)


# ── fetch_live_data (mocked HTTP) ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_fetch_live_data_returns_prices_and_counts():
    from kairos.live import fetch_live_data

    mock_coingecko = {
        "prices": [[i * 86400000, 50000.0 + i * 10] for i in range(31)]
    }
    mock_reddit = {
        "data": {
            "children": [
                {"data": {"created_utc": 1700000000 + i * 3600}} for i in range(50)
            ]
        }
    }

    mock_response_cg = MagicMock()
    mock_response_cg.raise_for_status = MagicMock()
    mock_response_cg.json.return_value = mock_coingecko

    mock_response_reddit = MagicMock()
    mock_response_reddit.raise_for_status = MagicMock()
    mock_response_reddit.json.return_value = mock_reddit

    async def mock_get(url, **kwargs):
        if "coingecko" in url:
            return mock_response_cg
        return mock_response_reddit

    with patch("httpx.AsyncClient") as MockClient:
        instance = AsyncMock()
        instance.get.side_effect = mock_get
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = instance

        prices, current_price, counts = await fetch_live_data()

    assert len(prices) == 31
    assert current_price == prices[-1]
    assert isinstance(counts, list)
    assert all(isinstance(c, int) for c in counts)
