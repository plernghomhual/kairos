"""Tests for multi-asset support (ETH, SOL) in the live pipeline."""
import asyncio
import numpy as np
import pytest

from kairos.live import run_pipeline, run_pipeline_with_context
from kairos.models.signal_event import SignalEvent


def _prices(n=60):
    np.random.seed(42)
    return list(np.cumsum(np.random.normal(0, 200, n)) + 50000)


def _fng(n=60):
    return [55] * n


def test_run_pipeline_eth():
    event = run_pipeline(_prices(), _fng(), asset="ETH")
    assert isinstance(event, SignalEvent)
    assert event.asset == "ETH"


def test_run_pipeline_sol():
    event = run_pipeline(_prices(), _fng(), asset="SOL")
    assert isinstance(event, SignalEvent)
    assert event.asset == "SOL"


def test_run_pipeline_unknown_asset_passes_through():
    # run_pipeline doesn't validate asset — just passes string to SignalEvent
    event = run_pipeline(_prices(), _fng(), asset="DOGE")
    assert event.asset == "DOGE"


def test_fetch_live_data_unknown_asset_raises():
    from kairos.live import fetch_live_data
    with pytest.raises(ValueError, match="Unsupported asset"):
        asyncio.run(fetch_live_data(asset="DOGE"))


def test_run_pipeline_with_context_eth():
    event, ctx = run_pipeline_with_context(_prices(n=200), _fng(n=200), asset="ETH")
    assert isinstance(event, SignalEvent)
    assert event.asset == "ETH"
    assert "divergence_applied" in ctx


def test_run_pipeline_btc_default_unchanged():
    event = run_pipeline(_prices(), _fng())
    assert event.asset == "BTC"
