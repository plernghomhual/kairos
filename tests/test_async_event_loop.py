"""Async supervision and sync/async runner tests."""

import asyncio

import pytest


def test_run_async_from_sync():
    from kairos.backtest.engine import _run_async

    async def returns_value():
        return "ok"

    assert _run_async(returns_value(), timeout=1.0) == "ok"


@pytest.mark.asyncio
async def test_run_async_from_async_context():
    from kairos.backtest.engine import _run_async

    async def returns_value():
        return "ok"

    assert await _run_async(returns_value(), timeout=1.0) == "ok"


def test_run_async_timeout_from_sync():
    from kairos.backtest.engine import _run_async

    async def hangs():
        await asyncio.sleep(1.0)

    with pytest.raises(asyncio.TimeoutError):
        _run_async(hangs(), timeout=0.01)
