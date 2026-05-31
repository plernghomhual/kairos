import asyncio

import pytest

from kairos.circuit_breaker import (
    CIRCUIT_BREAKERS,
    CircuitBreaker,
    CircuitBreakerState,
    get_health_summary,
)


@pytest.mark.asyncio
async def test_circuit_breaker_opens_after_threshold():
    breaker = CircuitBreaker("unit", failure_threshold=5, recovery_timeout=10.0)

    async def failing_call():
        raise RuntimeError("api down")

    for _ in range(5):
        result = await breaker.call(failing_call, fallback="fallback")
        assert result == "fallback"

    assert breaker.get_state() is CircuitBreakerState.OPEN


@pytest.mark.asyncio
async def test_circuit_breaker_rejects_when_open():
    breaker = CircuitBreaker("unit", failure_threshold=1, recovery_timeout=10.0)
    calls = 0

    async def failing_call():
        raise RuntimeError("api down")

    async def should_not_run():
        nonlocal calls
        calls += 1
        return "live"

    await breaker.call(failing_call, fallback="fallback")
    result = await breaker.call(should_not_run, fallback="fallback")

    assert result == "fallback"
    assert calls == 0
    assert breaker.get_state() is CircuitBreakerState.OPEN


@pytest.mark.asyncio
async def test_circuit_breaker_half_open_then_closes():
    breaker = CircuitBreaker(
        "unit",
        failure_threshold=1,
        recovery_timeout=0.01,
        half_open_max_requests=2,
    )

    async def failing_call():
        raise RuntimeError("api down")

    async def successful_call():
        return "live"

    await breaker.call(failing_call, fallback="fallback")
    await asyncio.sleep(0.03)

    assert breaker.get_state() is CircuitBreakerState.HALF_OPEN
    assert await breaker.call(successful_call, fallback="fallback") == "live"
    assert breaker.get_state() is CircuitBreakerState.HALF_OPEN
    assert await breaker.call(successful_call, fallback="fallback") == "live"
    assert breaker.get_state() is CircuitBreakerState.CLOSED


@pytest.mark.asyncio
async def test_circuit_breaker_half_open_then_opens():
    breaker = CircuitBreaker(
        "unit",
        failure_threshold=1,
        recovery_timeout=0.01,
        half_open_max_requests=1,
    )

    async def failing_call():
        raise RuntimeError("api down")

    await breaker.call(failing_call, fallback="fallback")
    await asyncio.sleep(0.03)
    assert breaker.get_state() is CircuitBreakerState.HALF_OPEN

    result = await breaker.call(failing_call, fallback="fallback")

    assert result == "fallback"
    assert breaker.get_state() is CircuitBreakerState.OPEN


def test_get_health_summary():
    for breaker in CIRCUIT_BREAKERS.values():
        breaker.reset()
    CIRCUIT_BREAKERS["github"].record_failure()
    CIRCUIT_BREAKERS["github"].record_failure()
    CIRCUIT_BREAKERS["github"].record_failure()
    CIRCUIT_BREAKERS["github"].record_failure()
    CIRCUIT_BREAKERS["github"].record_failure()

    summary = get_health_summary()

    assert summary["overall"] == "degraded"
    assert set(CIRCUIT_BREAKERS) <= set(summary["circuits"])
    assert "github" in summary["degraded_sources"]
    assert summary["circuits"]["github"]["state"] == "open"


@pytest.mark.asyncio
async def test_circuit_breaker_metrics():
    breaker = CircuitBreaker("unit", failure_threshold=3, recovery_timeout=10.0)

    async def successful_call():
        return "ok"

    async def failing_call():
        raise RuntimeError("api down")

    await breaker.call(successful_call)
    await breaker.call(failing_call, fallback=None)

    metrics = breaker.get_metrics()

    assert metrics["failure_count"] == 1
    assert metrics["success_count"] == 1
    assert metrics["state"] == "closed"
    assert metrics["last_failure_time"] is not None
    assert metrics["uptime_pct"] == 50.0
