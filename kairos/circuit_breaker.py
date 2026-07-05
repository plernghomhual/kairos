"""Circuit breakers and health reporting for external data sources."""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Any

from kairos.config import CIRCUIT_BREAKER_CONFIG

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_requests: int = 3,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_requests = half_open_max_requests
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_successes = 0
        self._half_open_requests = 0
        self._last_failure_time: float | None = None
        self._opened_at: float | None = None
        self._recovery_task: asyncio.Task | None = None
        self._recovery_probe: Callable[[], Awaitable[object] | object] | None = None
        self._lock: asyncio.Lock | None = None  # lazily created inside a running loop

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def call(self, coro_factory, fallback=None):
        """Execute a protected async call and return fallback on failure/open state."""
        async with self._get_lock():
            self._transition_from_open_if_ready()
            if self._state is CircuitBreakerState.OPEN:
                return await self._resolve_fallback(fallback)

            if self._state is CircuitBreakerState.HALF_OPEN:
                if self._half_open_requests >= self.half_open_max_requests:
                    return await self._resolve_fallback(fallback)
                self._half_open_requests += 1

        # Execute outside the lock so we don't hold it during network I/O.
        try:
            result = await coro_factory()
        except Exception as exc:
            async with self._get_lock():
                self.record_failure()
            logger.debug("Circuit breaker protected call failed for %s: %s", self.name, exc)
            return await self._resolve_fallback(fallback)

        async with self._get_lock():
            self.record_success()
        return result

    def record_success(self):
        self._success_count += 1
        if self._state is CircuitBreakerState.HALF_OPEN:
            self._half_open_successes += 1
            if self._half_open_successes >= self.half_open_max_requests:
                self._close()

    def record_failure(self):
        self._last_failure_time = time.time()
        if self._state is CircuitBreakerState.HALF_OPEN:
            self._open()
            return

        self._failure_count += 1
        if self._state is CircuitBreakerState.CLOSED and self._failure_count >= self.failure_threshold:
            self._open()

    def get_state(self) -> CircuitBreakerState:
        self._transition_from_open_if_ready()
        return self._state

    def get_metrics(self) -> dict:
        total = self._success_count + self._failure_count
        uptime_pct = 100.0 if total == 0 else round((self._success_count / total) * 100.0, 2)
        return {
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "state": self.get_state().value,
            "last_failure_time": self._last_failure_time,
            "uptime_pct": uptime_pct,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
            "half_open_max_requests": self.half_open_max_requests,
        }

    def reset(self) -> None:
        if self._recovery_task and not self._recovery_task.done():
            self._recovery_task.cancel()
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_successes = 0
        self._half_open_requests = 0
        self._last_failure_time = None
        self._opened_at = None
        self._recovery_task = None
        self._log_transition(CircuitBreakerState.CLOSED)

    def set_recovery_probe(self, probe: Callable[[], Awaitable[object] | object]) -> None:
        self._recovery_probe = probe

    async def _resolve_fallback(self, fallback):
        value = fallback() if callable(fallback) else fallback
        if inspect.isawaitable(value):
            return await value
        return value

    def _open(self) -> None:
        self._state = CircuitBreakerState.OPEN
        self._opened_at = time.time()
        self._half_open_successes = 0
        self._half_open_requests = 0
        self._log_transition(CircuitBreakerState.OPEN)
        self._schedule_recovery()

    def _half_open(self) -> None:
        self._state = CircuitBreakerState.HALF_OPEN
        self._half_open_successes = 0
        self._half_open_requests = 0
        self._log_transition(CircuitBreakerState.HALF_OPEN)

    def _close(self) -> None:
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._half_open_successes = 0
        self._half_open_requests = 0
        self._opened_at = None
        self._log_transition(CircuitBreakerState.CLOSED)

    def _transition_from_open_if_ready(self) -> None:
        if self._state is not CircuitBreakerState.OPEN or self._opened_at is None:
            return
        if time.time() - self._opened_at >= self.recovery_timeout:
            self._half_open()

    def _schedule_recovery(self) -> None:
        if self._recovery_task and not self._recovery_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._recovery_task = loop.create_task(self._recover_when_ready())

    async def _recover_when_ready(self) -> None:
        backoff = self.recovery_timeout
        while True:
            async with self._get_lock():
                if self._state is not CircuitBreakerState.OPEN:
                    return
            await asyncio.sleep(backoff)
            async with self._get_lock():
                if self._state is not CircuitBreakerState.OPEN:
                    return
                if self._recovery_probe is None:
                    self._half_open()
                    return
            try:
                result = self._recovery_probe()
                if inspect.isawaitable(result):
                    await result
            except Exception:
                async with self._get_lock():
                    self._last_failure_time = time.time()
                    self._opened_at = self._last_failure_time
                backoff = min(backoff * 2, self.recovery_timeout * 16)
                logger.warning(
                    "Circuit breaker recovery probe failed for %s",
                    self.name,
                    exc_info=True,
                )
                continue
            async with self._get_lock():
                if self._state is CircuitBreakerState.OPEN:
                    self._half_open()
            return

    def _log_transition(self, state: CircuitBreakerState) -> None:
        logger.info(
            "Circuit breaker %s transitioned to %s at %.3f",
            self.name,
            state.value,
            time.time(),
        )


def _make_breaker(name: str) -> CircuitBreaker:
    if name not in CIRCUIT_BREAKER_CONFIG:
        raise KeyError(f"No circuit breaker config for {name!r}. " f"Available: {sorted(CIRCUIT_BREAKER_CONFIG)}")
    config = CIRCUIT_BREAKER_CONFIG[name]
    return CircuitBreaker(
        name,
        failure_threshold=config["failure_threshold"],
        recovery_timeout=config["recovery_timeout"],
    )


CIRCUIT_BREAKERS: dict[str, CircuitBreaker] = {
    "coingecko": _make_breaker("coingecko"),
    "fng": _make_breaker("fng"),
    "github": _make_breaker("github"),
    "solana_rpc": _make_breaker("solana_rpc"),
    "macro": _make_breaker("macro"),
    "binance": _make_breaker("binance"),
}


def neutral_macro_vector() -> dict[str, Any]:
    return {
        "source": "default",
        "available": False,
        "series": {},
        "macro_regime": {
            "yield_curve_inverted": False,
            "fed_dovish_hawkish": 0.0,
            "inflation_regime": "stable",
            "macro_stress_level": 0.5,
            "real_yield": 0.0,
            "consumer_health": 0.5,
            "last_updated": None,
        },
    }


def unavailable_metrics() -> dict[str, Any]:
    return {"available": False, "count": 0, "value": 0.0}


def get_health_summary() -> dict:
    circuits = {name: breaker.get_metrics() for name, breaker in CIRCUIT_BREAKERS.items()}
    degraded_sources = [
        name
        for name, metrics in circuits.items()
        if metrics["state"] in {CircuitBreakerState.OPEN.value, CircuitBreakerState.HALF_OPEN.value}
    ]
    open_count = sum(1 for metrics in circuits.values() if metrics["state"] == CircuitBreakerState.OPEN.value)
    if open_count == len(circuits):
        overall = "unhealthy"
    elif degraded_sources:
        overall = "degraded"
    else:
        overall = "healthy"
    return {
        "overall": overall,
        "circuits": circuits,
        "degraded_sources": degraded_sources,
    }
