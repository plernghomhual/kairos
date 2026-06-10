import hashlib
import hmac
from unittest.mock import AsyncMock, patch

import pytest

import kairos.backtest.engine as engine
import kairos.ingest as ingest
import kairos.live as live
from kairos.ingest.github import _verify_webhook_signature
from kairos.live import DataFetchSupervisor, fetch_live_data, run_pipeline
from kairos.models.signal_event import SignalEvent


class _MockResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def _clear_live_data_cache():
    if hasattr(live, "_LIVE_DATA_CACHE"):
        live._LIVE_DATA_CACHE.clear()
    yield
    if hasattr(live, "_LIVE_DATA_CACHE"):
        live._LIVE_DATA_CACHE.clear()


@pytest.mark.asyncio
async def test_mock_coingecko_api_pipeline_returns_valid_signal(monkeypatch):
    async def noop_source(self, asset):
        return {"available": False}

    monkeypatch.setattr(DataFetchSupervisor, "_fetch_github_source", noop_source)
    monkeypatch.setattr(DataFetchSupervisor, "_fetch_whale_source", noop_source)
    monkeypatch.setattr(DataFetchSupervisor, "_fetch_macro_source", noop_source)

    price_payload = {
        "prices": [[i * 86400000, 50_000.0 + i * 25.0] for i in range(90)],
        "total_volumes": [[i * 86400000, 1_000_000.0 + i] for i in range(90)],
    }
    fng_payload = {"data": [{"value": "50"} for _ in range(90)]}

    async def mock_get(url, **kwargs):
        if "coingecko" in url:
            return _MockResponse(price_payload)
        return _MockResponse(fng_payload)

    with patch("httpx.AsyncClient") as mock_client:
        instance = AsyncMock()
        instance.get.side_effect = mock_get
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        mock_client.return_value = instance

        prices, _, fng_scores, fng_ok, volumes = await fetch_live_data("BTC")

    assert fng_ok is True
    assert len(prices) == len(fng_scores) == len(volumes) == 90
    event = run_pipeline(prices, fng_scores=fng_scores, volumes=volumes)
    assert isinstance(event, SignalEvent)


@pytest.mark.asyncio
async def test_mock_fng_api_failure_activates_fallback(monkeypatch):
    async def noop_source(self, asset):
        return {"available": False}

    monkeypatch.setattr(DataFetchSupervisor, "_fetch_github_source", noop_source)
    monkeypatch.setattr(DataFetchSupervisor, "_fetch_whale_source", noop_source)
    monkeypatch.setattr(DataFetchSupervisor, "_fetch_macro_source", noop_source)

    price_payload = {"prices": [[i * 86400000, 50_000.0 + i] for i in range(70)]}

    async def mock_get(url, **kwargs):
        if "coingecko" in url:
            return _MockResponse(price_payload)
        raise RuntimeError("fng down")

    with patch("httpx.AsyncClient") as mock_client:
        instance = AsyncMock()
        instance.get.side_effect = mock_get
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        mock_client.return_value = instance

        _, _, fng_scores, fng_ok, _ = await fetch_live_data("BTC")

    assert fng_ok is False
    assert fng_scores == live._FNG_FALLBACK


@pytest.mark.asyncio
async def test_mock_github_api_code_velocity_integrated(monkeypatch):
    async def fake_code_velocity():
        return {
            "available": True,
            "commits": 12,
            "contributors": 4,
            "pull_requests": 3,
            "stars": 100,
            "forks": 20,
            "churn": 2.5,
            "repos": ["org/repo"],
        }

    monkeypatch.setattr(ingest, "fetch_code_velocity", fake_code_velocity)
    result = await live.safe_fetch_code_velocity()
    assert result["available"] is True
    assert result["commit_velocity"] == 12
    assert result["merged_prs"] == 3
    assert result["repos_scraped"] == ["org/repo"]


@pytest.mark.asyncio
async def test_mock_solana_rpc_whale_flow_integrated(monkeypatch):
    async def fake_whales():
        return {
            "available": True,
            "transfers_count": 2,
            "net_flow_usd": -250_000.0,
            "largest_flow_usd": 200_000.0,
            "transfers": [{"signature": "abc"}],
        }

    monkeypatch.setattr(ingest, "fetch_whale_flows", fake_whales)
    result = await live.safe_fetch_whale_flows()
    assert result["available"] is True
    assert result["transfers_count"] == 2
    assert result["net_flow_usd"] == pytest.approx(-250_000.0)


@pytest.mark.asyncio
async def test_mock_binance_api_order_book_data_integrated(monkeypatch):
    calls = []

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url):
            calls.append(url)
            return _MockResponse(
                {
                    "bids": [["100.0", "2.0"]],
                    "asks": [["100.5", "3.0"]],
                }
            )

    monkeypatch.setattr(
        engine,
        "httpx",
        type("FakeHttpx", (), {"AsyncClient": FakeAsyncClient}),
        raising=False,
    )
    engine._order_book_cache.clear()

    depth = await engine._fetch_order_book_depth("BTC")

    assert depth["bids"] == [(100.0, 2.0)]
    assert depth["asks"] == [(100.5, 3.0)]
    assert depth["mid_price"] == pytest.approx(100.25)
    assert "BTCUSDT" in calls[0]


@pytest.mark.asyncio
async def test_all_apis_fail_pipeline_returns_neutral_safe_signal(monkeypatch):
    async def fails(self, asset):
        raise RuntimeError("api down")

    monkeypatch.setattr(DataFetchSupervisor, "_fetch_prices_source", fails)
    monkeypatch.setattr(DataFetchSupervisor, "_fetch_fng_source", fails)
    monkeypatch.setattr(DataFetchSupervisor, "_fetch_github_source", fails)
    monkeypatch.setattr(DataFetchSupervisor, "_fetch_whale_source", fails)
    monkeypatch.setattr(DataFetchSupervisor, "_fetch_macro_source", fails)

    result = await DataFetchSupervisor(max_retries=0).fetch_all("BTC")
    assert all(source["available"] is False for source in result.values())

    event = run_pipeline([], fng_scores=[])
    assert event.direction == "neutral"
    assert event.confidence == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_partial_api_failure_keeps_price_data(monkeypatch):
    async def prices(self, asset):
        return {
            "prices": [100.0, 101.0],
            "current_price": 101.0,
            "volumes": [10.0, 11.0],
        }

    async def ok(self, asset):
        return {"available": True}

    async def fails(self, asset):
        raise RuntimeError("github down")

    monkeypatch.setattr(DataFetchSupervisor, "_fetch_prices_source", prices)
    monkeypatch.setattr(DataFetchSupervisor, "_fetch_fng_source", ok)
    monkeypatch.setattr(DataFetchSupervisor, "_fetch_github_source", fails)
    monkeypatch.setattr(DataFetchSupervisor, "_fetch_whale_source", ok)
    monkeypatch.setattr(DataFetchSupervisor, "_fetch_macro_source", ok)

    result = await DataFetchSupervisor(max_retries=0).fetch_all("BTC")

    assert result["prices"]["available"] is True
    assert result["prices"]["data"]["current_price"] == 101.0
    assert result["github"]["available"] is False
    # Circuit breaker replaces exception message with fallback
    assert result["github"]["error"] is not None


def test_webhook_signature_valid(monkeypatch):
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", "test-secret")
    body = b'{"action": "push"}'
    sig = "sha256=" + hmac.new(b"test-secret", body, hashlib.sha256).hexdigest()
    assert _verify_webhook_signature(body, sig) is True


def test_webhook_signature_invalid(monkeypatch):
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", "test-secret")
    body = b'{"action": "push"}'
    assert _verify_webhook_signature(body, "sha256=badhash") is False


def test_webhook_signature_missing_when_secret_set(monkeypatch):
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", "test-secret")
    assert _verify_webhook_signature(b"body", None) is False


def test_webhook_no_secret_rejects_unsigned_payload(monkeypatch):
    monkeypatch.delenv("GITHUB_WEBHOOK_SECRET", raising=False)
    assert _verify_webhook_signature(b"any body", None) is False
