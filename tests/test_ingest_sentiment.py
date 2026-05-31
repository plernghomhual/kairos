from unittest.mock import AsyncMock, patch

import duckdb
import pytest

from kairos.db import create_schema
from kairos.ingest.sentiment import fetch_alt_sentiment

EXPECTED_KEYS = {
    "cryptopanic_sentiment",
    "developer_attention",
    "post_volume_24h",
    "sentiment_divergence",
    "composite",
    "sources_available",
    "fetch_ts",
}


def _conn(tmp_path):
    conn = duckdb.connect(str(tmp_path / "sentiment.db"))
    create_schema(conn)
    return conn


@pytest.mark.asyncio
async def test_fetch_alt_sentiment_returns_structure(tmp_path):
    conn = _conn(tmp_path)
    cryptopanic = {
        "available": True,
        "score": 0.5,
        "post_volume_24h": 4,
        "raw": {"results": [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}]},
    }
    developer = {
        "available": True,
        "score": 0.75,
        "post_volume_24h": 3,
        "raw": {"feeds": [{"url": "https://example.test/feed", "entries": 3}]},
    }

    with (
        patch(
            "kairos.ingest.sentiment._fetch_cryptopanic_sentiment",
            new=AsyncMock(return_value=cryptopanic),
        ),
        patch(
            "kairos.ingest.sentiment._fetch_developer_attention",
            new=AsyncMock(return_value=developer),
        ),
        patch("kairos.ingest.sentiment._fetch_fng_score", new=AsyncMock(return_value=None)),
    ):
        result = await fetch_alt_sentiment("BTC", conn=conn, cryptopanic_api_key="key")

    assert set(result) == EXPECTED_KEYS
    assert result["cryptopanic_sentiment"] == 0.5
    assert result["developer_attention"] == 0.75
    assert result["post_volume_24h"] == 7
    assert result["sentiment_divergence"] is False
    assert -1.0 <= result["composite"] <= 1.0
    assert result["sources_available"] == ["cryptopanic", "developer_rss"]

    cached = conn.execute("SELECT source, asset FROM sentiment_cache ORDER BY source").fetchall()
    assert cached == [("cryptopanic", "BTC"), ("developer_rss", "BTC")]
    conn.close()


@pytest.mark.asyncio
async def test_cryptopanic_fallback_on_error(tmp_path):
    conn = _conn(tmp_path)
    developer = {"available": False, "score": 0.0, "post_volume_24h": 0, "raw": {}}

    with (
        patch(
            "kairos.ingest.sentiment._fetch_cryptopanic_sentiment",
            new=AsyncMock(side_effect=RuntimeError("api unreachable")),
        ),
        patch(
            "kairos.ingest.sentiment._fetch_developer_attention",
            new=AsyncMock(return_value=developer),
        ),
        patch("kairos.ingest.sentiment._fetch_fng_score", new=AsyncMock(return_value=None)),
    ):
        result = await fetch_alt_sentiment("BTC", conn=conn, cryptopanic_api_key="key")

    assert result["cryptopanic_sentiment"] == 0.0
    assert result["developer_attention"] == 0.0
    assert result["post_volume_24h"] == 0
    assert result["sentiment_divergence"] is False
    assert result["composite"] == 0.0
    assert result["sources_available"] == []
    conn.close()


@pytest.mark.asyncio
async def test_sentiment_divergence_detection(tmp_path):
    conn = _conn(tmp_path)
    cryptopanic = {
        "available": True,
        "score": -0.3,
        "post_volume_24h": 10,
        "raw": {"results": [{"id": idx} for idx in range(10)]},
    }
    developer = {"available": False, "score": 0.0, "post_volume_24h": 0, "raw": {}}

    with (
        patch(
            "kairos.ingest.sentiment._fetch_cryptopanic_sentiment",
            new=AsyncMock(return_value=cryptopanic),
        ),
        patch(
            "kairos.ingest.sentiment._fetch_developer_attention",
            new=AsyncMock(return_value=developer),
        ),
        patch("kairos.ingest.sentiment._fetch_fng_score", new=AsyncMock(return_value=75)),
    ):
        result = await fetch_alt_sentiment("BTC", conn=conn, cryptopanic_api_key="key")

    assert result["sentiment_divergence"] is True
    conn.close()
