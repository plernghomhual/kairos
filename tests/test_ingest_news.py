from unittest.mock import AsyncMock, MagicMock, patch

import duckdb
import pytest

from kairos.db import create_schema
from kairos.ingest.news import fetch_and_store_news

MOCK_NEWS_RESPONSE = {
    "Data": [
        {
            "id": "1001",
            "title": "Bitcoin surges past 50k",
            "body": "BTC has risen sharply amid...",
            "published_on": 1609459200,
            "source": "CryptoCompare",
        },
        {
            "id": "1002",
            "title": "Ethereum upgrade complete",
            "body": "The merge has been finalized...",
            "published_on": 1609545600,
            "source": "CryptoCompare",
        },
    ]
}


@pytest.mark.asyncio
async def test_fetch_and_store_news(tmp_path):
    db_path = str(tmp_path / "test.db")
    conn = duckdb.connect(db_path)
    create_schema(conn)

    with patch("kairos.ingest.news.httpx.AsyncClient") as mock_client_cls:
        mock_response = MagicMock()
        mock_response.json.return_value = MOCK_NEWS_RESPONSE
        mock_response.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        await fetch_and_store_news(conn, api_key="test_key")

    rows = conn.execute("SELECT * FROM raw_news").fetchall()
    assert len(rows) == 2
    assert rows[0][2] in ("Bitcoin surges past 50k", "Ethereum upgrade complete")
    conn.close()


@pytest.mark.asyncio
async def test_articles_with_missing_fields_are_skipped(tmp_path):
    """Articles missing title or published_on must be silently skipped."""
    db_path = str(tmp_path / "test.db")
    conn = duckdb.connect(db_path)
    create_schema(conn)

    malformed = {
        "Data": [
            {"id": "1", "title": "Good article", "body": "body", "published_on": 1609459200, "source": "s"},
            {"id": "2", "title": None, "body": "body", "published_on": 1609545600, "source": "s"},
            {"id": "3", "body": "body", "published_on": None, "source": "s"},
        ]
    }

    with patch("kairos.ingest.news.httpx.AsyncClient") as mock_client_cls:
        mock_response = MagicMock()
        mock_response.json.return_value = malformed
        mock_response.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        await fetch_and_store_news(conn, api_key="key")

    rows = conn.execute("SELECT * FROM raw_news").fetchall()
    assert len(rows) == 1
    assert rows[0][2] == "Good article"
    conn.close()
