import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime, timezone
import duckdb
from kairos.db import create_schema
from kairos.ingest.price import fetch_and_store_ohlcv

MOCK_COINGECKO_RESPONSE = {
    "prices": [[1609459200000, 29000.0], [1609545600000, 31000.0]],
    "market_caps": [[1609459200000, 540000000000], [1609545600000, 580000000000]],
    "total_volumes": [[1609459200000, 35000000000], [1609545600000, 40000000000]],
}

@pytest.mark.asyncio
async def test_fetch_and_store_ohlcv(tmp_path):
    db_path = str(tmp_path / "test.db")
    conn = duckdb.connect(db_path)
    create_schema(conn)

    with patch("kairos.ingest.price.httpx.AsyncClient") as mock_client_cls:
        mock_response = MagicMock()
        mock_response.json.return_value = MOCK_COINGECKO_RESPONSE
        mock_response.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        await fetch_and_store_ohlcv("bitcoin", "BTC", conn, days=2)

    rows = conn.execute("SELECT * FROM price_candles WHERE asset = 'BTC'").fetchall()
    assert len(rows) == 2
    assert rows[0][5] in (29000.0, 31000.0)  # close price
    conn.close()
