from unittest.mock import AsyncMock, MagicMock, patch

import duckdb
import pytest

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


@pytest.mark.asyncio
async def test_missing_prices_key_raises():
    """CoinGecko response missing 'prices'/'total_volumes' must raise ValueError."""
    import duckdb as _duckdb

    from kairos.db import create_schema as _cs

    conn = _duckdb.connect(":memory:")
    _cs(conn)

    with patch("kairos.ingest.price.httpx.AsyncClient") as mock_client_cls:
        mock_response = MagicMock()
        mock_response.json.return_value = {"market_caps": []}
        mock_response.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        with pytest.raises(ValueError, match="missing"):
            await fetch_and_store_ohlcv("bitcoin", "BTC", conn, days=2)


@pytest.mark.asyncio
async def test_malformed_payload_retries_then_succeeds(tmp_path):
    db_path = str(tmp_path / "test.db")
    conn = duckdb.connect(db_path)
    create_schema(conn)

    with patch("kairos.ingest.price.httpx.AsyncClient") as mock_client_cls:
        bad_response = MagicMock()
        bad_response.status_code = 200
        bad_response.raise_for_status = MagicMock()
        bad_response.json.return_value = {"market_caps": []}
        good_response = MagicMock()
        good_response.status_code = 200
        good_response.raise_for_status = MagicMock()
        good_response.json.return_value = MOCK_COINGECKO_RESPONSE
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(side_effect=[bad_response, good_response])
        mock_client_cls.return_value = mock_client

        await fetch_and_store_ohlcv("bitcoin", "BTC", conn, days=2)

    rows = conn.execute("SELECT * FROM price_candles WHERE asset = 'BTC'").fetchall()
    assert len(rows) == 2
    assert mock_client.get.await_count == 2
    conn.close()


@pytest.mark.asyncio
async def test_mismatched_price_volume_lengths_warns(tmp_path, caplog):
    db_path = str(tmp_path / "test.db")
    conn = duckdb.connect(db_path)
    create_schema(conn)
    mismatched = {
        "prices": [[1609459200000, 29000.0], [1609545600000, 31000.0]],
        "total_volumes": [[1609459200000, 35000000000]],
    }

    with patch("kairos.ingest.price.httpx.AsyncClient") as mock_client_cls:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = mismatched
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        with caplog.at_level("WARNING", logger="kairos.ingest.price"):
            await fetch_and_store_ohlcv("bitcoin", "BTC", conn, days=2)

    rows = conn.execute("SELECT * FROM price_candles WHERE asset = 'BTC'").fetchall()
    assert len(rows) == 1
    assert "mismatched CoinGecko price/volume lengths" in caplog.text
    conn.close()
