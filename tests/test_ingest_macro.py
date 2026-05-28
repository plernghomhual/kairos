import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import duckdb
from kairos.db import create_schema
from kairos.ingest.macro import fetch_and_store_macro

MOCK_FRED_RESPONSE = {
    "observations": [
        {"date": "2021-01-01", "value": "0.25"},
        {"date": "2021-02-01", "value": "0.25"},
    ]
}

@pytest.mark.asyncio
async def test_fetch_and_store_macro(tmp_path):
    db_path = str(tmp_path / "test.db")
    conn = duckdb.connect(db_path)
    create_schema(conn)

    with patch("kairos.ingest.macro.httpx.AsyncClient") as mock_client_cls:
        mock_response = MagicMock()
        mock_response.json.return_value = MOCK_FRED_RESPONSE
        mock_response.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        await fetch_and_store_macro(conn, api_key="test")

    rows = conn.execute("SELECT * FROM macro_data WHERE series_id = 'DFF'").fetchall()
    assert len(rows) == 2
    conn.close()
