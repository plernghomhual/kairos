import httpx
import duckdb

FRED_SERIES = ["DFF", "M2SL", "UNRATE"]
FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"


async def fetch_and_store_macro(
    conn: duckdb.DuckDBPyConnection,
    api_key: str,
) -> None:
    async with httpx.AsyncClient(timeout=30.0) as client:
        for series_id in FRED_SERIES:
            params = {
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": 24,
            }
            resp = await client.get(FRED_BASE, params=params)
            resp.raise_for_status()
            observations = resp.json().get("observations", [])

            rows = [
                (series_id, obs["date"], float(obs["value"]))
                for obs in observations
                if obs["value"] != "."
            ]
            conn.executemany(
                "INSERT OR REPLACE INTO macro_data (series_id, ts, value) VALUES (?, ?, ?)",
                rows,
            )
