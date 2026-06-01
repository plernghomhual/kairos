import asyncio
import logging

import duckdb
import httpx

FRED_SERIES = ["DFF", "M2SL", "UNRATE"]
FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
_MAX_RETRIES = 3

logger = logging.getLogger(__name__)

# FRED sentinel values that represent missing data (beyond the documented ".")
_FRED_MISSING = frozenset({".", "", "nd"})


async def fetch_and_store_macro(
    conn: duckdb.DuckDBPyConnection,
    api_key: str,
) -> None:
    _httpx_log = logging.getLogger("httpx")
    _httpcore_log = logging.getLogger("httpcore")
    _prev = (_httpx_log.level, _httpcore_log.level)
    _httpx_log.setLevel(logging.ERROR)
    _httpcore_log.setLevel(logging.ERROR)
    try:
        await _fetch_and_store_macro_inner(conn, api_key)
    finally:
        _httpx_log.setLevel(_prev[0])
        _httpcore_log.setLevel(_prev[1])


async def _fetch_and_store_macro_inner(
    conn: duckdb.DuckDBPyConnection,
    api_key: str,
) -> None:
    backoff = 2.0
    async with httpx.AsyncClient(timeout=30.0) as client:
        for series_id in FRED_SERIES:
            params = {
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": 24,
            }
            last_exc: Exception | None = None
            for attempt in range(_MAX_RETRIES):
                try:
                    resp = await client.get(FRED_BASE, params=params)
                    if resp.status_code == 429:
                        await asyncio.sleep(backoff)
                        backoff = min(backoff * 2, 60.0)
                        last_exc = httpx.HTTPStatusError("429 rate limited", request=resp.request, response=resp)
                        continue
                    resp.raise_for_status()
                    observations = resp.json().get("observations", [])
                    break
                except (httpx.RequestError, httpx.HTTPStatusError) as exc:
                    last_exc = exc
                    logger.warning(
                        "FRED fetch failed for %s (attempt %d/%d): %s",
                        series_id,
                        attempt + 1,
                        _MAX_RETRIES,
                        exc,
                    )
                    if attempt < _MAX_RETRIES - 1:
                        await asyncio.sleep(backoff)
                        backoff = min(backoff * 2, 60.0)
            else:
                logger.error("Skipping FRED series %s after %d failed attempts: %s", series_id, _MAX_RETRIES, last_exc)
                continue

            rows = []
            for obs in observations:
                raw_val = obs.get("value", ".")
                if raw_val in _FRED_MISSING:
                    continue
                try:
                    rows.append((series_id, obs["date"], float(raw_val)))
                except (ValueError, TypeError):
                    logger.debug("Skipping non-numeric FRED value %r for %s on %s", raw_val, series_id, obs.get("date"))
                    continue

            conn.executemany(
                "INSERT OR REPLACE INTO macro_data (series_id, ts, value) VALUES (?, ?, ?)",
                rows,
            )
