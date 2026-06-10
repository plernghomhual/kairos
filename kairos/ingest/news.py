import logging
from datetime import datetime, timezone

import duckdb
import httpx

logger = logging.getLogger(__name__)
_MAX_BODY_LEN = 4096


async def fetch_and_store_news(
    conn: duckdb.DuckDBPyConnection,
    api_key: str = "",
    limit: int = 50,
) -> None:
    if not api_key:
        logger.debug("No CryptoCompare API key provided — anonymous requests are heavily rate-limited")

    url = "https://min-api.cryptocompare.com/data/v2/news/"
    params = {"lang": "EN", "sortOrder": "latest", "extraParams": "kairos"}
    if api_key:
        params["api_key"] = api_key

    _httpx_log = logging.getLogger("httpx")
    _httpcore_log = logging.getLogger("httpcore")
    _prev = (_httpx_log.level, _httpcore_log.level)
    _httpx_log.setLevel(logging.ERROR)
    _httpcore_log.setLevel(logging.ERROR)
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            articles = resp.json().get("Data", [])[:limit]
    finally:
        _httpx_log.setLevel(_prev[0])
        _httpcore_log.setLevel(_prev[1])

    rows = []
    for a in articles:
        published_on = a.get("published_on")
        title = a.get("title")
        if published_on is None or title is None:
            logger.debug("Skipping article with missing published_on or title: %r", a.get("id"))
            continue
        try:
            ts = datetime.fromtimestamp(int(published_on), tz=timezone.utc)
        except (ValueError, OSError):
            logger.debug("Skipping article with invalid timestamp %r", published_on)
            continue
        body = str(a.get("body") or "")[:_MAX_BODY_LEN]
        rows.append((str(a.get("id", "")), a.get("source", ""), title, body, ts))

    if rows:
        conn.executemany(
            """
            INSERT OR IGNORE INTO raw_news (id, source, title, body, published)
            VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )
