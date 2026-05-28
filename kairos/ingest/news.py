from datetime import datetime, timezone
import httpx
import duckdb


async def fetch_and_store_news(
    conn: duckdb.DuckDBPyConnection,
    api_key: str = "",
    limit: int = 50,
) -> None:
    url = "https://min-api.cryptocompare.com/data/v2/news/"
    params = {"lang": "EN", "sortOrder": "latest", "extraParams": "kairos"}
    if api_key:
        params["api_key"] = api_key

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        articles = resp.json().get("Data", [])[:limit]

    rows = []
    for a in articles:
        ts = datetime.fromtimestamp(a["published_on"], tz=timezone.utc)
        rows.append((str(a["id"]), a.get("source", ""), a["title"], a.get("body", ""), ts))

    conn.executemany(
        """
        INSERT OR IGNORE INTO raw_news (id, source, title, body, published)
        VALUES (?, ?, ?, ?, ?)
        """,
        rows,
    )
