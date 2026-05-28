from datetime import datetime, timezone
import httpx
import duckdb


async def fetch_and_store_ohlcv(
    coin_id: str,
    asset: str,
    conn: duckdb.DuckDBPyConnection,
    days: int = 30,
) -> None:
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": days, "interval": "daily"}

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

    prices = data["prices"]
    volumes = data["total_volumes"]

    rows = []
    for (ts_ms, close), (_, vol) in zip(prices, volumes):
        ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        rows.append((asset, ts, close, close, close, close, vol))

    conn.executemany(
        """
        INSERT OR REPLACE INTO price_candles
            (asset, ts, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
