import logging
from datetime import datetime, timezone

import duckdb
import httpx

_ALLOWED_COIN_IDS = frozenset({"bitcoin", "ethereum", "solana"})
_MAX_RETRIES = 3
logger = logging.getLogger(__name__)


async def fetch_and_store_ohlcv(
    coin_id: str,
    asset: str,
    conn: duckdb.DuckDBPyConnection,
    days: int = 30,
) -> None:
    if coin_id not in _ALLOWED_COIN_IDS:
        raise ValueError(f"Unsupported coin_id {coin_id!r}. Allowed: {sorted(_ALLOWED_COIN_IDS)}")

    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": days, "interval": "daily"}

    import asyncio

    backoff = 2.0
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(url, params=params)
                if resp.status_code == 429:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 30.0)
                    last_exc = httpx.HTTPStatusError("429 rate limited", request=resp.request, response=resp)
                    continue
                resp.raise_for_status()
                data = resp.json()
                if "prices" not in data or "total_volumes" not in data:
                    raise ValueError(
                        f"CoinGecko response missing 'prices' or 'total_volumes' keys for {coin_id}. "
                        f"Keys present: {list(data.keys())}"
                    )
                break
        except (httpx.RequestError, httpx.HTTPStatusError, ValueError, KeyError, TypeError) as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES - 1:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
    else:
        if isinstance(last_exc, (ValueError, KeyError, TypeError)):
            raise ValueError(
                f"CoinGecko response missing or invalid after {_MAX_RETRIES} attempts for {coin_id}"
            ) from last_exc
        raise RuntimeError(f"fetch_and_store_ohlcv failed after {_MAX_RETRIES} attempts") from last_exc

    prices = data["prices"]
    volumes = data["total_volumes"]
    if len(prices) != len(volumes):
        logger.warning(
            "mismatched CoinGecko price/volume lengths for %s: prices=%d volumes=%d",
            coin_id,
            len(prices),
            len(volumes),
        )

    rows = []
    for (ts_ms, close), (_, vol) in zip(prices, volumes):
        ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        # CoinGecko /market_chart returns daily close only — open/high/low are synthetic.
        rows.append((asset, ts, close, close, close, close, vol))

    conn.executemany(
        """
        INSERT OR REPLACE INTO price_candles
            (asset, ts, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
