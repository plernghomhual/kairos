from __future__ import annotations

import calendar
import json
import logging
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from time import struct_time
from typing import Any

import duckdb
import feedparser
import httpx

from kairos.config import CRYPTOPANIC_API_KEY, DB_PATH
from kairos.db import create_schema, get_connection

_LOG = logging.getLogger(__name__)

CRYPTOPANIC_URL = "https://cryptopanic.com/api/v1/posts/"
FNG_URL = "https://api.alternative.me/fng/"
CRYPTOPANIC_CACHE_TTL = timedelta(minutes=15)
DEVELOPER_CACHE_TTL = timedelta(hours=1)
SUPPORTED_ASSETS = {"BTC", "ETH", "SOL"}

DEVELOPER_FEEDS = {
    "BTC": [
        "https://bitcoin.stackexchange.com/feeds",
    ],
    "ETH": [
        "https://ethereum.stackexchange.com/feeds",
        "https://ethresear.ch/latest.rss",
    ],
    "SOL": [
        "https://solana.stackexchange.com/feeds",
    ],
}


async def fetch_alt_sentiment(
    asset: str = "BTC",
    *,
    conn: duckdb.DuckDBPyConnection | None = None,
    cryptopanic_api_key: str | None = None,
    fng_score: int | None = None,
) -> dict:
    """Aggregate crypto-native sentiment sources into a unified vector."""
    asset = asset.upper()
    if asset not in SUPPORTED_ASSETS:
        raise ValueError(f"Unsupported asset '{asset}'. Choose from: {', '.join(sorted(SUPPORTED_ASSETS))}")

    own_conn = conn is None
    if conn is None:
        conn = get_connection(DB_PATH)
        create_schema(conn)

    try:
        cryptopanic = await _load_or_fetch_cryptopanic(
            conn,
            asset,
            cryptopanic_api_key if cryptopanic_api_key is not None else CRYPTOPANIC_API_KEY,
        )
        developer = await _load_or_fetch_developer(conn, asset)

        if fng_score is None:
            fng_score = await _safe_fetch_fng_score()

        sources_available = []
        if cryptopanic["available"]:
            sources_available.append("cryptopanic")
        if developer["available"]:
            sources_available.append("developer_rss")

        composite = _compute_composite(cryptopanic, developer)
        fetch_ts = datetime.now(timezone.utc).isoformat()

        return {
            "cryptopanic_sentiment": round(float(cryptopanic["score"]), 4),
            "developer_attention": round(float(developer["score"]), 4),
            "post_volume_24h": int(cryptopanic["post_volume_24h"] + developer["post_volume_24h"]),
            "sentiment_divergence": _detect_sentiment_divergence(fng_score, cryptopanic),
            "composite": round(composite, 4),
            "sources_available": sources_available,
            "fetch_ts": fetch_ts,
        }
    finally:
        if own_conn:
            conn.close()


async def _load_or_fetch_cryptopanic(
    conn: duckdb.DuckDBPyConnection,
    asset: str,
    api_key: str,
) -> dict:
    cached = _load_cached(conn, "cryptopanic", asset, CRYPTOPANIC_CACHE_TTL)
    if cached is not None:
        return _summarize_cryptopanic(cached, asset)

    try:
        result = await _fetch_cryptopanic_sentiment(asset, api_key)
    except Exception as exc:
        _LOG.warning("CryptoPanic fetch failed for %s: %s", asset, exc)
        return _unavailable_source()

    if result.get("raw"):
        _store_cache(conn, "cryptopanic", asset, result["raw"])
    return result


async def _load_or_fetch_developer(
    conn: duckdb.DuckDBPyConnection,
    asset: str,
) -> dict:
    cached = _load_cached(conn, "developer_rss", asset, DEVELOPER_CACHE_TTL)
    if cached is not None:
        return _summarize_developer_attention(cached)

    try:
        result = await _fetch_developer_attention(asset)
    except Exception as exc:
        _LOG.warning("Developer RSS fetch failed for %s: %s", asset, exc)
        return _unavailable_source()

    if result["available"]:
        _store_cache(conn, "developer_rss", asset, result["raw"])
    return result


async def _fetch_cryptopanic_sentiment(asset: str, api_key: str) -> dict:
    if not api_key:
        return _unavailable_source()

    params = {
        "auth_token": api_key,
        "currencies": asset,
        "public": "true",
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(CRYPTOPANIC_URL, params=params)
        response.raise_for_status()
        raw = response.json()

    return _summarize_cryptopanic(raw, asset)


async def _fetch_developer_attention(asset: str) -> dict:
    feeds = DEVELOPER_FEEDS.get(asset, [])
    if not feeds:
        return _unavailable_source()

    raw = {"feeds": []}
    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
        for feed_url in feeds:
            try:
                response = await client.get(feed_url)
                response.raise_for_status()
            except Exception as exc:
                _LOG.debug("Developer RSS feed %s failed: %s", feed_url, exc)
                continue

            parsed = feedparser.parse(response.content)
            entries = []
            for entry in parsed.entries:
                entry_dt = _entry_datetime(entry)
                entries.append(
                    {
                        "title": entry.get("title", ""),
                        "published": entry_dt.isoformat() if entry_dt else None,
                        "comments": _entry_comment_count(entry),
                    }
                )
            raw["feeds"].append({"url": feed_url, "entries": entries})

    return _summarize_developer_attention(raw)


async def _safe_fetch_fng_score() -> int | None:
    try:
        return await _fetch_fng_score()
    except Exception as exc:
        _LOG.debug("FNG score fetch failed: %s", exc)
        return None


async def _fetch_fng_score() -> int | None:
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(FNG_URL, params={"limit": 1, "format": "json"})
        response.raise_for_status()
        data = response.json().get("data", [])
    if not data:
        return None
    return int(data[0]["value"])


def _summarize_cryptopanic(raw: dict, asset: str) -> dict:
    posts = [post for post in raw.get("results", []) if _post_matches_asset(post, asset)]
    total = len(posts)
    if total == 0:
        return _unavailable_source(raw=raw)

    positive_count = 0
    negative_count = 0
    for post in posts:
        sentiment = _post_sentiment(post)
        if sentiment > 0:
            positive_count += 1
        elif sentiment < 0:
            negative_count += 1

    score = (positive_count - negative_count) / total
    post_volume_24h = sum(1 for post in posts if _is_recent_post(post, hours=24))
    return {
        "available": True,
        "score": _clamp(score, -1.0, 1.0),
        "post_volume_24h": post_volume_24h,
        "raw": raw,
    }


def _summarize_developer_attention(raw: dict) -> dict:
    now = datetime.now(timezone.utc)
    current_week_volume = 0
    four_week_volume = 0
    post_volume_24h = 0

    for feed in raw.get("feeds", []):
        for entry in feed.get("entries", []):
            entry_dt = _parse_datetime(entry.get("published"))
            if entry_dt is None:
                continue

            volume = 1 + max(int(entry.get("comments") or 0), 0)
            age = now - entry_dt
            if timedelta(0) <= age <= timedelta(days=28):
                four_week_volume += volume
            if timedelta(0) <= age <= timedelta(days=7):
                current_week_volume += volume
            if timedelta(0) <= age <= timedelta(hours=24):
                post_volume_24h += volume

    if four_week_volume <= 0:
        return _unavailable_source(raw=raw)

    rolling_4week_avg = four_week_volume / 4.0
    score = min(current_week_volume / rolling_4week_avg, 2.0) / 2.0
    return {
        "available": True,
        "score": _clamp(score, 0.0, 1.0),
        "post_volume_24h": post_volume_24h,
        "raw": raw,
    }


def _compute_composite(cryptopanic: dict, developer: dict) -> float:
    weighted_sum = 0.0
    weight_sum = 0.0

    if cryptopanic["available"]:
        weighted_sum += float(cryptopanic["score"]) * 0.4
        weight_sum += 0.4
    if developer["available"]:
        developer_sentiment = (float(developer["score"]) * 2.0) - 1.0
        weighted_sum += developer_sentiment * 0.3
        weight_sum += 0.3

    if weight_sum == 0.0:
        return 0.0
    return _clamp(weighted_sum / weight_sum, -1.0, 1.0)


def _detect_sentiment_divergence(fng_score: int | None, cryptopanic: dict) -> bool:
    if fng_score is None or not cryptopanic["available"]:
        return False

    fng_bullish = fng_score >= 60
    fng_bearish = fng_score <= 40
    crypto_bullish = cryptopanic["score"] >= 0.2
    crypto_bearish = cryptopanic["score"] <= -0.2
    return bool((fng_bullish and crypto_bearish) or (fng_bearish and crypto_bullish))


def _load_cached(
    conn: duckdb.DuckDBPyConnection,
    source: str,
    asset: str,
    ttl: timedelta,
) -> dict | None:
    row = conn.execute(
        """
        SELECT raw_data, fetched_at
        FROM sentiment_cache
        WHERE source = ? AND asset = ?
        ORDER BY fetched_at DESC
        LIMIT 1
        """,
        [source, asset],
    ).fetchone()
    if row is None:
        return None

    raw_data, fetched_at = row
    fetched_dt = _coerce_datetime(fetched_at)
    if fetched_dt is None or datetime.now(timezone.utc) - fetched_dt > ttl:
        return None

    if isinstance(raw_data, str):
        return json.loads(raw_data)
    return raw_data


def _store_cache(
    conn: duckdb.DuckDBPyConnection,
    source: str,
    asset: str,
    raw_data: dict,
) -> None:
    conn.execute(
        "DELETE FROM sentiment_cache WHERE source = ? AND asset = ?",
        [source, asset],
    )
    conn.execute(
        """
        INSERT INTO sentiment_cache (source, asset, raw_data)
        VALUES (?, ?, CAST(? AS JSON))
        """,
        [source, asset, json.dumps(raw_data)],
    )


def _post_matches_asset(post: dict, asset: str) -> bool:
    currencies = post.get("currencies") or []
    if not currencies:
        return True
    return any(currency.get("code") == asset for currency in currencies if isinstance(currency, dict))


def _post_sentiment(post: dict) -> int:
    sentiment = str(post.get("sentiment") or post.get("kind") or "").lower()
    if "positive" in sentiment or sentiment == "bullish":
        return 1
    if "negative" in sentiment or sentiment == "bearish":
        return -1

    votes = post.get("votes") or {}
    positive_votes = int(votes.get("positive") or 0)
    negative_votes = int(votes.get("negative") or 0)
    if positive_votes > negative_votes:
        return 1
    if negative_votes > positive_votes:
        return -1
    return 0


def _is_recent_post(post: dict, *, hours: int) -> bool:
    published = _parse_datetime(post.get("published_at") or post.get("created_at"))
    if published is None:
        return False
    age = datetime.now(timezone.utc) - published
    return timedelta(0) <= age <= timedelta(hours=hours)


def _entry_datetime(entry: Any) -> datetime | None:
    for key in ("published_parsed", "updated_parsed"):
        parsed = entry.get(key)
        if isinstance(parsed, struct_time):
            return datetime.fromtimestamp(calendar.timegm(parsed), tz=timezone.utc)

    for key in ("published", "updated", "created"):
        parsed = _parse_datetime(entry.get(key))
        if parsed is not None:
            return parsed
    return None


def _entry_comment_count(entry: Any) -> int:
    for key in ("slash_comments", "comments_count", "comment_count"):
        value = entry.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0
    return 0


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return _ensure_utc(value)
    if not isinstance(value, str):
        return None

    try:
        return _ensure_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
    except ValueError:
        try:
            return _ensure_utc(parsedate_to_datetime(value))
        except (TypeError, ValueError):
            return None


def _coerce_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return _ensure_utc(value)
    return _parse_datetime(value)


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _unavailable_source(raw: dict | None = None) -> dict:
    return {
        "available": False,
        "score": 0.0,
        "post_volume_24h": 0,
        "raw": raw or {},
    }


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
