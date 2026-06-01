from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

import duckdb
import praw

SUBREDDITS = ["Bitcoin", "CryptoCurrency", "btc"]
_MAX_POST_BODY_LEN = 4096
_LOG = logging.getLogger(__name__)


async def fetch_and_store_posts(
    conn: duckdb.DuckDBPyConnection,
    client_id: str,
    client_secret: str,
    user_agent: str,
    limit: int = 100,
) -> None:
    if not client_id or not client_secret:
        _LOG.warning("Reddit credentials not configured; skipping fetch")
        return

    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )

    rows: list[tuple] = []
    for sub_name in SUBREDDITS:
        try:

            def _fetch_sub(name: str = sub_name) -> list[tuple]:
                subreddit = reddit.subreddit(name)
                result = []
                for post in subreddit.hot(limit=limit):
                    ts = datetime.fromtimestamp(post.created_utc, tz=timezone.utc)
                    result.append(
                        (
                            post.id,
                            name,
                            (post.title or "")[:_MAX_POST_BODY_LEN],
                            (post.selftext or "")[:_MAX_POST_BODY_LEN],
                            post.score,
                            post.num_comments,
                            ts,
                        )
                    )
                return result

            sub_rows = await asyncio.to_thread(_fetch_sub)
            rows.extend(sub_rows)
        except Exception as exc:
            _LOG.warning("Failed to fetch r/%s: %s", sub_name, exc)

    if rows:
        conn.executemany(
            """
            INSERT OR IGNORE INTO raw_posts
                (id, subreddit, title, body, score, num_comments, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
