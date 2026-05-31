from datetime import datetime, timezone

import duckdb
import praw

SUBREDDITS = ["Bitcoin", "CryptoCurrency", "btc"]


def fetch_and_store_posts(
    conn: duckdb.DuckDBPyConnection,
    client_id: str,
    client_secret: str,
    user_agent: str,
    limit: int = 100,
) -> None:
    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )

    rows = []
    for sub_name in SUBREDDITS:
        subreddit = reddit.subreddit(sub_name)
        for post in subreddit.hot(limit=limit):
            ts = datetime.fromtimestamp(post.created_utc, tz=timezone.utc)
            rows.append(
                (
                    post.id,
                    sub_name,
                    post.title,
                    post.selftext,
                    post.score,
                    post.num_comments,
                    ts,
                )
            )

    conn.executemany(
        """
        INSERT OR IGNORE INTO raw_posts
            (id, subreddit, title, body, score, num_comments, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
