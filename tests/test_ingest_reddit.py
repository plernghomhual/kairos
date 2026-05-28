import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone
import duckdb
from kairos.db import create_schema
from kairos.ingest.reddit import fetch_and_store_posts

def make_mock_post(post_id, title, body, score, num_comments, created_utc):
    post = MagicMock()
    post.id = post_id
    post.subreddit.display_name = "Bitcoin"
    post.title = title
    post.selftext = body
    post.score = score
    post.num_comments = num_comments
    post.created_utc = created_utc
    return post

def test_fetch_and_store_posts(tmp_path):
    db_path = str(tmp_path / "test.db")
    conn = duckdb.connect(db_path)
    create_schema(conn)

    mock_posts = [
        make_mock_post("t1", "BTC to the moon", "Great fundamentals", 1500, 300, 1609459200.0),
        make_mock_post("t2", "Selling all my BTC", "It's over", 200, 50, 1609545600.0),
    ]

    with patch("kairos.ingest.reddit.praw.Reddit") as mock_reddit_cls:
        mock_reddit = MagicMock()
        mock_subreddit = MagicMock()
        mock_subreddit.hot.return_value = mock_posts
        mock_reddit.subreddit.return_value = mock_subreddit
        mock_reddit_cls.return_value = mock_reddit

        fetch_and_store_posts(conn, client_id="x", client_secret="y", user_agent="z")

    rows = conn.execute("SELECT * FROM raw_posts").fetchall()
    assert len(rows) == 2
    titles = {row[2] for row in rows}
    assert "BTC to the moon" in titles
    conn.close()
