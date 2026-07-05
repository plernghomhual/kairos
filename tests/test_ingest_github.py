import json

import duckdb

from kairos.ingest.github import _persist_event


def test_persist_event_writes_github_events(tmp_path, monkeypatch):
    db_path = tmp_path / "github.db"
    monkeypatch.setattr("kairos.ingest.github.DB_PATH", str(db_path))
    event = {
        "event_type": "push",
        "repo": "owner/repo",
        "payload": {"ref": "refs/heads/main"},
        "received_at": "2026-07-04T12:00:00+00:00",
    }

    _persist_event(event)

    conn = duckdb.connect(str(db_path))
    try:
        row = conn.execute("SELECT event_type, repo, payload FROM github_events").fetchone()
    finally:
        conn.close()

    assert row is not None
    assert row[0] == "push"
    assert row[1] == "owner/repo"
    assert json.loads(row[2]) == {"ref": "refs/heads/main"}
