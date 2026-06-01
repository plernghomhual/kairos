from kairos.db import create_schema, get_connection


def test_schema_creates_all_tables(tmp_path):
    db_path = str(tmp_path / "test.db")
    conn = get_connection(db_path)
    create_schema(conn)
    tables = conn.execute("SHOW TABLES").fetchall()
    table_names = {row[0] for row in tables}
    assert "price_candles" in table_names
    assert "raw_news" in table_names
    assert "raw_posts" in table_names
    assert "macro_data" in table_names
    assert "signal_events" in table_names
    conn.close()


def test_default_connection_uses_kairos_db_path_env(tmp_path, monkeypatch):
    db_path = tmp_path / "runtime" / "kairos.db"
    db_path.parent.mkdir()
    monkeypatch.setenv("KAIROS_DB_PATH", str(db_path))

    conn = get_connection()
    create_schema(conn)
    conn.close()

    assert db_path.exists()


def test_create_schema_is_idempotent(tmp_path):
    db_path = str(tmp_path / "test.db")
    conn = get_connection(db_path)
    create_schema(conn)
    create_schema(conn)  # second call must not raise
    tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
    assert "signal_events" in tables
    conn.close()
