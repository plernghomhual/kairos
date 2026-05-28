import pytest
from kairos.db import get_connection, create_schema

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
