import pytest
from fastapi.testclient import TestClient
from kairos.api.server import create_app


def test_health_check():
    app = create_app(db_path=":memory:")
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_signals_returns_list(tmp_path):
    db_path = str(tmp_path / "test.db")
    import duckdb
    from kairos.db import create_schema
    conn = duckdb.connect(db_path)
    create_schema(conn)
    conn.close()

    app = create_app(db_path=db_path)
    client = TestClient(app)
    response = client.get("/signals")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
