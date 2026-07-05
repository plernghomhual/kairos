import subprocess
import sys

from fastapi.testclient import TestClient

from kairos.api.server import create_app


def test_health_check():
    app = create_app(db_path=":memory:")
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_signals_returns_list(tmp_path, monkeypatch):
    monkeypatch.setenv("KAIROS_API_KEY", "test-api-key")
    db_path = str(tmp_path / "test.db")
    import duckdb

    from kairos.db import create_schema

    conn = duckdb.connect(db_path)
    create_schema(conn)
    conn.close()

    app = create_app(db_path=db_path)
    client = TestClient(app)
    response = client.get("/signals", headers={"X-API-Key": "test-api-key"})
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_signal_routes_fail_closed_when_api_key_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("KAIROS_API_KEY", raising=False)
    db_path = str(tmp_path / "test.db")

    app = create_app(db_path=db_path)
    client = TestClient(app)

    response = client.get("/signals")

    assert response.status_code == 503
    assert response.json()["detail"] == "KAIROS_API_KEY is required"


def test_signal_routes_reject_wrong_api_key(tmp_path, monkeypatch):
    monkeypatch.setenv("KAIROS_API_KEY", "test-api-key")
    app = create_app(db_path=str(tmp_path / "test.db"))
    client = TestClient(app)

    response = client.get("/signals", headers={"X-API-Key": "test-api-kex"})

    assert response.status_code == 401


def test_docs_and_openapi_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("KAIROS_API_KEY", "test-api-key")
    app = create_app(db_path=str(tmp_path / "test.db"))
    client = TestClient(app)

    assert client.get("/docs").status_code == 404
    assert client.get("/redoc").status_code == 404
    assert client.get("/openapi.json").status_code == 404


def test_create_app_does_not_hold_duckdb_write_lock(tmp_path):
    db_path = str(tmp_path / "test.db")
    app = create_app(db_path=db_path)
    assert app is not None

    code = "import duckdb, sys; conn = duckdb.connect(sys.argv[1]); conn.close(); print('ok')"
    result = subprocess.run(
        [sys.executable, "-c", code, db_path],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )

    assert result.returncode == 0, result.stderr
