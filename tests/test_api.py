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
