#!/usr/bin/env python3
"""Health check for Kairos Docker containers.

Exit 0 if either the configured database is readable or a local API server
responds with HTTP 200 on /health. Exit 1 otherwise.
"""

import os
import sys
from pathlib import Path


def _database_is_readable(db_path: str) -> bool:
    if not Path(db_path).exists():
        return False

    try:
        import duckdb

        conn = duckdb.connect(db_path, read_only=True)
        try:
            conn.execute("SELECT 1").fetchone()
        finally:
            conn.close()
        return True
    except Exception:
        return False


def _api_is_healthy(api_port: str) -> bool:
    try:
        import httpx

        response = httpx.get(f"http://127.0.0.1:{api_port}/health", timeout=3.0)
        return response.status_code == 200
    except Exception:
        return False


def main() -> int:
    db_path = os.getenv("KAIROS_DB_PATH", "kairos.db")
    api_port = os.getenv("KAIROS_API_PORT", "8000")

    db_ok = _database_is_readable(db_path)
    api_ok = _api_is_healthy(api_port)

    if not db_ok:
        print("HEALTH FAIL: database not readable", flush=True)
    if not api_ok:
        print("HEALTH FAIL: API not responding", flush=True)

    # Both checks must pass — a running API with a crashed database is not healthy.
    return 0 if (db_ok and api_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
