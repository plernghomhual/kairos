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

    checks = 0
    if _database_is_readable(db_path):
        checks += 1
    if _api_is_healthy(api_port):
        checks += 1

    return 0 if checks >= 1 else 1


if __name__ == "__main__":
    sys.exit(main())
