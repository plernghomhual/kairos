import logging
import os
import threading
from importlib.metadata import version as _pkg_version

import duckdb
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from kairos.db import create_schema, get_connection

logger = logging.getLogger(__name__)


def _get_version() -> str:
    try:
        return _pkg_version("kairos")
    except Exception:
        return "unknown"


def create_app(db_path: str = "kairos.db") -> FastAPI:
    app = FastAPI(title="Kairos Signal Engine", version=_get_version())

    # CORS — restrict to configured origins; defaults to localhost-only
    raw_origins = os.getenv("KAIROS_CORS_ORIGINS", "http://localhost,http://127.0.0.1")
    allowed_origins = [o.strip() for o in raw_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["X-API-Key"],
    )

    # Schema bootstrap — close write connection immediately after to release lock
    _schema_conn = get_connection(db_path)
    try:
        create_schema(_schema_conn)
    finally:
        _schema_conn.close()

    # Thread-local read-only connections avoid per-request open/close overhead
    # without holding a write lock that blocks other processes.
    _read_local = threading.local()

    def _get_read_conn() -> duckdb.DuckDBPyConnection:
        if not hasattr(_read_local, "conn"):
            _read_local.conn = duckdb.connect(db_path, read_only=True)
        return _read_local.conn

    # API key auth — if KAIROS_API_KEY is set, all non-health routes require it.
    _api_key = os.getenv("KAIROS_API_KEY", "")
    if not _api_key:
        logger.warning(
            "KAIROS_API_KEY is not set — API is running without authentication. " "Set this env var before deploying."
        )

    def _require_auth(x_api_key: str = Header(default="")) -> None:
        if _api_key and x_api_key != _api_key:
            raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")

    def _fetchall(query: str, params: list | None = None):
        return _get_read_conn().execute(query, params or []).fetchall()

    def _fetchone(query: str, params: list | None = None):
        return _get_read_conn().execute(query, params or []).fetchone()

    @app.get("/health")
    def health():
        return {"status": "ok", "version": _get_version()}

    @app.get("/signals")
    def list_signals(
        limit: int = Query(default=20, ge=1, le=1000),
        x_api_key: str = Header(default=""),
    ):
        _require_auth(x_api_key)
        rows = _fetchall("SELECT * FROM signal_events ORDER BY triggered_at DESC LIMIT ?", [limit])
        cols = [
            "id",
            "asset",
            "direction",
            "confidence",
            "regime",
            "narrative_velocity",
            "narrative_tipping_point",
            "mechanism",
            "estimated_hours",
            "citations",
            "triggered_at",
        ]
        return [dict(zip(cols, row)) for row in rows]

    @app.get("/signals/latest")
    def latest_signal(x_api_key: str = Header(default="")):
        _require_auth(x_api_key)
        row = _fetchone("SELECT * FROM signal_events ORDER BY triggered_at DESC LIMIT 1")
        if row is None:
            return JSONResponse(status_code=404, content={"detail": "no signals yet"})
        cols = [
            "id",
            "asset",
            "direction",
            "confidence",
            "regime",
            "narrative_velocity",
            "narrative_tipping_point",
            "mechanism",
            "estimated_hours",
            "citations",
            "triggered_at",
        ]
        return dict(zip(cols, row))

    return app
