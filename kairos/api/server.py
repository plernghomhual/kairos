from fastapi import FastAPI
from kairos.db import get_connection, create_schema


def create_app(db_path: str = "kairos.db") -> FastAPI:
    app = FastAPI(title="Kairos Signal Engine", version="0.1.0")
    conn = get_connection(db_path)
    create_schema(conn)

    @app.get("/health")
    def health():
        return {"status": "ok", "version": "0.1.0"}

    @app.get("/signals")
    def list_signals(limit: int = 20):
        limit = max(1, min(limit, 1000))
        rows = conn.execute(
            "SELECT * FROM signal_events ORDER BY triggered_at DESC LIMIT ?", [limit]
        ).fetchall()
        cols = [
            "id", "asset", "direction", "confidence", "regime",
            "narrative_velocity", "narrative_tipping_point", "mechanism",
            "estimated_hours", "citations", "triggered_at",
        ]
        return [dict(zip(cols, row)) for row in rows]

    @app.get("/signals/latest")
    def latest_signal():
        row = conn.execute(
            "SELECT * FROM signal_events ORDER BY triggered_at DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return {"detail": "no signals yet"}
        cols = [
            "id", "asset", "direction", "confidence", "regime",
            "narrative_velocity", "narrative_tipping_point", "mechanism",
            "estimated_hours", "citations", "triggered_at",
        ]
        return dict(zip(cols, row))

    return app
