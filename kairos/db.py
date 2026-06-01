import json
import logging
import os
import pathlib
import time
from dataclasses import fields
from datetime import datetime, timezone

import duckdb

from kairos.models.signal_event import SignalEvent

_logger = logging.getLogger(__name__)
_DB_RETRY_MAX = 5
_DB_RETRY_BACKOFF = 0.5


def _resolve_db_path(db_path: str) -> str:
    if db_path == "kairos.db":
        return os.getenv("KAIROS_DB_PATH", db_path)
    return db_path


def get_connection(db_path: str = "kairos.db") -> duckdb.DuckDBPyConnection:
    """Connect with retry on lock contention."""
    db_path = _resolve_db_path(db_path)
    pathlib.Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    backoff = _DB_RETRY_BACKOFF
    last_exc: Exception | None = None
    for attempt in range(1, _DB_RETRY_MAX + 1):
        try:
            return duckdb.connect(db_path)
        except (duckdb.IOException, OSError) as exc:
            if "lock" in str(exc).lower():
                last_exc = exc
                _logger.warning(
                    "DuckDB lock conflict (attempt %d/%d); retrying in %.1fs",
                    attempt,
                    _DB_RETRY_MAX,
                    backoff,
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, 4.0)
            else:
                raise
    raise RuntimeError(
        f"Could not acquire DuckDB connection to {db_path!r} after {_DB_RETRY_MAX} attempts"
    ) from last_exc


def _feature_store_columns() -> str:
    from kairos.signals.ensemble import FeatureVector

    cols = []
    for f in fields(FeatureVector):
        sql_type = "BOOLEAN" if f.name == "narrative_tipping_point" else "DOUBLE"
        cols.append(f"    {f.name} {sql_type}")
    return ",\n".join(cols)


def create_schema(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS price_candles (
            asset       VARCHAR,
            ts          TIMESTAMP,
            open        DOUBLE,
            high        DOUBLE,
            low         DOUBLE,
            close       DOUBLE,
            volume      DOUBLE,
            PRIMARY KEY (asset, ts)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw_news (
            id          VARCHAR PRIMARY KEY,
            source      VARCHAR,
            title       VARCHAR,
            body        VARCHAR,
            published   TIMESTAMP,
            fetched_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw_posts (
            id          VARCHAR PRIMARY KEY,
            subreddit   VARCHAR,
            title       VARCHAR,
            body        VARCHAR,
            score       INTEGER,
            num_comments INTEGER,
            created_at  TIMESTAMP,
            fetched_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS macro_data (
            series_id   VARCHAR,
            ts          DATE,
            value       DOUBLE,
            PRIMARY KEY (series_id, ts)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS signal_events (
            id                      VARCHAR PRIMARY KEY,
            asset                   VARCHAR,
            direction               VARCHAR,
            confidence              DOUBLE,
            regime                  VARCHAR,
            narrative_velocity      DOUBLE,
            narrative_tipping_point BOOLEAN,
            mechanism               VARCHAR,
            estimated_hours         DOUBLE,
            citations               VARCHAR,
            triggered_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            price_at_signal         DOUBLE,
            price_at_expiry         DOUBLE,
            outcome                 VARCHAR
        )
    """)
    # Migrations — safe to run on old DBs (IF NOT EXISTS guards)
    conn.execute("ALTER TABLE signal_events ADD COLUMN IF NOT EXISTS price_at_signal DOUBLE")
    conn.execute("ALTER TABLE signal_events ADD COLUMN IF NOT EXISTS price_at_expiry DOUBLE")
    conn.execute("ALTER TABLE signal_events ADD COLUMN IF NOT EXISTS outcome VARCHAR")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_signal_asset_ts ON signal_events (asset, triggered_at DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_signal_asset_outcome ON signal_events (asset, outcome)")
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS feature_store (
            asset   VARCHAR,
            ts      TIMESTAMP,
{_feature_store_columns()},
            metadata VARCHAR,
            PRIMARY KEY (asset, ts)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sentiment_cache (
            source      VARCHAR,
            asset       VARCHAR,
            raw_data    JSON,
            fetched_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (source, asset)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS exchange_wallets (
            address VARCHAR PRIMARY KEY,
            exchange VARCHAR,
            label   VARCHAR
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS whale_transfers (
            signature   VARCHAR PRIMARY KEY,
            mint        VARCHAR,
            from_wallet VARCHAR,
            to_wallet   VARCHAR,
            usd_value   DOUBLE,
            direction   VARCHAR,
            slot        BIGINT,
            block_time  VARCHAR,
            detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS paper_trades (
            asset             VARCHAR,
            signal_id         VARCHAR,
            direction         VARCHAR,
            entry_price       DOUBLE,
            entry_time        TIMESTAMP,
            size              DOUBLE,
            exit_price        DOUBLE,
            exit_time         TIMESTAMP,
            pnl_pct           DOUBLE,
            closed            BOOLEAN DEFAULT FALSE,
            signal_confidence DOUBLE,
            signal_regime     VARCHAR,
            high_watermark    DOUBLE,
            low_watermark     DOUBLE,
            PRIMARY KEY (signal_id, asset)
        )
    """)
    conn.execute("ALTER TABLE paper_trades ADD COLUMN IF NOT EXISTS high_watermark DOUBLE")
    conn.execute("ALTER TABLE paper_trades ADD COLUMN IF NOT EXISTS low_watermark DOUBLE")


def save_signal(
    conn: duckdb.DuckDBPyConnection,
    event: SignalEvent,
    price_at_signal: float,
) -> None:
    """Persist a SignalEvent with the price at the time of signal generation."""
    conn.execute(
        """
        INSERT INTO signal_events (
            id, asset, direction, confidence, regime,
            narrative_velocity, narrative_tipping_point, mechanism,
            estimated_hours, citations, triggered_at,
            price_at_signal, price_at_expiry, outcome
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)
        ON CONFLICT(id) DO UPDATE SET
            asset                   = excluded.asset,
            direction               = excluded.direction,
            confidence              = excluded.confidence,
            regime                  = excluded.regime,
            narrative_velocity      = excluded.narrative_velocity,
            narrative_tipping_point = excluded.narrative_tipping_point,
            mechanism               = excluded.mechanism,
            estimated_hours         = excluded.estimated_hours,
            citations               = excluded.citations,
            triggered_at            = excluded.triggered_at,
            price_at_signal         = excluded.price_at_signal
        """,
        [
            event.id,
            event.asset,
            event.direction,
            event.confidence,
            event.regime,
            event.narrative_velocity,
            event.narrative_tipping_point,
            event.mechanism,
            event.estimated_hours,
            json.dumps(event.citations),
            event.triggered_at,
            price_at_signal,
        ],
    )


def resolve_expired_signals(
    conn: duckdb.DuckDBPyConnection,
    asset: str,
    current_price: float,
) -> int:
    """
    For every unresolved signal whose expiry time has passed, compute outcome
    and write price_at_expiry + outcome.

    Returns the number of signals resolved.
    """
    now = datetime.now(timezone.utc)
    rows = conn.execute(
        """
        SELECT id, direction, price_at_signal, triggered_at, estimated_hours
        FROM signal_events
        WHERE asset = ?
          AND outcome IS NULL
          AND price_at_signal IS NOT NULL
          AND triggered_at + INTERVAL (CAST(estimated_hours * 3600 AS INTEGER)) SECOND <= ?
        """,
        [asset, now],
    ).fetchall()

    # MVP note: outcome is evaluated against current_price at resolution time, not
    # the exact price at signal expiry. Accuracy degrades if kairos run is called
    # many hours after a signal expired. Full solution requires stored price history.
    resolved = 0
    for row_id, direction, price_at_signal, triggered_at, estimated_hours in rows:
        if direction == "bullish":
            outcome = "correct" if current_price > price_at_signal else "incorrect"
        elif direction == "bearish":
            outcome = "correct" if current_price < price_at_signal else "incorrect"
        else:
            outcome = "incorrect"

        conn.execute(
            """
            UPDATE signal_events
            SET price_at_expiry = ?, outcome = ?
            WHERE id = ?
            """,
            [current_price, outcome, row_id],
        )
        resolved += 1

    return resolved


def get_signal_history(
    conn: duckdb.DuckDBPyConnection,
    asset: str = "BTC",
    limit: int = 20,
) -> list[dict]:
    """Return the most recent signals for an asset as a list of dicts."""
    rows = conn.execute(
        """
        SELECT id, asset, direction, confidence, regime,
               narrative_velocity, narrative_tipping_point, mechanism,
               estimated_hours, citations, triggered_at,
               price_at_signal, price_at_expiry, outcome
        FROM signal_events
        WHERE asset = ?
        ORDER BY triggered_at DESC
        LIMIT ?
        """,
        [asset, limit],
    ).fetchall()

    columns = [
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
        "price_at_signal",
        "price_at_expiry",
        "outcome",
    ]
    results = []
    for row in rows:
        d = dict(zip(columns, row))
        raw_citations = d.get("citations")
        if isinstance(raw_citations, str):
            try:
                d["citations"] = json.loads(raw_citations)
            except (json.JSONDecodeError, ValueError):
                d["citations"] = []
        results.append(d)
    return results


def get_hit_rate(
    conn: duckdb.DuckDBPyConnection,
    asset: str = "BTC",
) -> dict:
    """
    Return hit-rate statistics for resolved signals of an asset.

    Keys: total_resolved, correct, incorrect, hit_rate (0.0–1.0 or None).
    """
    row = conn.execute(
        """
        SELECT
            COUNT(*) FILTER (WHERE outcome IS NOT NULL)        AS total_resolved,
            COUNT(*) FILTER (WHERE outcome = 'correct')        AS correct,
            COUNT(*) FILTER (WHERE outcome = 'incorrect')      AS incorrect
        FROM signal_events
        WHERE asset = ?
        """,
        [asset],
    ).fetchone()

    total_resolved, correct, incorrect = row
    hit_rate = (correct / total_resolved) if total_resolved > 0 else None
    return {
        "total_resolved": total_resolved,
        "correct": correct,
        "incorrect": incorrect,
        "hit_rate": hit_rate,
    }
