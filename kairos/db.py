import json
from datetime import datetime, timezone

import duckdb

from kairos.models.signal_event import SignalEvent


def get_connection(db_path: str = "kairos.db") -> duckdb.DuckDBPyConnection:
    return duckdb.connect(db_path)


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


def save_signal(
    conn: duckdb.DuckDBPyConnection,
    event: SignalEvent,
    price_at_signal: float,
) -> None:
    """Persist a SignalEvent with the price at the time of signal generation."""
    conn.execute(
        """
        INSERT OR REPLACE INTO signal_events (
            id, asset, direction, confidence, regime,
            narrative_velocity, narrative_tipping_point, mechanism,
            estimated_hours, citations, triggered_at,
            price_at_signal, price_at_expiry, outcome
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)
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
        "id", "asset", "direction", "confidence", "regime",
        "narrative_velocity", "narrative_tipping_point", "mechanism",
        "estimated_hours", "citations", "triggered_at",
        "price_at_signal", "price_at_expiry", "outcome",
    ]
    return [dict(zip(columns, row)) for row in rows]


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
