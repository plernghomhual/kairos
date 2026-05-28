import duckdb


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
            triggered_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
