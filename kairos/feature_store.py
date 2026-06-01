"""Local feature-vector persistence for live and backtest analytics."""

import json
import os
from dataclasses import fields
from datetime import datetime, timezone
from typing import Any

from rich.console import Console
from rich.table import Table

from kairos.db import create_schema, get_connection
from kairos.signals.ensemble import FeatureVector

_FEATURE_FIELDS = tuple(field.name for field in fields(FeatureVector))

# Track which db paths have already had create_schema run so we do not pay DDL
# overhead on every FeatureStore instantiation in hot-path code.
_SCHEMA_INITIALIZED: set[str] = set()


def _resolve_db_path(db_path: str) -> str:
    if db_path == "kairos.db":
        return os.getenv("KAIROS_DB_PATH", db_path)
    return db_path


def _asset_key(asset: str) -> str:
    return asset.upper()


def _to_db_ts(ts: datetime) -> datetime:
    if ts.tzinfo is not None:
        return ts.astimezone(timezone.utc).replace(tzinfo=None)
    return ts


def _from_db_ts(ts: datetime) -> datetime:
    if ts.tzinfo is not None:
        return ts.astimezone(timezone.utc)
    return ts.replace(tzinfo=timezone.utc)


def _metadata_to_json(metadata: dict | None) -> str:
    return json.dumps(metadata or {}, sort_keys=True)


def _metadata_from_json(raw: Any) -> dict:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        return json.loads(raw) if raw else {}
    return json.loads(str(raw))


def _feature_from_row(row: tuple) -> FeatureVector:
    values = {}
    for idx, name in enumerate(_FEATURE_FIELDS):
        raw = row[idx]
        if name == "narrative_tipping_point":
            values[name] = bool(raw)
        else:
            values[name] = 0.0 if raw is None else float(raw)
    return FeatureVector(**values)


class FeatureStore:
    def __init__(self, db_path: str = "kairos.db"):
        """Connect to the project DuckDB database and ensure schema exists."""
        self.db_path = _resolve_db_path(db_path)
        self.conn = get_connection(self.db_path)
        if self.db_path not in _SCHEMA_INITIALIZED:
            create_schema(self.conn)
            _SCHEMA_INITIALIZED.add(self.db_path)

    def close(self) -> None:
        self.conn.close()

    def store_feature(
        self,
        asset: str,
        ts: datetime,
        fv: FeatureVector,
        metadata: dict | None = None,
    ) -> None:
        """Persist a single feature vector with its timestamp."""
        values = [getattr(fv, name) for name in _FEATURE_FIELDS]
        self.conn.execute(
            f"""
            INSERT INTO feature_store (
                asset, ts, {", ".join(_FEATURE_FIELDS)}, metadata
            ) VALUES (
                ?, ?, {", ".join(["?"] * len(_FEATURE_FIELDS))}, ?
            )
            ON CONFLICT(asset, ts) DO UPDATE SET
                {", ".join(f"{col} = excluded.{col}" for col in _FEATURE_FIELDS)},
                metadata = excluded.metadata
            """,
            [_asset_key(asset), _to_db_ts(ts), *values, _metadata_to_json(metadata)],
        )

    def get_features(
        self,
        asset: str,
        since: datetime,
        limit: int = 1000,
    ) -> list[tuple[datetime, FeatureVector, dict]]:
        """Retrieve feature vectors since a timestamp, newest first."""
        rows = self.conn.execute(
            f"""
            SELECT ts, {", ".join(_FEATURE_FIELDS)}, metadata
            FROM feature_store
            WHERE asset = ? AND ts >= ?
            ORDER BY ts DESC
            LIMIT ?
            """,
            [_asset_key(asset), _to_db_ts(since), limit],
        ).fetchall()
        return [
            (
                _from_db_ts(row[0]),
                _feature_from_row(row[1:-1]),
                _metadata_from_json(row[-1]),
            )
            for row in rows
        ]

    def get_latest(self, asset: str) -> tuple[datetime, FeatureVector, dict] | None:
        """Get most recent feature vector for an asset."""
        row = self.conn.execute(
            f"""
            SELECT ts, {", ".join(_FEATURE_FIELDS)}, metadata
            FROM feature_store
            WHERE asset = ?
            ORDER BY ts DESC
            LIMIT 1
            """,
            [_asset_key(asset)],
        ).fetchone()
        if row is None:
            return None
        return (
            _from_db_ts(row[0]),
            _feature_from_row(row[1:-1]),
            _metadata_from_json(row[-1]),
        )

    def get_feature_count(self, asset: str) -> int:
        """Total stored vectors for an asset."""
        row = self.conn.execute(
            "SELECT COUNT(*) FROM feature_store WHERE asset = ?",
            [_asset_key(asset)],
        ).fetchone()
        return int(row[0])

    def prune(self, asset: str, keep_last: int = 1000) -> int:
        """Delete oldest vectors beyond keep_last. Returns count removed."""
        keep_last = max(0, keep_last)
        total = self.get_feature_count(asset)
        to_remove = max(0, total - keep_last)
        if to_remove == 0:
            return 0
        self.conn.execute(
            """
            DELETE FROM feature_store
            WHERE asset = ? AND ts IN (
                SELECT ts
                FROM feature_store
                WHERE asset = ?
                ORDER BY ts ASC
                LIMIT ?
            )
            """,
            [_asset_key(asset), _asset_key(asset), to_remove],
        )
        return to_remove

    def get_statistics(self, asset: str) -> dict:
        """Return count, oldest_ts, newest_ts, field_stats (mean/std per field)."""
        row = self.conn.execute(
            """
            SELECT COUNT(*), MIN(ts), MAX(ts)
            FROM feature_store
            WHERE asset = ?
            """,
            [_asset_key(asset)],
        ).fetchone()
        count = int(row[0])
        if count == 0:
            return {
                "count": 0,
                "oldest_ts": None,
                "newest_ts": None,
                "field_stats": {},
            }

        # Build a single aggregation query for all fields to avoid N+1 round trips.
        agg_exprs = []
        for name in _FEATURE_FIELDS:
            expr = f'CAST("{name}" AS DOUBLE)' if name == "narrative_tipping_point" else f'"{name}"'
            agg_exprs.append(f"AVG({expr})")
            agg_exprs.append(f"STDDEV_POP({expr})")
        agg_row = self.conn.execute(
            f"SELECT {', '.join(agg_exprs)} FROM feature_store WHERE asset = ?",
            [_asset_key(asset)],
        ).fetchone()

        field_stats = {}
        for i, name in enumerate(_FEATURE_FIELDS):
            mean_val = agg_row[i * 2]
            std_val = agg_row[i * 2 + 1]
            field_stats[name] = {
                "mean": 0.0 if mean_val is None else float(mean_val),
                "std": 0.0 if std_val is None else float(std_val),
            }

        return {
            "count": count,
            "oldest_ts": _from_db_ts(row[1]),
            "newest_ts": _from_db_ts(row[2]),
            "field_stats": field_stats,
        }


def print_feature_stats(asset: str) -> None:
    """Print a rich table of feature statistics for an asset."""
    console = Console()
    fs = FeatureStore()
    try:
        stats = fs.get_statistics(asset)
    finally:
        fs.close()

    if stats["count"] == 0:
        console.print(f"No feature vectors stored for {asset.upper()}.", style="dim")
        return

    table = Table(title=f"Feature Store Statistics — {asset.upper()}")
    table.add_column("Field", style="dim")
    table.add_column("Mean", justify="right")
    table.add_column("Std", justify="right")
    for name, values in stats["field_stats"].items():
        table.add_row(name, f"{values['mean']:.6f}", f"{values['std']:.6f}")

    console.print(
        f"{stats['count']} vectors from {stats['oldest_ts'].isoformat()} " f"to {stats['newest_ts'].isoformat()}"
    )
    console.print(table)
