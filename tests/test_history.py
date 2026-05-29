"""Tests for signal history persistence and hit-rate tracking."""
from datetime import datetime, timezone

import duckdb
import pytest

from kairos.db import (
    create_schema,
    get_hit_rate,
    get_signal_history,
    resolve_expired_signals,
    save_signal,
)
from kairos.models.signal_event import SignalEvent


def _make_event(direction: str = "bullish", estimated_hours: float = 24.0, asset: str = "BTC") -> SignalEvent:
    return SignalEvent(
        asset=asset,
        direction=direction,
        confidence=0.75,
        regime="accumulation",
        narrative_velocity=0.5,
        narrative_tipping_point=False,
        mechanism="test mechanism",
        estimated_hours=estimated_hours,
        citations=["source1"],
    )


def _make_conn() -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(":memory:")
    create_schema(conn)
    return conn


# ---------------------------------------------------------------------------
# test_save_and_retrieve_signal
# ---------------------------------------------------------------------------

def test_save_and_retrieve_signal():
    conn = _make_conn()
    event = _make_event(direction="bullish")
    save_signal(conn, event, price_at_signal=50_000.0)

    rows = get_signal_history(conn, asset="BTC", limit=10)
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == event.id
    assert row["direction"] == "bullish"
    assert row["price_at_signal"] == 50_000.0
    assert row["outcome"] is None


# ---------------------------------------------------------------------------
# test_resolve_correct_bullish
# ---------------------------------------------------------------------------

def test_resolve_correct_bullish():
    conn = _make_conn()
    # Insert directly so triggered_at is in the past and the signal is expired
    conn.execute(
        """
        INSERT INTO signal_events (
            id, asset, direction, confidence, regime,
            narrative_velocity, narrative_tipping_point, mechanism,
            estimated_hours, citations, triggered_at,
            price_at_signal, price_at_expiry, outcome
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)
        """,
        ["id-bull-correct", "BTC", "bullish", 0.8, "accumulation",
         0.5, False, "test", 1, "[]", "2020-01-01 00:00:00",
         40_000.0],
    )
    resolved = resolve_expired_signals(conn, "BTC", current_price=45_000.0)
    assert resolved == 1
    row = conn.execute("SELECT outcome FROM signal_events WHERE id = 'id-bull-correct'").fetchone()
    assert row[0] == "correct"


# ---------------------------------------------------------------------------
# test_resolve_correct_bearish
# ---------------------------------------------------------------------------

def test_resolve_correct_bearish():
    conn = _make_conn()
    conn.execute(
        """
        INSERT INTO signal_events (
            id, asset, direction, confidence, regime,
            narrative_velocity, narrative_tipping_point, mechanism,
            estimated_hours, citations, triggered_at,
            price_at_signal, price_at_expiry, outcome
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)
        """,
        ["id-bear-correct", "BTC", "bearish", 0.8, "distribution",
         0.5, False, "test", 1, "[]", "2020-01-01 00:00:00",
         40_000.0],
    )
    resolved = resolve_expired_signals(conn, "BTC", current_price=35_000.0)
    assert resolved == 1
    row = conn.execute("SELECT outcome FROM signal_events WHERE id = 'id-bear-correct'").fetchone()
    assert row[0] == "correct"


# ---------------------------------------------------------------------------
# test_resolve_incorrect
# ---------------------------------------------------------------------------

def test_resolve_incorrect():
    conn = _make_conn()
    conn.execute(
        """
        INSERT INTO signal_events (
            id, asset, direction, confidence, regime,
            narrative_velocity, narrative_tipping_point, mechanism,
            estimated_hours, citations, triggered_at,
            price_at_signal, price_at_expiry, outcome
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)
        """,
        ["id-bull-wrong", "BTC", "bullish", 0.6, "transition",
         0.3, False, "test", 1, "[]", "2020-01-01 00:00:00",
         50_000.0],
    )
    resolved = resolve_expired_signals(conn, "BTC", current_price=45_000.0)
    assert resolved == 1
    row = conn.execute("SELECT outcome FROM signal_events WHERE id = 'id-bull-wrong'").fetchone()
    assert row[0] == "incorrect"


# ---------------------------------------------------------------------------
# test_hit_rate_calculation — 3 correct + 1 incorrect → 0.75
# ---------------------------------------------------------------------------

def test_hit_rate_calculation():
    conn = _make_conn()
    for i, outcome in enumerate(["correct", "correct", "correct", "incorrect"]):
        conn.execute(
            """
            INSERT INTO signal_events (
                id, asset, direction, confidence, regime,
                narrative_velocity, narrative_tipping_point, mechanism,
                estimated_hours, citations, triggered_at,
                price_at_signal, price_at_expiry, outcome
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [f"id-{i}", "BTC", "bullish", 0.7, "accumulation",
             0.4, False, "test", 1, "[]", "2020-01-01 00:00:00",
             40_000.0, 42_000.0, outcome],
        )
    stats = get_hit_rate(conn, asset="BTC")
    assert stats["total_resolved"] == 4
    assert stats["correct"] == 3
    assert stats["incorrect"] == 1
    assert abs(stats["hit_rate"] - 0.75) < 1e-9


# ---------------------------------------------------------------------------
# test_unresolved_not_counted
# ---------------------------------------------------------------------------

def test_unresolved_not_counted():
    conn = _make_conn()
    # One resolved, one unresolved
    conn.execute(
        """
        INSERT INTO signal_events (
            id, asset, direction, confidence, regime,
            narrative_velocity, narrative_tipping_point, mechanism,
            estimated_hours, citations, triggered_at,
            price_at_signal, price_at_expiry, outcome
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ["id-resolved", "BTC", "bullish", 0.7, "accumulation",
         0.4, False, "test", 1, "[]", "2020-01-01 00:00:00",
         40_000.0, 42_000.0, "correct"],
    )
    conn.execute(
        """
        INSERT INTO signal_events (
            id, asset, direction, confidence, regime,
            narrative_velocity, narrative_tipping_point, mechanism,
            estimated_hours, citations, triggered_at,
            price_at_signal, price_at_expiry, outcome
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)
        """,
        ["id-pending", "BTC", "bullish", 0.8, "accumulation",
         0.5, False, "test", 24, "[]", datetime.now(timezone.utc).isoformat(),
         50_000.0],
    )
    stats = get_hit_rate(conn, asset="BTC")
    assert stats["total_resolved"] == 1
    assert stats["correct"] == 1
    assert stats["hit_rate"] == 1.0
