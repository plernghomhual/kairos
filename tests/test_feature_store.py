from datetime import datetime, timedelta, timezone

import pytest

from kairos.feature_store import FeatureStore
from kairos.signals.ensemble import FeatureVector


def _assert_fv_close(actual: FeatureVector, expected: FeatureVector) -> None:
    for name in expected.__dataclass_fields__:
        got = getattr(actual, name)
        want = getattr(expected, name)
        if isinstance(want, bool):
            assert got is want
        else:
            assert got == pytest.approx(want)


def _fv(kalman_slope: float = 1.0) -> FeatureVector:
    return FeatureVector(
        kalman_slope=kalman_slope,
        volume_z_score=0.2,
        anomaly_score=0.0,
        narrative_velocity=0.03,
        narrative_tipping_point=True,
        saturation=0.6,
        regime_lv_up=1.0,
        regime_hv_up=0.0,
        regime_lv_down=0.0,
        regime_hv_down=0.0,
        causal_bullish=0.7,
        causal_confidence=0.8,
        macro_dff=0.25,
    )


def test_get_latest_returns_none_when_empty(tmp_path):
    fs = FeatureStore(str(tmp_path / "features.db"))
    try:
        assert fs.get_latest("BTC") is None
        assert fs.get_feature_count("BTC") == 0
    finally:
        fs.close()


def test_store_feature_roundtrips_and_persists_across_restarts(tmp_path):
    db_path = str(tmp_path / "features.db")
    ts = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    metadata = {"candle_count": 365, "signal": {"direction": "bullish"}}
    expected = _fv()

    fs = FeatureStore(db_path)
    try:
        fs.store_feature("BTC", ts, expected, metadata)
        rows = fs.get_features("BTC", ts - timedelta(seconds=1))
        assert len(rows) == 1
        got_ts, got_fv, got_metadata = rows[0]
        assert got_ts == ts
        _assert_fv_close(got_fv, expected)
        assert got_metadata == metadata
    finally:
        fs.close()

    reopened = FeatureStore(db_path)
    try:
        latest = reopened.get_latest("BTC")
        assert latest is not None
        got_ts, got_fv, got_metadata = latest
        assert got_ts == ts
        _assert_fv_close(got_fv, expected)
        assert got_metadata == metadata
        assert reopened.get_feature_count("BTC") == 1
    finally:
        reopened.close()


def test_feature_statistics_and_prune(tmp_path):
    db_path = str(tmp_path / "features.db")
    fs = FeatureStore(db_path)
    try:
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for i in range(3):
            fs.store_feature("BTC", base + timedelta(days=i), _fv(float(i)), {"i": i})

        stats = fs.get_statistics("BTC")
        assert stats["count"] == 3
        assert stats["oldest_ts"] == base
        assert stats["newest_ts"] == base + timedelta(days=2)
        assert stats["field_stats"]["kalman_slope"]["mean"] == pytest.approx(1.0)

        assert fs.prune("BTC", keep_last=2) == 1
        assert fs.get_feature_count("BTC") == 2
        remaining = fs.get_features("BTC", base - timedelta(seconds=1))
        assert [ts for ts, _, _ in remaining] == [
            base + timedelta(days=2),
            base + timedelta(days=1),
        ]
    finally:
        fs.close()
