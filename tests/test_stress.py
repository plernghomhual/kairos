"""
Stress tests: edge cases, adversarial inputs, invariant checks.
"""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from kairos.signals.kalman import kalman_smooth
from kairos.signals.anomaly import detect_anomalies
from kairos.signals.narrative import compute_narrative_features
from kairos.signals.regime import fit_regime_model, predict_regime
from kairos.signals.causal import CausalDAG
from kairos.signals.ensemble import SignalEnsemble, FeatureVector
from kairos.backtest.runner import evaluate_hit_rate
from kairos.api.server import create_app


# ── Kalman ────────────────────────────────────────────────────────────────────

def test_kalman_empty_returns_empty():
    r = kalman_smooth(np.array([]))
    assert len(r) == 0
    assert r.dtype == np.float64


def test_kalman_single_element():
    r = kalman_smooth(np.array([42.0]))
    assert len(r) == 1
    assert r[0] == pytest.approx(42.0)


def test_kalman_constant_signal():
    r = kalman_smooth(np.ones(100) * 50000)
    assert np.allclose(r, 50000, atol=1e-4)


def test_kalman_output_dtype_float64():
    r = kalman_smooth(np.array([1.0, 2.0, 3.0]))
    assert r.dtype == np.float64


def test_kalman_smooths_spike():
    base = np.ones(50) * 100.0
    base[25] = 10000.0
    r = kalman_smooth(base)
    assert r[25] < 10000.0  # spike is damped


def test_kalman_preserves_length():
    for n in [2, 10, 100, 1000]:
        r = kalman_smooth(np.random.rand(n))
        assert len(r) == n


# ── Anomaly ───────────────────────────────────────────────────────────────────

def test_anomaly_single_sample():
    r = detect_anomalies(np.array([[1.0, 2.0]]))
    assert len(r) == 1


def test_anomaly_output_is_bool():
    np.random.seed(0)
    r = detect_anomalies(np.random.rand(50, 2))
    assert r.dtype == bool


def test_anomaly_all_identical():
    r = detect_anomalies(np.ones((100, 2)))
    # all identical → no anomalies (can't separate anything)
    assert r.sum() == 0


def test_anomaly_flags_clear_outlier():
    np.random.seed(42)
    data = np.random.normal(0, 1, (200, 2))
    data[-1] = [100.0, 100.0]
    r = detect_anomalies(data)
    assert r[-1] == True


def test_anomaly_contamination_rate():
    np.random.seed(42)
    data = np.random.normal(0, 1, (200, 2))
    r = detect_anomalies(data, contamination=0.1)
    rate = r.sum() / len(r)
    assert rate <= 0.15  # within contamination band


# ── Narrative ─────────────────────────────────────────────────────────────────

def test_narrative_empty_no_crash():
    r = compute_narrative_features([])
    assert r["narrative_velocity"] == 0.0
    assert r["narrative_tipping_point"] == False
    assert r["saturation"] == 0.0


def test_narrative_single_no_crash():
    r = compute_narrative_features([500])
    assert 0.0 <= r["saturation"] <= 1.0


def test_narrative_saturation_clamped_to_1():
    # 15000 posts, population=1000 → raw saturation=15 → should clamp to 1.0
    r = compute_narrative_features([5000, 5000, 5000], population=1000)
    assert r["saturation"] == pytest.approx(1.0)


def test_narrative_all_zeros():
    r = compute_narrative_features([0, 0, 0, 0])
    assert r["narrative_velocity"] == 0.0
    assert r["narrative_tipping_point"] == False


def test_narrative_velocity_positive_for_growth():
    r = compute_narrative_features([10, 20, 40, 80, 160])
    assert r["narrative_velocity"] > 0


def test_narrative_tipping_point_triggers():
    # fast growth, low saturation; population=10_000 keeps velocity > 0.1 threshold
    r = compute_narrative_features([100, 200, 400, 800, 1600], population=10_000)
    assert r["narrative_tipping_point"] == True


# ── Causal DAG ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def dag():
    return CausalDAG()


def test_causal_bullish_plus_bearish_equals_one(dag):
    combos = [
        (a, b, c, d)
        for a in [False, True]
        for b in ["accumulation", "distribution", "transition", "unknown"]
        for c in [False, True]
        for d in [False, True]
    ]
    for ntp, regime, anom, macro in combos:
        r = dag.infer_price_impact(ntp, regime, anom, macro)
        assert round(r["bullish"] + r["bearish"], 4) == 1.0, (
            f"bullish+bearish != 1.0 for {ntp},{regime},{anom},{macro}: {r}"
        )


def test_causal_confidence_in_range(dag):
    for anom in [False, True]:
        for regime in ["accumulation", "distribution", "transition"]:
            r = dag.infer_price_impact(True, regime, anom, False)
            assert 0.0 <= r["confidence"] <= 1.0


def test_causal_unknown_regime_is_neutral(dag):
    r = dag.infer_price_impact(False, "totally_unknown", False, False)
    assert r["bullish"] == pytest.approx(0.5)
    assert r["citations"] == []


def test_causal_tipping_increases_bullish(dag):
    without = dag.infer_price_impact(False, "accumulation", False, False)
    with_ = dag.infer_price_impact(True, "accumulation", False, False)
    assert with_["bullish"] > without["bullish"]


def test_causal_macro_stress_decreases_bullish(dag):
    without = dag.infer_price_impact(True, "accumulation", False, False)
    with_ = dag.infer_price_impact(True, "accumulation", False, True)
    assert with_["bullish"] < without["bullish"]


def test_causal_anomaly_reduces_confidence(dag):
    without = dag.infer_price_impact(True, "transition", False, False)
    with_ = dag.infer_price_impact(True, "transition", True, False)
    assert with_["confidence"] < without["confidence"]


# ── Ensemble ──────────────────────────────────────────────────────────────────

def _fv(bullish: bool = True) -> FeatureVector:
    return FeatureVector(
        kalman_slope=0.02 if bullish else -0.02,
        volume_z_score=1.5 if bullish else -1.5,
        anomaly_score=0.1,
        narrative_velocity=2.3 if bullish else 0.1,
        narrative_tipping_point=bullish,
        saturation=0.15,
        regime_accumulation=1.0 if bullish else 0.0,
        regime_distribution=0.0 if bullish else 1.0,
        regime_transition=0.0,
        causal_bullish=0.75 if bullish else 0.25,
        causal_confidence=0.85,
        macro_dff=0.25,
    )


@pytest.fixture(scope="module")
def fitted_ensemble():
    e = SignalEnsemble()
    e.fit([_fv(True)] * 50 + [_fv(False)] * 50)
    return e


def test_ensemble_empty_fit_raises():
    with pytest.raises(ValueError, match="empty"):
        SignalEnsemble().fit([])


def test_ensemble_single_class_raises():
    with pytest.raises(ValueError, match="bullish"):
        SignalEnsemble().fit([_fv(True)] * 20)


def test_ensemble_unfitted_predict_raises():
    with pytest.raises(RuntimeError, match="fit()"):
        SignalEnsemble().predict("BTC", _fv(), citations=[])


def test_ensemble_confidence_in_range(fitted_ensemble):
    for bullish in [True, False]:
        ev = fitted_ensemble.predict("BTC", _fv(bullish), citations=[])
        assert 0.0 <= ev.confidence <= 1.0


def test_ensemble_direction_valid(fitted_ensemble):
    for bullish in [True, False]:
        ev = fitted_ensemble.predict("BTC", _fv(bullish), citations=[])
        assert ev.direction in ("bullish", "bearish")


def test_ensemble_hours_within_bounds(fitted_ensemble):
    # Quiet accumulation → moderate estimate within valid range
    fv = FeatureVector(0.01, 0.5, 0.0, 0.0, True, 0.1, 1.0, 0.0, 0.0, 0.7, 0.9, 0.25)
    ev = fitted_ensemble.predict("BTC", fv, citations=[])
    assert 12.0 <= ev.estimated_hours <= 168.0


def test_ensemble_transition_anomaly_hours_shorter(fitted_ensemble):
    # Transition + anomaly + high vol → faster than quiet accumulation
    fv_quiet = FeatureVector(0.01, 0.5, 0.0, 0.0, False, 0.1, 1.0, 0.0, 0.0, 0.7, 0.9, 0.25)
    fv_hot = FeatureVector(0.05, 2.0, 1.0, 0.5, True, 0.1, 0.0, 0.0, 1.0, 0.8, 0.9, 0.25)
    ev_quiet = fitted_ensemble.predict("BTC", fv_quiet, citations=[], regime="accumulation")
    ev_hot = fitted_ensemble.predict("BTC", fv_hot, citations=[], regime="transition")
    assert ev_hot.estimated_hours < ev_quiet.estimated_hours


def test_ensemble_citations_passed_through(fitted_ensemble):
    cites = ["Shiller 2017", "Minsky Stage 2"]
    ev = fitted_ensemble.predict("BTC", _fv(True), citations=cites)
    assert ev.citations == cites


# ── Backtest ──────────────────────────────────────────────────────────────────

def _sig():
    return {"direction": "bullish", "triggered_at": "2021-01-01", "estimated_hours": 36}


def test_backtest_empty():
    r = evaluate_hit_rate([], [])
    assert r.total == 0
    assert r.hit_rate == 0.0
    assert r.passes_threshold == False


def test_backtest_100_percent():
    r = evaluate_hit_rate([_sig()] * 10, [True] * 10)
    assert r.hit_rate == 1.0
    assert r.passes_threshold == True


def test_backtest_zero_percent():
    r = evaluate_hit_rate([_sig()] * 10, [False] * 10)
    assert r.hit_rate == 0.0
    assert r.passes_threshold == False


def test_backtest_exactly_threshold():
    # 7/10 = 0.7 → passes
    r = evaluate_hit_rate([_sig()] * 10, [True] * 7 + [False] * 3)
    assert r.passes_threshold == True


def test_backtest_just_below_threshold():
    # 6/10 = 0.6 → fails
    r = evaluate_hit_rate([_sig()] * 10, [True] * 6 + [False] * 4)
    assert r.passes_threshold == False


def test_backtest_length_mismatch_raises():
    with pytest.raises(ValueError):
        evaluate_hit_rate([_sig()] * 5, [True] * 10)


# ── API ───────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def api_client(tmp_path_factory):
    db = str(tmp_path_factory.mktemp("db") / "stress.db")
    import duckdb
    from kairos.db import create_schema
    conn = duckdb.connect(db)
    create_schema(conn)
    conn.close()
    app = create_app(db_path=db)
    return TestClient(app)


def test_api_health(api_client):
    r = api_client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_api_signals_empty(api_client):
    r = api_client.get("/signals")
    assert r.status_code == 200
    assert r.json() == []


def test_api_latest_empty(api_client):
    r = api_client.get("/signals/latest")
    assert r.status_code == 200
    assert "detail" in r.json()


def test_api_negative_limit_clamped(api_client):
    r = api_client.get("/signals?limit=-1")
    assert r.status_code == 200


def test_api_zero_limit_clamped(api_client):
    r = api_client.get("/signals?limit=0")
    assert r.status_code == 200


def test_api_huge_limit_clamped(api_client):
    r = api_client.get("/signals?limit=999999")
    assert r.status_code == 200


def test_api_invalid_limit_type(api_client):
    r = api_client.get("/signals?limit=abc")
    assert r.status_code == 422  # FastAPI type validation
