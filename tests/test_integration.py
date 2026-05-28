import numpy as np
import pytest
from kairos.signals.kalman import kalman_smooth
from kairos.signals.anomaly import detect_anomalies
from kairos.signals.narrative import compute_narrative_features
from kairos.signals.regime import fit_regime_model, predict_regime
from kairos.signals.causal import CausalDAG
from kairos.signals.ensemble import SignalEnsemble, FeatureVector
from kairos.models.signal_event import SignalEvent


def test_full_pipeline_produces_signal_event():
    np.random.seed(42)

    # Layer 1: Reality
    raw_prices = np.random.normal(50000, 500, 100)
    smoothed = kalman_smooth(raw_prices)
    kalman_slope = float(np.polyfit(range(len(smoothed[-10:])), smoothed[-10:], 1)[0])
    features_2d = np.column_stack([raw_prices, np.random.normal(1e9, 1e8, 100)])
    anomaly_flags = detect_anomalies(features_2d)
    anomaly_score = float(anomaly_flags[-1])

    # Layer 2: Narrative
    post_counts = [10, 15, 25, 40, 65, 100, 160]
    narrative = compute_narrative_features(post_counts)

    # Layer 3: Regime
    regime_features = np.column_stack([
        np.random.normal(0.001, 0.005, 100),
        np.random.normal(0.005, 0.001, 100),
    ])
    model = fit_regime_model(regime_features)
    regime = predict_regime(model, regime_features[-5:])

    # Causal DAG
    dag = CausalDAG()
    causal = dag.infer_price_impact(
        narrative_tipping_point=narrative["narrative_tipping_point"],
        regime=regime,
        anomaly_detected=bool(anomaly_score),
        macro_stress=False,
    )

    # Ensemble
    fv = FeatureVector(
        kalman_slope=kalman_slope,
        volume_z_score=1.2,
        anomaly_score=anomaly_score,
        narrative_velocity=narrative["narrative_velocity"],
        narrative_tipping_point=narrative["narrative_tipping_point"],
        saturation=narrative["saturation"],
        regime_accumulation=1.0 if regime == "accumulation" else 0.0,
        regime_distribution=1.0 if regime == "distribution" else 0.0,
        regime_transition=1.0 if regime == "transition" else 0.0,
        causal_bullish=causal["bullish"],
        causal_confidence=causal["confidence"],
        macro_dff=0.25,
    )

    ensemble = SignalEnsemble()
    training_fvs = [
        FeatureVector(0.01, 1.0, 0.0, 2.0, True, 0.1, 1.0, 0.0, 0.0, 0.7, 0.9, 0.25)
    ] * 40 + [
        FeatureVector(-0.01, -1.0, 0.1, 0.1, False, 0.5, 0.0, 1.0, 0.0, 0.3, 0.7, 0.5)
    ] * 40
    ensemble.fit(training_fvs)

    event = ensemble.predict("BTC", fv, citations=causal["citations"], regime=regime)

    assert isinstance(event, SignalEvent)
    assert event.asset == "BTC"
    assert event.direction in ("bullish", "bearish")
    assert 0.0 <= event.confidence <= 1.0
    assert len(event.citations) > 0
    assert event.estimated_hours > 0
