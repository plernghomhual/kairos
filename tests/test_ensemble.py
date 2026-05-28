import numpy as np
import pytest
from kairos.signals.ensemble import SignalEnsemble, FeatureVector
from kairos.models.signal_event import SignalEvent

def make_feature_vector(bullish: bool = True) -> FeatureVector:
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

def test_ensemble_generates_signal_event():
    ensemble = SignalEnsemble()
    ensemble.fit([make_feature_vector(True)] * 30 + [make_feature_vector(False)] * 30)
    fv = make_feature_vector(True)
    event = ensemble.predict("BTC", fv, citations=["Shiller 2017"])
    assert isinstance(event, SignalEvent)
    assert event.asset == "BTC"
    assert event.direction in ("bullish", "bearish")
    assert 0.0 <= event.confidence <= 1.0

def test_bullish_features_predict_bullish():
    ensemble = SignalEnsemble()
    ensemble.fit([make_feature_vector(True)] * 50 + [make_feature_vector(False)] * 50)
    event = ensemble.predict("BTC", make_feature_vector(True), citations=[])
    assert event.direction == "bullish"
