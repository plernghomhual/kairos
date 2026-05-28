from kairos.signals.causal import CausalDAG


def test_narrative_tipping_increases_bullish_prob():
    dag = CausalDAG()
    base = dag.infer_price_impact(
        narrative_tipping_point=False,
        regime="accumulation",
        anomaly_detected=False,
        macro_stress=False,
    )
    elevated = dag.infer_price_impact(
        narrative_tipping_point=True,
        regime="accumulation",
        anomaly_detected=False,
        macro_stress=False,
    )
    assert elevated["bullish"] > base["bullish"]


def test_transition_regime_reduces_confidence():
    dag = CausalDAG()
    result = dag.infer_price_impact(
        narrative_tipping_point=True,
        regime="transition",
        anomaly_detected=True,
        macro_stress=True,
    )
    assert result["confidence"] < 0.6


def test_output_has_required_keys():
    dag = CausalDAG()
    result = dag.infer_price_impact(False, "accumulation", False, False)
    assert "bullish" in result
    assert "bearish" in result
    assert "confidence" in result
    assert "citations" in result
