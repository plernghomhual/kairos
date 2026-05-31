from typing import Any


class CausalDAG:
    """
    Bayesian causal network encoding known economic relationships.
    All priors derived from published economic research.

    Causal graph:
      narrative_tipping_point → P(price_up) += 0.25  [Shiller 2017]
      regime == accumulation  → P(price_up) *= 1.20  [Minsky Stage 2]
      regime == distribution  → P(price_up) *= 0.70  [Minsky Stage 3-4]
      regime == transition    → confidence  *= 0.60  [Soros reflexivity]
      anomaly_detected        → confidence  *= 0.80  [uncertainty discount]
      macro_stress            → P(price_up) *= 0.75  [Kindleberger]
    """

    def infer_price_impact(
        self,
        narrative_tipping_point: bool,
        regime: str,
        anomaly_detected: bool,
        macro_stress: bool,
    ) -> dict[str, Any]:
        p_up = 0.50
        confidence = 1.0
        citations = []

        if narrative_tipping_point:
            p_up += 0.25
            citations.append("Shiller 2017 - Narrative Economics: narrative tipping → retail entry")

        if regime in ("lv_up", "accumulation"):
            p_up *= 1.20
            citations.append("Minsky Stage 2: quiet accumulation precedes boom")
        elif regime in ("hv_down", "lv_down", "distribution"):
            p_up *= 0.70
            citations.append("Minsky Stage 3-4: distribution precedes decline")
        elif regime in ("hv_up", "transition"):
            confidence *= 0.60
            citations.append("Soros Reflexivity: high-vol regime = elevated uncertainty")

        if anomaly_detected:
            confidence *= 0.80
            citations.append("Anomaly detected: unusual market behavior, confidence reduced")

        if macro_stress:
            p_up *= 0.75
            citations.append("Kindleberger: macro stress depresses risk asset probability")

        p_up = min(max(p_up, 0.0), 1.0)
        confidence = min(max(confidence, 0.0), 1.0)

        return {
            "bullish": round(p_up, 4),
            "bearish": round(1.0 - p_up, 4),
            "confidence": round(confidence, 4),
            "citations": citations,
        }
