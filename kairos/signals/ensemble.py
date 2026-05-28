from dataclasses import dataclass
import numpy as np
import xgboost as xgb
from kairos.models.signal_event import SignalEvent


@dataclass
class FeatureVector:
    kalman_slope: float
    volume_z_score: float
    anomaly_score: float
    narrative_velocity: float
    narrative_tipping_point: bool
    saturation: float
    regime_accumulation: float
    regime_distribution: float
    regime_transition: float
    causal_bullish: float
    causal_confidence: float
    macro_dff: float

    def to_array(self) -> np.ndarray:
        return np.array([
            self.kalman_slope,
            self.volume_z_score,
            self.anomaly_score,
            self.narrative_velocity,
            float(self.narrative_tipping_point),
            self.saturation,
            self.regime_accumulation,
            self.regime_distribution,
            self.regime_transition,
            self.causal_bullish,
            self.causal_confidence,
            self.macro_dff,
        ], dtype=np.float32)


class SignalEnsemble:
    def __init__(self) -> None:
        self._model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=42,
        )
        self._fitted = False

    def fit(self, feature_vectors: list[FeatureVector]) -> None:
        X = np.vstack([fv.to_array() for fv in feature_vectors])
        y = np.array([1 if fv.causal_bullish > 0.5 else 0 for fv in feature_vectors])
        self._model.fit(X, y)
        self._fitted = True

    def predict(
        self,
        asset: str,
        fv: FeatureVector,
        citations: list[str],
        regime: str = "accumulation",
    ) -> SignalEvent:
        X = fv.to_array().reshape(1, -1)
        proba = self._model.predict_proba(X)[0]
        bullish_prob = float(proba[1])
        direction = "bullish" if bullish_prob > 0.5 else "bearish"
        confidence = bullish_prob if direction == "bullish" else 1.0 - bullish_prob
        estimated_hours = 48.0 / max(fv.narrative_velocity, 0.1)

        return SignalEvent(
            asset=asset,
            direction=direction,
            confidence=round(confidence, 4),
            regime=regime,
            narrative_velocity=round(fv.narrative_velocity, 4),
            narrative_tipping_point=fv.narrative_tipping_point,
            mechanism=f"narrative_momentum({fv.narrative_velocity:.2f}) → regime({regime}) → price",
            estimated_hours=round(min(estimated_hours, 168.0), 1),
            citations=citations,
        )
