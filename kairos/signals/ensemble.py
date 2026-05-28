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


def _estimate_hours(fv: "FeatureVector", regime: str) -> float:
    """Multi-factor time estimate. Accounts for regime urgency, anomalies, vol, and momentum."""
    base = 72.0  # 3-day baseline for a quiet accumulation market
    if regime == "transition":
        base *= 0.50   # market actively changing direction — faster
    elif regime == "distribution":
        base *= 0.75   # selling pressure building — moderately faster
    if fv.anomaly_score > 0:
        base *= 0.60   # unusual price behavior → move likely sooner
    if fv.volume_z_score > 1.5:
        base *= 0.70   # high volume activity → accelerating
    elif fv.volume_z_score < -1.0:
        base *= 1.30   # low volume → slower
    if fv.narrative_velocity > 0.03:
        base *= 0.65   # sentiment shifting fast → catalyst already in motion
    return round(min(max(base, 12.0), 168.0), 1)


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
        if not feature_vectors:
            raise ValueError("feature_vectors must not be empty")
        X = np.vstack([fv.to_array() for fv in feature_vectors])
        y = np.array([1 if fv.causal_bullish > 0.5 else 0 for fv in feature_vectors])
        if len(set(y)) < 2:
            raise ValueError("Training data must contain both bullish (1) and bearish (0) examples")
        self._model.fit(X, y)
        self._fitted = True

    def fit_raw(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train on a pre-built feature matrix with real forward-return labels."""
        if len(set(y.tolist())) < 2:
            raise ValueError("Training data must contain both bullish (1) and bearish (0) examples")
        self._model.fit(X, y)
        self._fitted = True

    def fit_synthetic_fallback(self) -> None:
        """Last-resort fallback when historical data is insufficient."""
        b = FeatureVector(0.02, 1.5, 0.0, 0.5, True, 0.3, 1.0, 0.0, 0.0, 0.75, 0.9, 0.25)
        br = FeatureVector(-0.02, -1.5, 0.1, 0.1, False, 0.5, 0.0, 1.0, 0.0, 0.25, 0.7, 0.5)
        X = np.vstack([fv.to_array() for fv in [b] * 50 + [br] * 50])
        y = np.array([1] * 50 + [0] * 50)
        self._model.fit(X, y)
        self._fitted = True

    def predict(
        self,
        asset: str,
        fv: FeatureVector,
        citations: list[str],
        regime: str = "accumulation",
    ) -> SignalEvent:
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")
        X = fv.to_array().reshape(1, -1)
        proba = self._model.predict_proba(X)[0]
        bullish_prob = float(proba[1])
        direction = "bullish" if bullish_prob > 0.5 else "bearish"
        confidence = bullish_prob if direction == "bullish" else 1.0 - bullish_prob
        estimated_hours = _estimate_hours(fv, regime)

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
