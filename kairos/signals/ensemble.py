import json
import logging
import os
import time
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from kairos.models.signal_event import SignalEvent

REGIMES = ["lv_up", "hv_up", "lv_down", "hv_down"]

FEATURE_DIM: int = 0
_NUMERIC_TYPES = (int, float, np.integer, np.floating)
_PARAM_CACHE_DIR = Path(os.getenv("KAIROS_CACHE_DIR", str(Path.home() / ".kairos")))
DEFAULT_MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 4,
    "learning_rate": 0.1,
    "eval_metric": "logloss",
    "random_state": 42,
}
# Per-regime overrides applied on top of DEFAULT_MODEL_PARAMS when tuning fails or
# training data is insufficient.  Only regimes that need non-default values are listed.
DEFAULT_MODEL_PARAMS_BY_REGIME: dict[str, dict] = {
    "lv_down": {
        "max_depth": 3,
        "reg_lambda": 5.0,
    },
}
PARAM_GRID = {
    "lv_up": {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.05, 0.1, 0.2],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "min_child_weight": [1, 3, 5],
        "gamma": [0, 0.1, 0.2],
        "reg_alpha": [0, 0.1, 1.0],
        "reg_lambda": [1.0, 2.0, 3.0],
    },
    "hv_up": {
        "n_estimators": [50, 100, 150],
        "max_depth": [2, 3],
        "learning_rate": [0.03, 0.05, 0.08],
        "subsample": [0.6, 0.8],
        "colsample_bytree": [0.6, 0.8],
        "min_child_weight": [3, 5, 7],
        "gamma": [0.1, 0.2, 0.4],
        "reg_alpha": [0.1, 0.5, 1.0],
        "reg_lambda": [2.0, 3.0, 5.0],
    },
    "lv_down": {
        "n_estimators": [75, 150, 250],
        "max_depth": [2, 3, 4],
        "learning_rate": [0.04, 0.08, 0.15],
        "subsample": [0.7, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0],
        "min_child_weight": [3, 5, 7],
        "gamma": [0, 0.05, 0.15],
        "reg_alpha": [0, 0.05, 0.5],
        "reg_lambda": [3.0, 5.0, 10.0],
    },
    "hv_down": {
        "n_estimators": [50, 100, 150],
        "max_depth": [2, 3],
        "learning_rate": [0.02, 0.05, 0.08],
        "subsample": [0.6, 0.75],
        "colsample_bytree": [0.6, 0.75, 0.9],
        "min_child_weight": [5, 7, 10],
        "gamma": [0.2, 0.4, 0.8],
        "reg_alpha": [0.5, 1.0, 2.0],
        "reg_lambda": [3.0, 5.0, 8.0],
    },
}
logger = logging.getLogger(__name__)


def _get_feature_dim() -> int:
    global FEATURE_DIM
    if FEATURE_DIM == 0:
        FEATURE_DIM = len(fields(FeatureVector))
    return FEATURE_DIM


def _is_finite_float(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, _NUMERIC_TYPES):
        return False
    return bool(np.isfinite(float(value)))


def _safe_float(value: object, default: float) -> float:
    if isinstance(value, (bool, np.bool_)):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if np.isfinite(result) else default


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


@dataclass
class FeatureVector:
    kalman_slope: float
    volume_z_score: float
    anomaly_score: float
    narrative_velocity: float
    narrative_tipping_point: bool
    saturation: float
    regime_lv_up: float
    regime_hv_up: float
    regime_lv_down: float
    regime_hv_down: float
    causal_bullish: float
    causal_confidence: float
    macro_dff: float

    def to_array(self) -> np.ndarray:
        return np.array(
            [
                self.kalman_slope,
                self.volume_z_score,
                self.anomaly_score,
                self.narrative_velocity,
                float(self.narrative_tipping_point),
                self.saturation,
                self.regime_lv_up,
                self.regime_hv_up,
                self.regime_lv_down,
                self.regime_hv_down,
                self.causal_bullish,
                self.causal_confidence,
                self.macro_dff,
            ],
            dtype=np.float32,
        )

    def validate(self) -> list[str]:
        errors: list[str] = []

        for name in ("kalman_slope", "volume_z_score"):
            if not _is_finite_float(getattr(self, name)):
                errors.append(f"{name} must be a finite float")

        for name in (
            "anomaly_score",
            "narrative_velocity",
            "saturation",
            "causal_bullish",
            "causal_confidence",
        ):
            value = getattr(self, name)
            if not _is_finite_float(value):
                errors.append(f"{name} must be a finite float in [0.0, 1.0]")
            elif not 0.0 <= float(value) <= 1.0:
                errors.append(f"{name} must be in [0.0, 1.0]")

        if not isinstance(self.narrative_tipping_point, bool):
            errors.append("narrative_tipping_point must be bool")

        regime_values = [
            self.regime_lv_up,
            self.regime_hv_up,
            self.regime_lv_down,
            self.regime_hv_down,
        ]
        for name, value in zip(
            (
                "regime_lv_up",
                "regime_hv_up",
                "regime_lv_down",
                "regime_hv_down",
            ),
            regime_values,
        ):
            if not _is_finite_float(value):
                errors.append(f"{name} must be 0.0 or 1.0")
            elif float(value) not in (0.0, 1.0):
                errors.append(f"{name} must be 0.0 or 1.0")
        if all(_is_finite_float(value) for value in regime_values) and sum(float(v) for v in regime_values) != 1.0:
            errors.append("exactly one regime_* field must be 1.0")

        if not _is_finite_float(self.macro_dff):
            errors.append("macro_dff must be a finite float in [-2.0, 2.0]")
        elif not -2.0 <= float(self.macro_dff) <= 2.0:
            errors.append("macro_dff must be in [-2.0, 2.0]")

        return errors


def sanitize_fv(fv: FeatureVector) -> FeatureVector:
    """Replace NaN/Inf with safe defaults, clamp ranges. Returns new FeatureVector."""
    regime_values = [
        _safe_float(fv.regime_lv_up, 0.0),
        _safe_float(fv.regime_hv_up, 0.0),
        _safe_float(fv.regime_lv_down, 0.0),
        _safe_float(fv.regime_hv_down, 0.0),
    ]
    if regime_values.count(1.0) == 1 and all(v in (0.0, 1.0) for v in regime_values):
        regime_lv_up, regime_hv_up, regime_lv_down, regime_hv_down = regime_values
    else:
        regime_lv_up, regime_hv_up, regime_lv_down, regime_hv_down = 1.0, 0.0, 0.0, 0.0

    return FeatureVector(
        kalman_slope=_safe_float(fv.kalman_slope, 0.0),
        volume_z_score=_safe_float(fv.volume_z_score, 0.0),
        anomaly_score=_clamp(_safe_float(fv.anomaly_score, 0.0), 0.0, 1.0),
        narrative_velocity=_clamp(_safe_float(fv.narrative_velocity, 0.0), 0.0, 1.0),
        narrative_tipping_point=fv.narrative_tipping_point if isinstance(fv.narrative_tipping_point, bool) else False,
        saturation=_clamp(_safe_float(fv.saturation, 0.5), 0.0, 1.0),
        regime_lv_up=regime_lv_up,
        regime_hv_up=regime_hv_up,
        regime_lv_down=regime_lv_down,
        regime_hv_down=regime_hv_down,
        causal_bullish=_clamp(_safe_float(fv.causal_bullish, 0.5), 0.0, 1.0),
        causal_confidence=_clamp(_safe_float(fv.causal_confidence, 0.5), 0.0, 1.0),
        macro_dff=_clamp(_safe_float(fv.macro_dff, 0.0), -2.0, 2.0),
    )


def _estimate_hours(fv: "FeatureVector", regime: str) -> float:
    fv = sanitize_fv(fv)
    base = 72.0
    if regime == "hv_up":
        base *= 0.50
    elif regime == "lv_down":
        base *= 0.75
    elif regime == "hv_down":
        base *= 0.40
    if fv.anomaly_score > 0:
        base *= 0.60
    if fv.volume_z_score > 1.5:
        base *= 0.70
    elif fv.volume_z_score < -1.0:
        base *= 1.30
    if fv.narrative_velocity > 0.03:
        base *= 0.65
    return round(min(max(base, 12.0), 168.0), 1)


def _json_safe_params(params: dict) -> dict:
    safe: dict = {}
    for name, value in params.items():
        if isinstance(value, np.integer):
            safe[name] = int(value)
        elif isinstance(value, np.floating):
            safe[name] = float(value)
        else:
            safe[name] = value
    return safe


def _merged_model_params(params: dict | None = None, regime: str | None = None) -> dict:
    merged = dict(DEFAULT_MODEL_PARAMS)
    if regime and regime in DEFAULT_MODEL_PARAMS_BY_REGIME:
        merged.update(DEFAULT_MODEL_PARAMS_BY_REGIME[regime])
    if params:
        merged.update(params)
    return _json_safe_params(merged)


def _default_model(params: dict | None = None, regime: str | None = None) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(**_merged_model_params(params, regime=regime))


def _best_params_path(regime: str) -> Path:
    _PARAM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _PARAM_CACHE_DIR / f"best_params_{regime}.json"


def _load_best_params_payload(regime: str) -> dict | None:
    path = _best_params_path(regime)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("best_params"), dict):
        return None
    return payload


def _save_best_params(regime: str, params: dict, validation_auc: float) -> None:
    payload = {
        "regime": regime,
        "best_params": _json_safe_params(params),
        "validation_auc": float(validation_auc),
        "created_at": time.time(),
    }
    with open(_best_params_path(regime), "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _param_space_size(param_grid: dict[str, list]) -> int:
    size = 1
    for values in param_grid.values():
        size *= len(values)
    return size


def tune_sub_models(
    X: np.ndarray,
    y: np.ndarray,
    regime_labels: np.ndarray,
    cv_folds: int = 3,
    n_iter: int = 50,
) -> dict[str, dict]:
    """Run randomized search per regime and return {regime: best_params}."""
    best_by_regime: dict[str, dict] = {}
    for reg in REGIMES:
        mask = regime_labels == reg
        row_count = int(mask.sum())
        if row_count < 30:
            logger.warning("Skipping %s tuning: only %s rows (<30)", reg, row_count)
            continue
        X_reg = X[mask]
        y_reg = y[mask]
        classes, class_counts = np.unique(y_reg, return_counts=True)
        if len(classes) < 2:
            logger.warning("Skipping %s tuning: only one target class present", reg)
            continue
        if int(class_counts.min()) < 2:
            logger.warning("Skipping %s tuning: minority class has fewer than 2 rows", reg)
            continue

        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X_reg,
                y_reg,
                test_size=0.2,
                random_state=42,
                stratify=y_reg,
            )
            _, train_class_counts = np.unique(y_train, return_counts=True)
            effective_folds = min(cv_folds, int(train_class_counts.min()))
            if effective_folds < 2:
                logger.warning("Skipping %s tuning: not enough class balance for CV", reg)
                continue

            search = RandomizedSearchCV(
                estimator=_default_model({"early_stopping_rounds": 10}, regime=reg),
                param_distributions=PARAM_GRID[reg],
                n_iter=min(n_iter, 50, _param_space_size(PARAM_GRID[reg])),
                scoring="roc_auc",
                cv=effective_folds,
                n_jobs=-1,
                random_state=42,
                error_score="raise",
            )
            search.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            best_params = _json_safe_params(search.best_params_)
            best_by_regime[reg] = best_params
            _save_best_params(reg, best_params, float(search.best_score_))
        except Exception as exc:
            logger.warning("Tuning failed for %s; using default XGBoost params: %s", reg, exc)
            best_by_regime[reg] = _merged_model_params(regime=reg)
    return best_by_regime


class SignalEnsemble:
    def __init__(self, params: dict[str, dict] | None = None) -> None:
        self._models: dict[str, xgb.XGBClassifier] = {}
        self._fitted: dict[str, bool] = {}
        self._trained_at: float | None = None
        self._candle_count: int = 0
        self._params: dict[str, dict] = {
            reg: _json_safe_params(reg_params) for reg, reg_params in (params or {}).items() if reg in REGIMES
        }
        self.trained_with_params: dict[str, dict] = {}
        self.validation_auc_by_regime: dict[str, float] = {}
        self.is_synthetic: bool = False

    def load_best_params(self, asset: str) -> bool:
        del asset  # Best-param files are currently shared by regime.
        loaded = False
        for reg in REGIMES:
            payload = _load_best_params_payload(reg)
            if payload is None:
                continue
            self._params[reg] = _json_safe_params(payload["best_params"])
            validation_auc = payload.get("validation_auc")
            if validation_auc is not None:
                self.validation_auc_by_regime[reg] = float(validation_auc)
            loaded = True
        return loaded

    def _params_for_regime(self, regime: str) -> dict:
        return self._params.get(regime, {})

    def fit_raw(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray | None = None,
        candle_count: int = 0,
    ) -> None:
        if len(set(y.tolist())) < 2:
            raise ValueError("Training data must contain both bullish (1) and bearish (0) examples")
        if regime_labels is None:
            regime_labels = np.array(["lv_up"] * len(y))
        for reg in REGIMES:
            mask = regime_labels == reg
            if mask.sum() < 10:
                continue
            X_reg = X[mask]
            y_reg = y[mask]
            if len(set(y_reg.tolist())) < 2:
                continue
            params = self._params_for_regime(reg)
            model = _default_model(params, regime=reg)
            model.fit(X_reg, y_reg)
            self._models[reg] = model
            self._fitted[reg] = True
            self.trained_with_params[reg] = _merged_model_params(params, regime=reg)
        if not any(self._fitted.values()):
            raise ValueError(
                "No regime sub-model received enough training data (need ≥10 rows per regime with both classes)"
            )
        self._trained_at = time.time()
        self._candle_count = candle_count
        self.is_synthetic = False

    def fit_synthetic_fallback(self) -> None:
        for reg in REGIMES:
            b = FeatureVector(
                0.02,
                1.5,
                0.0,
                0.5,
                True,
                0.3,
                1.0 if reg == "lv_up" else 0.0,
                1.0 if reg == "hv_up" else 0.0,
                1.0 if reg == "lv_down" else 0.0,
                1.0 if reg == "hv_down" else 0.0,
                0.75,
                0.9,
                0.25,
            )
            br = FeatureVector(
                -0.02,
                -1.5,
                0.1,
                0.1,
                False,
                0.5,
                1.0 if reg == "lv_up" else 0.0,
                1.0 if reg == "hv_up" else 0.0,
                1.0 if reg == "lv_down" else 0.0,
                1.0 if reg == "hv_down" else 0.0,
                0.25,
                0.7,
                0.5,
            )
            X = np.vstack([fv.to_array() for fv in [b] * 50 + [br] * 50])
            y = np.array([1] * 50 + [0] * 50)
            params = self._params_for_regime(reg)
            model = _default_model(params, regime=reg)
            model.fit(X, y)
            self._models[reg] = model
            self._fitted[reg] = True
            self.trained_with_params[reg] = _merged_model_params(params, regime=reg)
        self._trained_at = time.time()
        self._candle_count = 0
        self.is_synthetic = True

    def fit(
        self,
        fv_list: list,
        prices: list[float] | None = None,
        lookahead: int = 2,
    ) -> None:
        """Train from a list of FeatureVectors.

        Parameters
        ----------
        fv_list:
            Feature vectors for training.
        prices:
            Historical prices aligned with fv_list. When provided, labels are
            derived from forward returns: ``prices[i + lookahead] > prices[i]``.
            Must have at least ``len(fv_list) + lookahead`` elements.
            **Required** to avoid circular label leakage — without it the method
            falls back to ``kalman_slope > 0`` which is a predictor feature and
            produces inflated in-sample performance.
        lookahead:
            Number of bars ahead to use for forward-return labels (default 2).
        """
        if not fv_list:
            raise ValueError("Cannot fit on empty list of feature vectors")
        n = len(fv_list)
        if prices is not None:
            if len(prices) < n + lookahead:
                raise ValueError(
                    f"prices must have at least {n + lookahead} elements to compute "
                    f"{lookahead}-bar forward returns for {n} feature vectors "
                    f"(got {len(prices)})"
                )
            y = np.array([1 if prices[i + lookahead] > prices[i] else 0 for i in range(n)])
        else:
            raise ValueError(
                "ensemble.fit() requires prices= to avoid circular label leakage. "
                "Pass a list of prices with at least len(fv_list) + lookahead elements."
            )
        if len(set(y.tolist())) < 2:
            raise ValueError("Need both bullish and bearish examples")
        X = np.vstack([fv.to_array() for fv in fv_list])
        regime_labels = np.array(["lv_up"] * n)
        for i, fv in enumerate(fv_list):
            if getattr(fv, "regime_hv_up", 0.0) > 0.5:
                regime_labels[i] = "hv_up"
            elif getattr(fv, "regime_lv_down", 0.0) > 0.5:
                regime_labels[i] = "lv_down"
            elif getattr(fv, "regime_hv_down", 0.0) > 0.5:
                regime_labels[i] = "hv_down"
        try:
            self.fit_raw(X, y, regime_labels)
        except ValueError:
            self.fit_synthetic_fallback()

    def save(self, path: str) -> None:
        """Save each regime sub-model in XGBoost native format + JSON metadata."""
        import json as _json
        from pathlib import Path as _Path

        if not any(self._fitted.values()):
            raise RuntimeError("Cannot save unfitted ensemble")
        base = _Path(path)
        regimes = list(self._models.keys())
        for reg, model in self._models.items():
            model.save_model(str(base.parent / f"{base.stem}_{reg}.ubj"))
        with open(str(base) + ".meta.json", "w") as f:
            _json.dump({"regimes": regimes, "is_synthetic": self.is_synthetic}, f)

    @classmethod
    def load(cls, path: str) -> "SignalEnsemble":
        """Load from XGBoost native format + JSON metadata."""
        import json as _json
        from pathlib import Path as _Path

        import xgboost as xgb

        base = _Path(path)
        meta_path = _Path(str(base) + ".meta.json")
        if not meta_path.exists():
            raise FileNotFoundError(f"No model metadata at {meta_path}")
        with open(meta_path) as f:
            meta = _json.load(f)
        ensemble = cls()
        ensemble.is_synthetic = bool(meta.get("is_synthetic", False))
        for reg in meta.get("regimes", []):
            reg_path = base.parent / f"{base.stem}_{reg}.ubj"
            if not reg_path.exists():
                raise FileNotFoundError(f"Missing sub-model: {reg_path}")
            model = xgb.XGBClassifier()
            model.load_model(str(reg_path))
            ensemble._models[reg] = model
            ensemble._fitted[reg] = True
        return ensemble

    def is_stale(self, current_candle_count: int, max_growth: int = 50) -> bool:
        return current_candle_count - self._candle_count > max_growth

    def fitted_regimes(self) -> list[str]:
        return [r for r in REGIMES if self._fitted.get(r, False)]

    def predict(self, asset: str, fv: FeatureVector, citations: list[str], regime: str = "lv_up") -> SignalEvent:
        errors = fv.validate()
        if errors:
            raise ValueError("FeatureVector validation failed: " + "; ".join(errors))
        model = self._models.get(regime) if self._fitted.get(regime, False) else None
        actual_regime = regime
        if model is None:
            fitted = self.fitted_regimes()
            if not fitted:
                raise RuntimeError("No sub-models fitted — call fit_raw() or fit_synthetic_fallback() first")
            actual_regime = fitted[0]
            logger.warning(
                "Ensemble falling back from requested regime %s to fitted regime %s",
                regime,
                actual_regime,
            )
            model = self._models[actual_regime]
        X = fv.to_array().reshape(1, -1)
        proba = model.predict_proba(X)[0]
        bullish_prob = float(proba[1])
        direction = "bullish" if bullish_prob > 0.5 else "bearish"
        confidence = bullish_prob if direction == "bullish" else 1.0 - bullish_prob

        if actual_regime == "lv_down":
            momentum_strength = min(abs(fv.kalman_slope) * 0.005, 0.5)
            if fv.kalman_slope < 0:
                if direction == "bearish":
                    confidence = min(confidence + momentum_strength, 0.95)
                else:
                    confidence = max(confidence - momentum_strength * 1.5, 0.51)
            elif fv.kalman_slope > 0 and direction == "bullish":
                confidence = min(confidence + momentum_strength * 0.5, 0.95)

        estimated_hours = _estimate_hours(fv, actual_regime)
        return SignalEvent(
            asset=asset,
            direction=direction,
            confidence=round(confidence, 4),
            regime=actual_regime,
            narrative_velocity=round(fv.narrative_velocity, 4),
            narrative_tipping_point=fv.narrative_tipping_point,
            mechanism=f"regime_routed({actual_regime}) → narrative({fv.narrative_velocity:.2f}) → price",
            estimated_hours=round(min(estimated_hours, 168.0), 1),
            citations=citations,
        )
