import json

import numpy as np
import pytest

from kairos.models.signal_event import SignalEvent
from kairos.signals import ensemble as ensemble_mod
from kairos.signals.ensemble import (
    DEFAULT_MODEL_PARAMS_BY_REGIME,
    REGIMES,
    FeatureVector,
    SignalEnsemble,
    tune_sub_models,
)


def make_feature_vector(bullish: bool = True) -> FeatureVector:
    return FeatureVector(
        kalman_slope=0.02 if bullish else -0.02,
        volume_z_score=1.5 if bullish else -1.5,
        anomaly_score=0.1,
        narrative_velocity=0.7 if bullish else 0.1,
        narrative_tipping_point=bullish,
        saturation=0.15,
        regime_lv_up=1.0 if bullish else 0.0,
        regime_hv_up=0.0,
        regime_lv_down=0.0 if bullish else 1.0,
        regime_hv_down=0.0,
        causal_bullish=0.75 if bullish else 0.25,
        causal_confidence=0.85,
        macro_dff=0.25,
    )


def test_feature_vector_validate_accepts_valid_input():
    assert make_feature_vector(True).validate() == []


def test_feature_vector_validate_reports_invalid_values():
    fv = FeatureVector(
        kalman_slope=float("nan"),
        volume_z_score=float("inf"),
        anomaly_score=1.5,
        narrative_velocity=-0.1,
        narrative_tipping_point=1,
        saturation=1.2,
        regime_lv_up=1.0,
        regime_hv_up=1.0,
        regime_lv_down=0.0,
        regime_hv_down=0.0,
        causal_bullish=-0.1,
        causal_confidence=1.2,
        macro_dff=3.0,
    )

    errors = fv.validate()

    assert any("kalman_slope" in err for err in errors)
    assert any("volume_z_score" in err for err in errors)
    assert any("anomaly_score" in err for err in errors)
    assert any("narrative_velocity" in err for err in errors)
    assert any("narrative_tipping_point" in err for err in errors)
    assert any("saturation" in err for err in errors)
    assert any("exactly one regime" in err for err in errors)
    assert any("causal_bullish" in err for err in errors)
    assert any("causal_confidence" in err for err in errors)
    assert any("macro_dff" in err for err in errors)


def test_sanitize_fv_replaces_nan_with_defaults():
    from kairos.signals.ensemble import sanitize_fv

    fv = FeatureVector(
        kalman_slope=float("nan"),
        volume_z_score=float("-inf"),
        anomaly_score=float("nan"),
        narrative_velocity=float("inf"),
        narrative_tipping_point="yes",
        saturation=float("nan"),
        regime_lv_up=float("nan"),
        regime_hv_up=float("inf"),
        regime_lv_down=0.0,
        regime_hv_down=0.0,
        causal_bullish=float("nan"),
        causal_confidence=float("nan"),
        macro_dff=float("inf"),
    )

    cleaned = sanitize_fv(fv)

    assert cleaned.validate() == []
    assert cleaned.kalman_slope == 0.0
    assert cleaned.volume_z_score == 0.0
    assert cleaned.anomaly_score == 0.0
    assert cleaned.narrative_velocity == 0.0
    assert cleaned.narrative_tipping_point is False
    assert cleaned.saturation == 0.5
    assert cleaned.regime_lv_up == 1.0
    assert cleaned.regime_hv_up == 0.0
    assert cleaned.causal_bullish == 0.5
    assert cleaned.causal_confidence == 0.5
    assert cleaned.macro_dff == 0.0


def test_sanitize_fv_clamps_out_of_range_values():
    from kairos.signals.ensemble import sanitize_fv

    fv = FeatureVector(
        kalman_slope=1.0,
        volume_z_score=2.0,
        anomaly_score=4.0,
        narrative_velocity=-2.0,
        narrative_tipping_point=True,
        saturation=3.0,
        regime_lv_up=0.2,
        regime_hv_up=0.7,
        regime_lv_down=0.0,
        regime_hv_down=0.0,
        causal_bullish=2.0,
        causal_confidence=-1.0,
        macro_dff=9.0,
    )

    cleaned = sanitize_fv(fv)

    assert cleaned.validate() == []
    assert cleaned.anomaly_score == 1.0
    assert cleaned.narrative_velocity == 0.0
    assert cleaned.saturation == 1.0
    assert cleaned.regime_lv_up == 1.0
    assert cleaned.regime_hv_up == 0.0
    assert cleaned.causal_bullish == 1.0
    assert cleaned.causal_confidence == 0.0
    assert cleaned.macro_dff == 2.0


def test_predict_rejects_nan_feature_vector_before_xgboost():
    ensemble = SignalEnsemble()
    ensemble.fit_synthetic_fallback()
    fv = make_feature_vector(True)
    fv.kalman_slope = float("nan")

    with pytest.raises(ValueError, match="FeatureVector validation failed"):
        ensemble.predict("BTC", fv, citations=[])


def test_ensemble_generates_signal_event():
    ensemble = SignalEnsemble()
    ensemble.fit_synthetic_fallback()
    fv = make_feature_vector(True)
    event = ensemble.predict("BTC", fv, citations=["Shiller 2017"])
    assert isinstance(event, SignalEvent)
    assert event.asset == "BTC"
    assert event.direction in ("bullish", "bearish")
    assert 0.0 <= event.confidence <= 1.0


def test_bullish_features_predict_bullish():
    ensemble = SignalEnsemble()
    ensemble.fit_synthetic_fallback()
    event = ensemble.predict("BTC", make_feature_vector(True), citations=[])
    assert event.direction == "bullish"


def test_predict_raises_if_not_fitted():
    ensemble = SignalEnsemble()
    with pytest.raises(RuntimeError, match="No sub-models fitted"):
        ensemble.predict("BTC", make_feature_vector(), citations=[])


class DummyRandomizedSearchCV:
    def __init__(
        self,
        estimator,
        param_distributions,
        n_iter,
        scoring,
        cv,
        n_jobs,
        random_state,
        error_score,
    ):
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.error_score = error_score
        self.best_params_ = {}
        self.best_score_ = 0.8123

    def fit(self, X, y, **fit_params):
        self.best_params_ = {name: values[-1] for name, values in self.param_distributions.items()}
        return self


def make_tuning_data(rows_per_regime: int = 36, scarce_regime: str | None = None):
    X_rows = []
    y_rows = []
    regime_rows = []
    for regime_idx, regime in enumerate(REGIMES):
        rows = 12 if regime == scarce_regime else rows_per_regime
        for i in range(rows):
            bullish = i % 2 == 0
            row = np.zeros(13, dtype=np.float32)
            row[0] = 0.02 if bullish else -0.02
            row[1] = float(regime_idx)
            row[6 + regime_idx] = 1.0
            row[10] = 0.75 if bullish else 0.25
            row[11] = 0.8
            row[12] = 0.25
            X_rows.append(row)
            y_rows.append(1 if bullish else 0)
            regime_rows.append(regime)
    return np.vstack(X_rows), np.array(y_rows), np.array(regime_rows)


def test_tune_sub_models_returns_dict(tmp_path, monkeypatch):
    monkeypatch.setattr(ensemble_mod, "RandomizedSearchCV", DummyRandomizedSearchCV)
    monkeypatch.setattr(ensemble_mod, "_PARAM_CACHE_DIR", tmp_path)
    X, y, regimes = make_tuning_data()

    best_params = tune_sub_models(X, y, regimes, n_iter=2)

    assert set(best_params) == set(REGIMES)
    assert len({tuple(sorted(params.items())) for params in best_params.values()}) > 1
    base_defaults = {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.1}
    for regime, params in best_params.items():
        # Effective defaults for this regime: global defaults plus any per-regime overrides.
        effective_defaults = {
            **base_defaults,
            **DEFAULT_MODEL_PARAMS_BY_REGIME.get(regime, {}),
        }
        changed_defaults = [
            name
            for name, default in effective_defaults.items()
            if name in base_defaults and params.get(name) != default
        ]
        assert len(changed_defaults) >= 3


def test_tune_skipped_regime(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(ensemble_mod, "RandomizedSearchCV", DummyRandomizedSearchCV)
    monkeypatch.setattr(ensemble_mod, "_PARAM_CACHE_DIR", tmp_path)
    X, y, regimes = make_tuning_data(scarce_regime="hv_down")

    best_params = tune_sub_models(X, y, regimes, n_iter=2)

    assert "hv_down" not in best_params
    assert set(best_params) == {"lv_up", "hv_up", "lv_down"}
    assert "Skipping hv_down tuning" in caplog.text


def test_best_params_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(ensemble_mod, "RandomizedSearchCV", DummyRandomizedSearchCV)
    monkeypatch.setattr(ensemble_mod, "_PARAM_CACHE_DIR", tmp_path)
    X, y, regimes = make_tuning_data()

    tune_sub_models(X, y, regimes, n_iter=2)

    path = tmp_path / "best_params_lv_up.json"
    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["regime"] == "lv_up"
    assert payload["validation_auc"] == pytest.approx(0.8123)
    assert payload["best_params"]["n_estimators"] == 200


def test_custom_params_used():
    ensemble = SignalEnsemble(
        params={
            "hv_down": {
                "n_estimators": 50,
                "max_depth": 2,
                "learning_rate": 0.03,
                "subsample": 0.6,
            }
        }
    )

    ensemble.fit_synthetic_fallback()

    hv_down_params = ensemble._models["hv_down"].get_params()
    lv_up_params = ensemble._models["lv_up"].get_params()
    assert hv_down_params["n_estimators"] == 50
    assert hv_down_params["max_depth"] == 2
    assert hv_down_params["learning_rate"] == 0.03
    assert hv_down_params["subsample"] == 0.6
    assert lv_up_params["n_estimators"] == 100


def test_load_best_params_applies_cached_params(tmp_path, monkeypatch):
    monkeypatch.setattr(ensemble_mod, "_PARAM_CACHE_DIR", tmp_path)
    (tmp_path / "best_params_hv_up.json").write_text(
        json.dumps(
            {
                "regime": "hv_up",
                "validation_auc": 0.91,
                "best_params": {
                    "n_estimators": 150,
                    "max_depth": 3,
                    "learning_rate": 0.08,
                },
            }
        )
    )

    ensemble = SignalEnsemble()
    loaded = ensemble.load_best_params("BTC")
    ensemble.fit_synthetic_fallback()

    hv_up_params = ensemble._models["hv_up"].get_params()
    lv_up_params = ensemble._models["lv_up"].get_params()
    assert loaded is True
    assert hv_up_params["n_estimators"] == 150
    assert hv_up_params["max_depth"] == 3
    assert hv_up_params["learning_rate"] == 0.08
    assert ensemble.validation_auc_by_regime["hv_up"] == pytest.approx(0.91)
    assert lv_up_params["n_estimators"] == 100


def test_feature_vector_zero_signal_values_predicts(trained_ensemble):
    fv = FeatureVector(
        kalman_slope=0.0,
        volume_z_score=0.0,
        anomaly_score=0.0,
        narrative_velocity=0.0,
        narrative_tipping_point=False,
        saturation=0.0,
        regime_lv_up=1.0,
        regime_hv_up=0.0,
        regime_lv_down=0.0,
        regime_hv_down=0.0,
        causal_bullish=0.5,
        causal_confidence=0.5,
        macro_dff=0.0,
    )
    event = trained_ensemble.predict("BTC", fv, citations=[], regime="lv_up")
    assert event.direction in ("bullish", "bearish")
    assert 0.0 <= event.confidence <= 1.0


def test_feature_vector_extreme_valid_values_predicts(trained_ensemble):
    fv = FeatureVector(
        kalman_slope=1_000_000.0,
        volume_z_score=-1_000_000.0,
        anomaly_score=1.0,
        narrative_velocity=1.0,
        narrative_tipping_point=True,
        saturation=1.0,
        regime_lv_up=0.0,
        regime_hv_up=1.0,
        regime_lv_down=0.0,
        regime_hv_down=0.0,
        causal_bullish=1.0,
        causal_confidence=1.0,
        macro_dff=2.0,
    )
    event = trained_ensemble.predict("BTC", fv, citations=[], regime="hv_up")
    assert event.direction in ("bullish", "bearish")
    assert 0.0 <= event.confidence <= 1.0
    assert 12.0 <= event.estimated_hours <= 168.0


def test_fit_raw_zero_rows_raises_value_error():
    with pytest.raises(ValueError, match="both bullish"):
        SignalEnsemble().fit_raw(np.empty((0, 13)), np.array([]), np.array([]))


def test_fit_raw_one_row_raises_value_error():
    with pytest.raises(ValueError, match="both bullish"):
        SignalEnsemble().fit_raw(
            np.zeros((1, 13)),
            np.array([1]),
            np.array(["lv_up"]),
        )


def test_predict_before_fit_runtime_error(sample_feature_vector):
    with pytest.raises(RuntimeError, match="No sub-models fitted"):
        SignalEnsemble().predict("BTC", sample_feature_vector, citations=[])


def test_synthetic_fallback_fits_all_four_sub_models():
    ensemble = SignalEnsemble()
    ensemble.fit_synthetic_fallback()
    assert set(ensemble.fitted_regimes()) == set(REGIMES)
    assert set(ensemble._models) == set(REGIMES)
