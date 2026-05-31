import numpy as np
import pytest

from kairos.signals import regime as regime_mod
from kairos.signals.regime import (
    fit_regime_model,
    optimize_regime_model,
    predict_regime,
    predict_regime_with_confidence,
    smooth_regime,
)


def make_regime_data():
    np.random.seed(0)
    lv_up = np.column_stack(
        [
            np.random.normal(0.001, 0.005, 60),
            np.random.normal(0.005, 0.001, 60),
        ]
    )
    lv_down = np.column_stack(
        [
            np.random.normal(-0.001, 0.005, 60),
            np.random.normal(0.005, 0.001, 60),
        ]
    )
    hv_up = np.column_stack(
        [
            np.random.normal(0, 0.02, 30),
            np.random.normal(0.02, 0.005, 30),
        ]
    )
    hv_down = np.column_stack(
        [
            np.random.normal(0, 0.025, 30),
            np.random.normal(0.025, 0.005, 30),
        ]
    )
    return np.vstack([lv_up, lv_down, hv_up, hv_down])


def test_fit_and_predict_regime(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    data = make_regime_data()
    model = optimize_regime_model(data)
    regime = predict_regime(model, data[-1:])
    assert regime in ("lv_up", "hv_up", "lv_down", "hv_down")


def test_regime_is_consistent(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    data = make_regime_data()
    model = optimize_regime_model(data)
    r1 = predict_regime(model, data[-1:])
    r2 = predict_regime(model, data[-1:])
    assert r1 == r2


def test_regime_confidence_range(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    data = make_regime_data()
    model = optimize_regime_model(data)
    regime, confidence = predict_regime_with_confidence(model, data[-5:])
    assert regime in ("lv_up", "hv_up", "lv_down", "hv_down")
    assert 0.0 <= confidence <= 1.0


def test_regime_smoothing():
    history = ["lv_up", "lv_up", "lv_up"]
    assert smooth_regime(history, "hv_down", alpha=0.5) == "lv_up"
    assert smooth_regime(history, "hv_down", alpha=0.7) == "hv_down"


def test_regime_optimization_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    data = make_regime_data()
    model = optimize_regime_model(data)
    cache_file = tmp_path / ".kairos" / "hmm_params.json"
    assert cache_file.exists()
    assert model.n_components in (3, 4, 5)


def test_predict_regime_single_data_point_returns_valid_default(fitted_hmm):
    regime = predict_regime(fitted_hmm, np.array([[0.0, 0.0]]))
    assert regime in ("lv_up", "hv_up", "lv_down", "hv_down")


def test_fit_regime_model_duplicate_rows_does_not_crash():
    rows = np.tile(
        np.array([[0.001, 0.005], [-0.001, 0.005], [0.0, 0.02], [-0.003, 0.02]]),
        (20, 1),
    )
    model = fit_regime_model(rows)
    assert predict_regime(model, rows[-1:]) in ("lv_up", "hv_up", "lv_down", "hv_down")


@pytest.mark.parametrize(("negative_return", "volatility"), [(-0.003, 0.004), (-0.02, 0.04)])
def test_negative_returns_only_identifies_down_regime(negative_return, volatility):
    features = np.column_stack(
        [
            np.full(80, negative_return),
            np.full(80, volatility),
        ]
    )
    model = fit_regime_model(features)
    assert predict_regime(model, features[-1:]) in ("lv_down", "hv_down")


def test_very_large_regime_values_do_not_overflow():
    features = np.column_stack(
        [
            np.linspace(-10.0, 10.0, 80),
            np.linspace(1.0, 20.0, 80),
        ]
    )
    model = fit_regime_model(features)
    regime, confidence = predict_regime_with_confidence(model, features[-3:])
    assert regime in ("lv_up", "hv_up", "lv_down", "hv_down")
    assert 0.0 <= confidence <= 1.0


def test_empty_regime_array_raises_value_error():
    with pytest.raises(ValueError, match="2D matrix"):
        fit_regime_model(np.array([]))


def test_new_hmm_model_uses_tighter_tolerance():
    model = regime_mod._new_model(4, "diag", 100, 42)
    assert model.tol == pytest.approx(0.01)


def test_transition_penalty_penalizes_sticky_matrices():
    sticky = np.array(
        [
            [0.98, 0.01, 0.01],
            [0.01, 0.98, 0.01],
            [0.01, 0.01, 0.98],
        ]
    )
    transitional = np.array(
        [
            [0.34, 0.33, 0.33],
            [0.33, 0.34, 0.33],
            [0.33, 0.33, 0.34],
        ]
    )
    assert regime_mod._transition_matrix_penalty(sticky) > regime_mod._transition_matrix_penalty(transitional)


def test_select_best_params_rejects_dominated_label_candidates(monkeypatch):
    features = np.ones((12, 2), dtype=float)
    dominated = {
        "n_components": 4,
        "covariance_type": "diag",
        "n_iter": 100,
        "random_state": 42,
        "aic": 1.0,
        "bic": 1.0,
        "converged": True,
        "label_sequence": ["lv_down"] * 9 + ["lv_up"] * 3,
        "transmat_penalty": 0.01,
    }
    diverse = {
        "n_components": 4,
        "covariance_type": "full",
        "n_iter": 100,
        "random_state": 0,
        "aic": 100.0,
        "bic": 100.0,
        "converged": True,
        "label_sequence": ["lv_down", "lv_up", "hv_down", "hv_up"] * 3,
        "transmat_penalty": 0.5,
    }

    def fake_fit_candidate_chain(features, *, n_components, covariance_type, random_state):
        return [dominated if random_state == 42 else diverse]

    monkeypatch.setattr(regime_mod, "N_COMPONENT_OPTIONS", (4,))
    monkeypatch.setattr(regime_mod, "COVARIANCE_OPTIONS", ("diag",))
    monkeypatch.setattr(regime_mod, "N_ITER_OPTIONS", (100,))
    monkeypatch.setattr(regime_mod, "RANDOM_STATE_OPTIONS", (42, 0))
    monkeypatch.setattr(regime_mod, "_fit_candidate_chain", fake_fit_candidate_chain)

    params = regime_mod._select_best_params(features)

    assert params["covariance_type"] == "full"
    assert params["random_state"] == 0


def test_optimize_regime_model_force_retrains_dominated_fit(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    features = np.ones((20, 2), dtype=float)
    calls = []

    class DummyModel:
        def __init__(self, params):
            self.params = params

    def fake_select_best_params(features):
        return {
            "n_components": 3,
            "covariance_type": "diag",
            "n_iter": 100,
            "random_state": 42,
        }

    def fake_fit_with_params(features, params):
        calls.append(dict(params))
        return DummyModel(dict(params))

    def fake_predict_label_sequence(model, features):
        if model.params["covariance_type"] == "full":
            return ["lv_down", "lv_up", "hv_down", "hv_up"] * 5
        return ["lv_down"] * len(features)

    monkeypatch.setattr(regime_mod, "_select_best_params", fake_select_best_params)
    monkeypatch.setattr(regime_mod, "_fit_with_params", fake_fit_with_params)
    monkeypatch.setattr(regime_mod, "_predict_label_sequence", fake_predict_label_sequence)

    model = optimize_regime_model(features)

    assert calls[-1]["n_components"] == 4
    assert calls[-1]["covariance_type"] == "full"
    assert model.params == calls[-1]
