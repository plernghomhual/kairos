import numpy as np
import pytest
from kairos.signals.regime import fit_regime_model, predict_regime


def make_regime_data():
    np.random.seed(0)
    acc = np.column_stack([
        np.random.normal(0.001, 0.005, 60),
        np.random.normal(0.005, 0.001, 60),
    ])
    dist = np.column_stack([
        np.random.normal(-0.001, 0.005, 60),
        np.random.normal(0.005, 0.001, 60),
    ])
    trans = np.column_stack([
        np.random.normal(0, 0.02, 30),
        np.random.normal(0.02, 0.005, 30),
    ])
    return np.vstack([acc, dist, trans])


def test_fit_and_predict_regime():
    data = make_regime_data()
    model = fit_regime_model(data, n_states=3)
    regime = predict_regime(model, data[-1:])
    assert regime in ("accumulation", "distribution", "transition")


def test_regime_is_consistent():
    data = make_regime_data()
    model = fit_regime_model(data, n_states=3)
    r1 = predict_regime(model, data[-1:])
    r2 = predict_regime(model, data[-1:])
    assert r1 == r2
