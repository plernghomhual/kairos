import numpy as np
from hmmlearn.hmm import GaussianHMM

REGIME_LABELS = {0: "accumulation", 1: "distribution", 2: "transition"}


def fit_regime_model(features: np.ndarray, n_states: int = 3) -> GaussianHMM:
    """
    Fit Gaussian HMM to feature matrix (returns, volatility).
    State 0 = lowest volatility regime (accumulation).
    """
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=200,
        random_state=42,
    )
    model.fit(features)
    return model


def predict_regime(model: GaussianHMM, features: np.ndarray) -> str:
    """
    Predict current market regime. Maps HMM state to label sorted by volatility.
    """
    state = model.predict(features)[-1]
    means = model.means_[:, 1] if model.means_.shape[1] > 1 else model.means_[:, 0]
    sorted_states = np.argsort(means)
    rank = int(np.where(sorted_states == state)[0][0])
    return REGIME_LABELS.get(rank, "accumulation")
