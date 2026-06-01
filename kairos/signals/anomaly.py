import numpy as np
from sklearn.ensemble import IsolationForest


def detect_anomalies(
    features: np.ndarray,
    contamination: float = 0.1,
    *,
    fit_features: np.ndarray | None = None,
) -> np.ndarray:
    """Detect anomalies using Isolation Forest. Returns boolean array: True = anomaly.

    Parameters
    ----------
    features:
        Data to score (predict on).
    contamination:
        Expected proportion of outliers (default 10%).
    fit_features:
        Optional separate array to *fit* the detector on. When provided, the
        detector is trained on ``fit_features`` and then applied to ``features``.
        Use this in training pipelines to avoid fitting on data that includes
        future bars: pass only the warm-up portion of the window as
        ``fit_features`` and the full window as ``features``.
        When ``None`` (default, live/serving path), the detector fits and
        predicts on the same ``features`` array.
    """
    clf = IsolationForest(contamination=contamination, random_state=42)
    train_data = fit_features if fit_features is not None else features
    clf.fit(train_data)
    labels = clf.predict(features)
    return labels == -1
