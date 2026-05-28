import numpy as np
from sklearn.ensemble import IsolationForest


def detect_anomalies(features: np.ndarray, contamination: float = 0.1) -> np.ndarray:
    """
    Detect anomalies using Isolation Forest. Returns boolean array: True = anomaly.
    contamination = expected proportion of outliers (default 10%).
    """
    clf = IsolationForest(contamination=contamination, random_state=42)
    labels = clf.fit_predict(features)
    return (labels == -1)
