import numpy as np
import pytest
from kairos.signals.anomaly import detect_anomalies


def test_detects_obvious_spike():
    np.random.seed(42)
    normal = np.random.normal(100, 2, 100)
    spike = np.array([500.0])
    data = np.concatenate([normal, spike]).reshape(-1, 1)
    flags = detect_anomalies(data)
    assert len(flags) == len(data)
    assert flags[-1] == True  # spike is flagged


def test_no_false_positives_on_clean_data():
    np.random.seed(42)
    clean = np.random.normal(100, 1, 200).reshape(-1, 1)
    flags = detect_anomalies(clean)
    false_positive_rate = flags.sum() / len(flags)
    assert false_positive_rate < 0.15
