import numpy as np

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


def test_fit_features_isolates_training_window_no_lookahead():
    """Modifying future bars must not change anomaly labels for the training window."""
    rng = np.random.default_rng(0)
    features = rng.standard_normal((50, 2))
    fit_window = features[:25].copy()

    labels_v1 = detect_anomalies(features, fit_features=fit_window)

    features_modified = features.copy()
    features_modified[25:] = rng.standard_normal((25, 2)) * 100  # extreme future data
    labels_v2 = detect_anomalies(features_modified, fit_features=fit_window)

    # Past-window predictions must be identical regardless of future data changes
    np.testing.assert_array_equal(labels_v1[:25], labels_v2[:25])
