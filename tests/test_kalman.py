import numpy as np
from kairos.signals.kalman import kalman_smooth


def test_kalman_smooth_reduces_noise():
    np.random.seed(42)
    true_signal = np.linspace(100, 200, 50)
    noisy_signal = true_signal + np.random.normal(0, 10, 50)
    smoothed = kalman_smooth(noisy_signal)

    assert len(smoothed) == len(noisy_signal)
    noise_before = np.std(np.diff(noisy_signal))
    noise_after = np.std(np.diff(smoothed))
    assert noise_after < noise_before


def test_kalman_smooth_returns_float_array():
    signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = kalman_smooth(signal)
    assert result.dtype == np.float64
