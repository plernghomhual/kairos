import math

import numpy as np
import pytest

from kairos.live import _fng_narrative, run_pipeline
from kairos.models.signal_event import SignalEvent
from kairos.signals.anomaly import detect_anomalies
from kairos.signals.kalman import kalman_smooth
from kairos.signals.regime import fit_regime_model, predict_regime


def _prices(n=80):
    return [100.0 + i for i in range(n)]


def _assert_valid_event(event):
    assert isinstance(event, SignalEvent)
    assert event.direction in ("bullish", "bearish", "neutral")
    assert 0.0 <= event.confidence <= 1.0
    assert math.isfinite(event.confidence)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_price_array_is_sanitized_without_crash(bad_value):
    prices = _prices()
    prices[10] = bad_value
    _assert_valid_event(run_pipeline(prices, fng_scores=[50] * len(prices)))


def test_nan_in_fng_scores_is_filtered_without_crash():
    fng_scores = [50] * 80
    fng_scores[12] = float("nan")
    _assert_valid_event(run_pipeline(_prices(), fng_scores=fng_scores))


def test_empty_volume_array_falls_back_to_price_delta_proxy():
    event = run_pipeline(_prices(), fng_scores=[50] * 80, volumes=[])
    _assert_valid_event(event)


def test_very_large_price_jump_is_detectable_as_anomaly():
    prices = np.array([100.0] * 79 + [10_000.0])
    features = np.column_stack([prices, np.ones_like(prices)])
    flags = detect_anomalies(features)
    assert flags.shape == (80,)
    assert bool(flags[-1]) is True


def test_repeated_identical_prices_still_fit_hmm():
    features = np.tile(
        np.array([[0.0, 0.001], [0.0, 0.002], [0.0, 0.003], [0.0, 0.004]]),
        (20, 1),
    )
    model = fit_regime_model(features)
    assert predict_regime(model, features[-1:]) in (
        "lv_up",
        "hv_up",
        "lv_down",
        "hv_down",
    )


def test_negative_price_values_return_safe_signal():
    prices = _prices()
    prices[20] = -1.0
    _assert_valid_event(run_pipeline(prices, fng_scores=[50] * 80))


def test_out_of_range_fng_scores_are_clamped():
    narrative = _fng_narrative([200, 250, 300])
    assert narrative["fng_raw"] == 100
    assert narrative["saturation"] == pytest.approx(1.0)


def test_string_in_numeric_price_array_raises_clear_error():
    prices = _prices()
    prices[5] = "not-a-number"
    with pytest.raises(ValueError, match="could not convert"):
        run_pipeline(prices, fng_scores=[50] * 80)


def test_mixed_numeric_price_types_convert_safely():
    prices = _prices()
    prices[5] = "105.0"
    _assert_valid_event(run_pipeline(prices, fng_scores=[50] * 80))


def test_kalman_preserves_nan_as_non_finite_signal_marker():
    smoothed = kalman_smooth(np.array([1.0, np.nan, 2.0]))
    assert len(smoothed) == 3
    assert np.isnan(smoothed[1])


def test_anomaly_detector_handles_nan_rows_as_non_anomalous():
    flags = detect_anomalies(np.array([[1.0, 2.0], [np.nan, 3.0]]))
    assert flags.tolist() == [False, False]


def test_fng_negative_values_clamp_to_zero():
    narrative = _fng_narrative([-50, -10, -1])
    assert narrative["fng_raw"] == 0
    assert narrative["saturation"] == pytest.approx(0.0)
