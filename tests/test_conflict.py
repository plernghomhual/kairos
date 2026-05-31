"""Tests for the layer-conflict detection module."""

import pytest

from kairos.signals.conflict import (
    _compute_vma,
    capitulation_triggered,
    detect_conflict,
    layer_votes,
    regime_vote,
    seller_exhaustion_active,
    sentiment_vote,
    trend_vote,
    volume_status_label,
)

# ── trend_vote ────────────────────────────────────────────────────────────────


def test_trend_vote_strong_up():
    assert trend_vote(150) == "bullish"


def test_trend_vote_gentle_up():
    assert trend_vote(30) == "bullish"


def test_trend_vote_flat_zero():
    assert trend_vote(0) == "neutral"


def test_trend_vote_flat_small():
    assert trend_vote(10) == "neutral"


def test_trend_vote_gentle_down():
    assert trend_vote(-30) == "bearish"


def test_trend_vote_strong_down():
    assert trend_vote(-150) == "bearish"


def test_trend_vote_boundary_positive():
    assert trend_vote(20) == "neutral"
    assert trend_vote(20.01) == "bullish"


def test_trend_vote_boundary_negative():
    assert trend_vote(-20) == "neutral"
    assert trend_vote(-20.01) == "bearish"


# ── regime_vote ───────────────────────────────────────────────────────────────


def test_regime_lv_up():
    assert regime_vote("lv_up") == "bullish"


def test_regime_lv_down():
    assert regime_vote("lv_down") == "bearish"


def test_regime_hv_up():
    assert regime_vote("hv_up") == "bearish"


def test_regime_hv_down():
    assert regime_vote("hv_down") == "bullish"


def test_regime_unknown():
    assert regime_vote("unknown") == "neutral"
    assert regime_vote("") == "neutral"


# ── sentiment_vote ────────────────────────────────────────────────────────────


def test_sentiment_extreme_fear():
    assert sentiment_vote(10) == "bullish"
    assert sentiment_vote(25) == "bullish"


def test_sentiment_fear():
    assert sentiment_vote(30) == "bullish"
    assert sentiment_vote(40) == "bullish"


def test_sentiment_neutral():
    assert sentiment_vote(45) == "neutral"
    assert sentiment_vote(50) == "neutral"
    assert sentiment_vote(55) == "neutral"


def test_sentiment_greed():
    assert sentiment_vote(60) == "bearish"
    assert sentiment_vote(70) == "bearish"


def test_sentiment_extreme_greed():
    assert sentiment_vote(80) == "bearish"
    assert sentiment_vote(100) == "bearish"


def test_sentiment_none():
    assert sentiment_vote(None) == "neutral"


# ── detect_conflict ───────────────────────────────────────────────────────────


def test_conflict_all_bullish_no_conflict():
    """Trend up, lv_up, fear → all bullish → no conflict."""
    assert detect_conflict(30, "lv_up", 20) is False


def test_conflict_all_bearish_no_conflict():
    """Trend down, lv_down, greed → all bearish → no conflict."""
    assert detect_conflict(-30, "lv_down", 80) is False


def test_conflict_trend_bearish_sentiment_bullish():
    """Trend down, lv_down, fear → trend says sell, sentiment says buy."""
    assert detect_conflict(-30, "lv_down", 20) is True


def test_conflict_trend_bullish_sentiment_bearish():
    """Trend up, lv_up, greed → conflict."""
    assert detect_conflict(30, "lv_up", 80) is True


def test_conflict_hv_up_neutral_layers():
    """hv_up → all neutral → no conflict."""
    assert detect_conflict(0, "hv_up", 50) is False


def test_conflict_no_fng():
    """No FNG available → neutral → no conflict even with extreme trend."""
    assert detect_conflict(30, "lv_up", None) is False
    assert detect_conflict(-30, "lv_down", None) is False


# ── layer_votes ───────────────────────────────────────────────────────────────


def test_layer_votes_keys():
    votes = layer_votes(30, "lv_up", 20)
    assert set(votes.keys()) == {"trend", "regime", "sentiment"}


def test_layer_votes_values():
    votes = layer_votes(-50, "lv_down", 80)
    assert votes["trend"] == "bearish"
    assert votes["regime"] == "bearish"
    assert votes["sentiment"] == "bearish"


# ── Volume helpers ────────────────────────────────────────────────────────────


def test_compute_vma_normal():
    assert _compute_vma([10, 20, 30], 2) == 25.0


def test_compute_vma_empty():
    assert _compute_vma([], 20) == 0.0


def test_compute_vma_exact_period():
    assert _compute_vma([100] * 20, 20) == 100.0


def test_compute_vma_short_history():
    assert _compute_vma([50, 50], 5) == 50.0


def test_volume_status_climax():
    assert volume_status_label(150, [100] * 20) == "Climax"


def test_volume_status_exhaustion():
    assert volume_status_label(50, [100] * 20) == "Exhaustion"


def test_volume_status_normal():
    assert volume_status_label(110, [100] * 20) == "Normal"


def test_volume_status_empty_history():
    assert volume_status_label(100, []) == "Normal"


def test_capitulation_triggered_true():
    assert capitulation_triggered(10, 200, [100] * 20) is True


def test_capitulation_triggered_fng_not_extreme():
    assert capitulation_triggered(50, 200, [100] * 20) is False


def test_capitulation_triggered_low_volume():
    assert capitulation_triggered(10, 100, [100] * 20) is False


def test_capitulation_triggered_no_fng():
    assert capitulation_triggered(None, 200, [100] * 20) is False


def test_seller_exhaustion_active_true():
    # Falling fast (slope < -100) + low volume (< 0.75 VMA)
    assert seller_exhaustion_active(-150, 50, [100] * 20) is True


def test_seller_exhaustion_volume_too_high():
    assert seller_exhaustion_active(-150, 150, [100] * 20) is False


def test_seller_exhaustion_not_falling_fast():
    assert seller_exhaustion_active(-50, 50, [100] * 20) is False


def test_seller_exhaustion_empty_history():
    assert seller_exhaustion_active(-150, 50, []) is False


@pytest.mark.parametrize(
    ("slope", "fng_score", "expected"),
    [
        (1000.0, 0, False),
        (1000.0, 100, True),
        (-1000.0, 0, True),
        (-1000.0, 100, True),
    ],
)
def test_detect_conflict_extreme_values(slope, fng_score, expected):
    assert detect_conflict(slope, "lv_up", fng_score) is expected


def test_volume_status_empty_history_extreme_volume_is_normal():
    assert volume_status_label(1_000_000_000.0, []) == "Normal"


def test_capitulation_triggered_exact_boundaries_are_inactive():
    assert capitulation_triggered(25, 150, [100] * 20) is False
    assert capitulation_triggered(24, 149.99, [100] * 20) is False
    assert capitulation_triggered(24, 150, [100] * 20) is True


def test_seller_exhaustion_exact_boundaries_are_inactive():
    assert seller_exhaustion_active(-100, 75, [100] * 20) is False
    assert seller_exhaustion_active(-100.01, 75, [100] * 20) is False
    assert seller_exhaustion_active(-100.01, 74.99, [100] * 20) is True
