import pytest
from kairos.signals.narrative import compute_narrative_features


def test_growing_narrative_has_positive_velocity():
    post_counts = [10, 15, 22, 35, 52, 80, 120]
    result = compute_narrative_features(post_counts)
    assert result["narrative_velocity"] > 0
    assert "narrative_tipping_point" in result
    assert isinstance(result["narrative_tipping_point"], bool)


def test_flat_narrative_has_near_zero_velocity():
    flat_counts = [50, 51, 49, 50, 52, 48, 50]
    result = compute_narrative_features(flat_counts)
    assert abs(result["narrative_velocity"]) < 0.5


def test_tipping_point_detected_on_exponential_growth():
    exponential = [5, 8, 13, 21, 34, 55, 89]
    result = compute_narrative_features(exponential, population=1000)
    assert result["narrative_tipping_point"] == True
