"""Layer-conflict detection — shared between live pipeline and backtest engine.

When the three signal layers (trend, regime, sentiment) disagree,
the conflict gate returns NEUTRAL. The system stands aside rather
than guessing during mixed signals.

Overrides (bypass conflict gate):
  - Capitulation Buy: FNG < 25 + Volume >= 1.5x VMA → force BUY
  - Seller Exhaustion: Falling Fast + Volume < 0.75x VMA → partial allocation
"""

import numpy as np


def _compute_vma(volumes: list[float], period: int = 20) -> float:
    """Simple moving average of volumes over the given period."""
    if not volumes:
        return 0.0
    window = volumes[-period:] if len(volumes) >= period else volumes
    return float(np.mean(window))


def volume_status_label(volume: float, volumes_history: list[float]) -> str:
    """Classify current volume relative to its 20-day average.

    Returns 'Climax' (>= 150% VMA), 'Exhaustion' (<= 75% VMA), or 'Normal'.
    """
    vma = _compute_vma(volumes_history)
    if vma <= 0:
        return "Normal"
    ratio = volume / vma
    if ratio >= 1.5:
        return "Climax"
    if ratio <= 0.75:
        return "Exhaustion"
    return "Normal"


def capitulation_triggered(fng_score: int | None, volume: float, volumes_history: list[float]) -> bool:
    """Extreme Fear + Volume Climax signals a potential bottom.

    Conditions:
      1. FNG < 25 (Extreme Fear)
      2. Volume >= 1.5x 20-day VMA (Volume Climax)
    """
    if fng_score is None or fng_score >= 25:
        return False
    vma = _compute_vma(volumes_history)
    if vma <= 0:
        return False
    return volume / vma >= 1.5


def seller_exhaustion_active(kalman_slope: float, volume: float, volumes_history: list[float]) -> bool:
    """Falling Fast + low volume → downtrend running out of steam.

    Conditions:
      1. Kalman slope < -100 (Falling Fast)
      2. Volume < 0.75x 20-day VMA
    """
    if kalman_slope >= -100:
        return False
    vma = _compute_vma(volumes_history)
    if vma <= 0:
        return False
    return volume / vma < 0.75


def trend_vote(kalman_slope: float) -> str:
    """Map Kalman slope to layer vote: bullish / bearish / neutral."""
    if kalman_slope > 20:
        return "bullish"
    if kalman_slope < -20:
        return "bearish"
    return "neutral"


def regime_vote(regime: str) -> str:
    """Map HMM regime to layer vote — 4-regime quadrant model."""
    mapping = {
        "lv_up": "bullish",
        "hv_up": "bearish",
        "lv_down": "bearish",
        "hv_down": "bullish",
    }
    return mapping.get(regime, "neutral")


def sentiment_vote(fng_score: int | None) -> str:
    """Map Fear & Greed score to contrarian layer vote.

    Extreme fear → buy zone (bullish)
    Fear → lean buy (bullish)
    Greed → caution (bearish)
    Extreme greed → top zone (bearish)
    Neutral / unavailable → neutral
    """
    if fng_score is None:
        return "neutral"
    if fng_score <= 40:
        return "bullish"
    if fng_score >= 60:
        return "bearish"
    return "neutral"


def detect_conflict(
    kalman_slope: float,
    regime: str,
    fng_score: int | None = None,
) -> bool:
    """True when at least one layer votes bullish and one votes bearish.

    When layers disagree, the XGBoost ensemble is making a low-conviction
    prediction that historically loses money. The conflict gate catches
    this and forces a NEUTRAL signal.
    """
    tv = trend_vote(kalman_slope)
    rv = regime_vote(regime)
    sv = sentiment_vote(fng_score)

    has_bullish = sum(1 for v in (tv, rv, sv) if v == "bullish")
    has_bearish = sum(1 for v in (tv, rv, sv) if v == "bearish")
    return has_bullish > 0 and has_bearish > 0


def layer_votes(
    kalman_slope: float,
    regime: str,
    fng_score: int | None = None,
) -> dict[str, str]:
    """Return each layer's vote for display."""
    return {
        "trend": trend_vote(kalman_slope),
        "regime": regime_vote(regime),
        "sentiment": sentiment_vote(fng_score),
    }
