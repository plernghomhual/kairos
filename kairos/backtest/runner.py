from dataclasses import dataclass
from typing import Any

HIT_RATE_THRESHOLD = 0.70


@dataclass
class HitRateResult:
    total: int
    hits: int
    hit_rate: float
    passes_threshold: bool


def evaluate_hit_rate(
    signals: list[dict[str, Any]],
    actuals: list[bool],
) -> HitRateResult:
    """Evaluate signal hit rate.

    Parameters
    ----------
    signals:
        List of signal dicts (used for length validation; not directly inspected).
    actuals:
        Pre-aligned correctness booleans — ``True`` when the corresponding signal
        was directionally correct (price moved as predicted). Must be the same
        length as *signals* and already aligned element-by-element (index 0 of
        *actuals* corresponds to index 0 of *signals*).

    The caller is responsible for aligning signal direction against realised
    outcomes before calling this function.
    """
    if len(signals) != len(actuals):
        raise ValueError(f"signals ({len(signals)}) and actuals ({len(actuals)}) must be same length")
    hits = sum(1 for correct in actuals if correct)
    total = len(actuals)
    hit_rate = hits / total if total > 0 else 0.0
    return HitRateResult(
        total=total,
        hits=hits,
        hit_rate=round(hit_rate, 4),
        passes_threshold=hit_rate >= HIT_RATE_THRESHOLD,
    )
