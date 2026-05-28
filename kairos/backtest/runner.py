from dataclasses import dataclass
from typing import Any


HIT_RATE_THRESHOLD = 0.70


@dataclass
class BacktestResult:
    total: int
    hits: int
    hit_rate: float
    passes_threshold: bool


def evaluate_hit_rate(
    signals: list[dict[str, Any]],
    actuals: list[bool],
) -> BacktestResult:
    assert len(signals) == len(actuals), "signals and actuals must be same length"
    hits = sum(1 for correct in actuals if correct)
    total = len(actuals)
    hit_rate = hits / total if total > 0 else 0.0
    return BacktestResult(
        total=total,
        hits=hits,
        hit_rate=round(hit_rate, 4),
        passes_threshold=hit_rate >= HIT_RATE_THRESHOLD,
    )
