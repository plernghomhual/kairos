import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

_VALID_DIRECTIONS = frozenset({"bullish", "bearish", "neutral"})


@dataclass
class SignalEvent:
    asset: str
    direction: str  # "bullish" | "bearish" | "neutral"
    confidence: float  # 0.0 – 1.0
    regime: str  # "accumulation" | "distribution" | "transition"
    narrative_velocity: float  # SIR model dI/dt normalized
    narrative_tipping_point: bool
    mechanism: str  # human-readable causal chain
    estimated_hours: float  # hours until price impact expected
    citations: list[str]
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    triggered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if self.direction not in _VALID_DIRECTIONS:
            raise ValueError(f"direction must be one of {sorted(_VALID_DIRECTIONS)}, got {self.direction!r}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence}")
