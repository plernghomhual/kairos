from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List
import uuid


@dataclass
class SignalEvent:
    asset: str
    direction: str                   # "bullish" | "bearish"
    confidence: float                # 0.0 – 1.0
    regime: str                      # "accumulation" | "distribution" | "transition"
    narrative_velocity: float        # SIR model dI/dt normalized
    narrative_tipping_point: bool
    mechanism: str                   # human-readable causal chain
    estimated_hours: float           # hours until price impact expected
    citations: List[str]
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    triggered_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
