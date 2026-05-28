from dataclasses import asdict
from datetime import datetime, timezone
from kairos.models.signal_event import SignalEvent

def test_signal_event_serializes_to_dict():
    event = SignalEvent(
        id="abc123",
        asset="BTC",
        direction="bullish",
        confidence=0.74,
        regime="accumulation",
        narrative_velocity=2.3,
        narrative_tipping_point=True,
        mechanism="narrative_momentum → retail_fomo → price",
        estimated_hours=36.0,
        citations=["Shiller 2017 - Narrative Economics", "Minsky Stage 2"],
        triggered_at=datetime(2026, 5, 28, 14, 22, 0, tzinfo=timezone.utc),
    )
    d = asdict(event)
    assert d["asset"] == "BTC"
    assert d["confidence"] == 0.74
    assert len(d["citations"]) == 2
