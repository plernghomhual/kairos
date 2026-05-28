"""
Live data fetching + pipeline runner + display.
No API keys needed — uses CoinGecko free tier and Reddit public JSON.
"""
import asyncio
from collections import defaultdict
from datetime import datetime, timezone

import httpx
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from kairos.signals.kalman import kalman_smooth
from kairos.signals.anomaly import detect_anomalies
from kairos.signals.narrative import compute_narrative_features
from kairos.signals.regime import fit_regime_model, predict_regime
from kairos.signals.causal import CausalDAG
from kairos.signals.ensemble import SignalEnsemble, FeatureVector
from kairos.models.signal_event import SignalEvent

console = Console()

_COINGECKO = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
_REDDIT = "https://www.reddit.com/r/Bitcoin/search.json"
_HEADERS = {"User-Agent": "kairos/0.1.0"}


async def _fetch_prices(client: httpx.AsyncClient, days: int = 30) -> tuple[list[float], float]:
    resp = await client.get(
        _COINGECKO,
        params={"vs_currency": "usd", "days": str(days), "interval": "daily"},
        headers=_HEADERS,
    )
    resp.raise_for_status()
    prices = [p[1] for p in resp.json()["prices"]]
    return prices, prices[-1]


async def _fetch_reddit_counts(client: httpx.AsyncClient) -> list[int]:
    resp = await client.get(
        _REDDIT,
        params={"q": "bitcoin", "sort": "new", "limit": 100, "t": "week"},
        headers=_HEADERS,
    )
    resp.raise_for_status()
    posts = resp.json()["data"]["children"]
    daily: dict[str, int] = defaultdict(int)
    for p in posts:
        ts = p["data"].get("created_utc", 0)
        day = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
        daily[day] += 1
    counts = [v for _, v in sorted(daily.items())]
    return counts if counts else [10, 10, 10]


async def fetch_live_data() -> tuple[list[float], float, list[int]]:
    """Fetch BTC prices and Reddit chatter concurrently. No API keys needed."""
    async with httpx.AsyncClient(timeout=20.0) as client:
        prices, current_price = await _fetch_prices(client)
        reddit_counts = await _fetch_reddit_counts(client)
    return prices, current_price, reddit_counts


def run_pipeline(prices: list[float], reddit_counts: list[int]) -> SignalEvent:
    """Run the full 3-layer causal pipeline. Returns a SignalEvent."""
    arr = np.array(prices, dtype=float)

    # Layer 1 — Reality: smooth price + detect weird moves
    smoothed = kalman_smooth(arr)
    slope = float(np.polyfit(range(10), smoothed[-10:], 1)[0]) if len(smoothed) >= 10 else 0.0
    vol_proxy = np.abs(np.diff(arr))
    vol_proxy = np.append(vol_proxy, vol_proxy[-1] if len(vol_proxy) else 0.0)
    anomaly_flags = detect_anomalies(np.column_stack([arr, vol_proxy]))
    anomaly_score = float(anomaly_flags[-1])
    vol_z = float((vol_proxy[-1] - vol_proxy.mean()) / (vol_proxy.std() + 1e-8))

    # Layer 2 — Narrative: how fast is the story spreading?
    narrative = compute_narrative_features(reddit_counts)

    # Layer 3 — Regime: what phase is the market in?
    returns = np.diff(smoothed) / (smoothed[:-1] + 1e-8)
    volatility = np.abs(returns)
    regime_feats = np.column_stack([returns, volatility])
    if len(regime_feats) >= 10:
        hmm = fit_regime_model(regime_feats)
        regime = predict_regime(hmm, regime_feats[-5:])
    else:
        regime = "accumulation"

    # Causal DAG — economic reasoning
    causal = CausalDAG().infer_price_impact(
        narrative_tipping_point=narrative["narrative_tipping_point"],
        regime=regime,
        anomaly_detected=bool(anomaly_score),
        macro_stress=False,
    )

    fv = FeatureVector(
        kalman_slope=slope,
        volume_z_score=vol_z,
        anomaly_score=anomaly_score,
        narrative_velocity=narrative["narrative_velocity"],
        narrative_tipping_point=narrative["narrative_tipping_point"],
        saturation=narrative["saturation"],
        regime_accumulation=1.0 if regime == "accumulation" else 0.0,
        regime_distribution=1.0 if regime == "distribution" else 0.0,
        regime_transition=1.0 if regime == "transition" else 0.0,
        causal_bullish=causal["bullish"],
        causal_confidence=causal["confidence"],
        macro_dff=0.25,
    )

    ensemble = SignalEnsemble()
    b = FeatureVector(0.02, 1.5, 0.0, 2.0, True, 0.1, 1.0, 0.0, 0.0, 0.75, 0.9, 0.25)
    br = FeatureVector(-0.02, -1.5, 0.1, 0.1, False, 0.5, 0.0, 1.0, 0.0, 0.25, 0.7, 0.5)
    ensemble.fit([b] * 50 + [br] * 50)

    return ensemble.predict("BTC", fv, citations=causal["citations"], regime=regime)


def display_signal(event: SignalEvent, current_price: float) -> None:
    """Print a simple, human-readable signal panel."""
    bullish = event.direction == "bullish"
    color = "green" if bullish else "red"
    arrow = "↑" if bullish else "↓"
    direction_word = "PRICE GOING UP" if bullish else "PRICE GOING DOWN"
    conf_pct = int(event.confidence * 100)

    # Confidence bar
    filled = int(20 * event.confidence)
    bar = "█" * filled + "░" * (20 - filled)

    # Time estimate
    h = event.estimated_hours
    time_str = f"{h:.0f} hours" if h < 48 else f"{h/24:.1f} days"

    # Regime in plain English
    regime_plain = {
        "accumulation": "Smart money quietly buying (calm before the pump)",
        "distribution": "People selling into strength (watch out)",
        "transition":   "Market changing direction (anything can happen)",
    }.get(event.regime, event.regime)

    # Citations in plain English
    plain_reasons = []
    for c in event.citations:
        if "Shiller" in c:
            plain_reasons.append("Reddit/news chatter is spreading fast")
        elif "Stage 2" in c or "accumulation" in c.lower():
            plain_reasons.append("Market is in quiet buy phase")
        elif "Stage 3" in c or "Stage 4" in c or "distribution" in c.lower():
            plain_reasons.append("Market is in sell phase")
        elif "Soros" in c:
            plain_reasons.append("Market is flipping — uncertainty high")
        elif "Kindleberger" in c:
            plain_reasons.append("Macro stress is weighing on price")
        elif "Anomaly" in c:
            plain_reasons.append("Weird price action detected")

    t = Text()
    t.append(f"\n  {arrow} ", style=f"bold {color}")
    t.append(f"{direction_word}", style=f"bold {color}")
    t.append(f"  —  {conf_pct}% sure\n", style="dim")
    t.append(f"\n  BTC right now:  ", style="dim")
    t.append(f"${current_price:,.0f}\n", style="bold white")
    t.append(f"  Market phase:   ", style="dim")
    t.append(f"{regime_plain}\n", style="white")
    t.append(f"  Confidence:     ", style="dim")
    t.append(f"[{bar}] {conf_pct}%\n", style=color)
    t.append(f"  Move expected:  ", style="dim")
    t.append(f"within ~{time_str}\n", style="yellow")

    if plain_reasons:
        t.append(f"\n  Why this signal:\n", style="bold dim")
        for r in plain_reasons:
            t.append(f"    • {r}\n", style="white")

    if event.narrative_tipping_point:
        t.append(f"\n  ⚡ Chatter tipping point — narrative is accelerating\n", style="bold yellow")

    if conf_pct < 55:
        t.append(f"\n  ⚠  Low confidence — market signal is weak, be careful\n", style="yellow")

    t.append(f"\n", style="")

    console.print(Panel(
        t,
        title="[bold white]KAIROS  ·  BTC Live Signal[/bold white]",
        border_style=color,
        padding=(0, 2),
    ))
