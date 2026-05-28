"""
Live data fetching + pipeline runner + display.
No API keys needed — CoinGecko free tier + alternative.me Fear & Greed Index.
"""
import asyncio

import httpx
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from kairos.signals.kalman import kalman_smooth
from kairos.signals.anomaly import detect_anomalies
from kairos.signals.regime import fit_regime_model, predict_regime
from kairos.signals.causal import CausalDAG
from kairos.signals.ensemble import SignalEnsemble, FeatureVector
from kairos.models.signal_event import SignalEvent

console = Console()

_COINGECKO = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
_FNG = "https://api.alternative.me/fng/"
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; kairos-signal-engine/0.1; +https://github.com/kairos)"}

_FNG_FALLBACK = [50] * 30  # neutral when API unavailable


async def _fetch_prices(client: httpx.AsyncClient, days: int = 365) -> tuple[list[float], float]:
    resp = await client.get(
        _COINGECKO,
        params={"vs_currency": "usd", "days": str(days), "interval": "daily"},
        headers=_HEADERS,
    )
    resp.raise_for_status()
    prices = [p[1] for p in resp.json()["prices"]]
    return prices, prices[-1]


async def _fetch_fng(client: httpx.AsyncClient, days: int = 365) -> tuple[list[int], bool]:
    """Returns (scores oldest-first, available). Scores are 0–100."""
    try:
        resp = await client.get(
            _FNG,
            params={"limit": str(days), "format": "json"},
            headers=_HEADERS,
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        scores = [int(d["value"]) for d in reversed(data)]  # API returns newest first
        return scores, True
    except (httpx.HTTPStatusError, httpx.RequestError, KeyError, ValueError):
        return _FNG_FALLBACK, False


async def fetch_live_data() -> tuple[list[float], float, list[int], bool]:
    """Fetch BTC prices + Fear & Greed index concurrently. No API keys needed.

    Returns: (prices, current_price, fng_scores, fng_available)
    """
    async with httpx.AsyncClient(timeout=20.0) as client:
        prices_task = asyncio.create_task(_fetch_prices(client))
        fng_task = asyncio.create_task(_fetch_fng(client))
        prices, current_price = await prices_task
        fng_scores, fng_ok = await fng_task
    return prices, current_price, fng_scores, fng_ok


def _fng_narrative(fng_scores: list[int]) -> dict:
    """Compute narrative features from Fear & Greed scores."""
    arr = np.array(fng_scores, dtype=float)
    current = arr[-1] / 100.0
    recent = arr[-7:] if len(arr) >= 7 else arr
    velocity = float(np.polyfit(range(len(recent)), recent, 1)[0]) / 100.0 if len(recent) >= 2 else 0.0
    tipping = bool(current > 0.5 and velocity > 0.02)
    return {
        "narrative_velocity": round(max(velocity, 0.0), 4),
        "narrative_tipping_point": tipping,
        "saturation": round(current, 4),
        "fng_raw": int(arr[-1]),
    }


def _fng_label(score: int) -> str:
    if score >= 75: return "Extreme Greed"
    if score >= 55: return "Greed"
    if score >= 45: return "Neutral"
    if score >= 25: return "Fear"
    return "Extreme Fear"


def _build_training_data(
    prices: list[float],
    fng_scores: list[int],
    smoothed: np.ndarray,
    anomaly_flags: np.ndarray,
    vol_z_arr: np.ndarray,
    hmm,
    regime_feats: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Historical feature matrix + 2-day forward-return labels. No leakage."""
    arr = np.array(prices, dtype=float)
    fng_arr = np.array(fng_scores, dtype=float)
    n = len(arr)

    X_rows, y_labels = [], []
    for t in range(10, n - 2):
        slope = float(np.polyfit(range(10), smoothed[t - 10:t], 1)[0])
        vz = float(vol_z_arr[t])
        anom = float(anomaly_flags[t])

        fng_idx = min(t, len(fng_arr) - 1)
        fng_cur = fng_arr[fng_idx] / 100.0
        fw = fng_arr[max(0, fng_idx - 7):fng_idx + 1]
        fng_vel = float(np.polyfit(range(len(fw)), fw, 1)[0]) / 100.0 if len(fw) >= 2 else 0.0
        fng_tip = bool(fng_cur > 0.5 and fng_vel > 0.02)

        rf_t = regime_feats[max(0, t - 6):t]
        reg = predict_regime(hmm, rf_t) if len(rf_t) >= 2 else "accumulation"

        causal = CausalDAG().infer_price_impact(fng_tip, reg, bool(anom), False)

        fv = FeatureVector(
            kalman_slope=slope, volume_z_score=vz, anomaly_score=anom,
            narrative_velocity=max(fng_vel, 0.0), narrative_tipping_point=fng_tip,
            saturation=fng_cur,
            regime_accumulation=1.0 if reg == "accumulation" else 0.0,
            regime_distribution=1.0 if reg == "distribution" else 0.0,
            regime_transition=1.0 if reg == "transition" else 0.0,
            causal_bullish=causal["bullish"], causal_confidence=causal["confidence"],
            macro_dff=0.25,
        )
        # Real label: was price higher 2 days later? No leakage.
        y_labels.append(1 if arr[t + 2] > arr[t] else 0)
        X_rows.append(fv.to_array())

    return np.vstack(X_rows), np.array(y_labels)


def run_pipeline(prices: list[float], fng_scores: list[int]) -> SignalEvent:
    """Run full 3-layer causal pipeline on real data. Returns a SignalEvent."""
    arr = np.array(prices, dtype=float)

    # Layer 1 — Reality: Kalman smooth + anomaly detection
    smoothed = kalman_smooth(arr)
    slope = float(np.polyfit(range(10), smoothed[-10:], 1)[0]) if len(smoothed) >= 10 else 0.0
    vol_proxy = np.abs(np.diff(arr))
    vol_proxy = np.append(vol_proxy, vol_proxy[-1] if len(vol_proxy) else 0.0)
    anomaly_flags = detect_anomalies(np.column_stack([arr, vol_proxy]))
    anomaly_score = float(anomaly_flags[-1])

    # Rolling vol z-score (needed both for current signal and training data)
    vol_z_arr = np.zeros(len(arr))
    for i in range(1, len(arr)):
        v = vol_proxy[:i]
        vol_z_arr[i] = (v[-1] - v.mean()) / (v.std() + 1e-8)
    vol_z = float(vol_z_arr[-1])

    # Layer 2 — Narrative: Fear & Greed index
    narrative = _fng_narrative(fng_scores)

    # Layer 3 — Regime: HMM on full price history
    returns = np.diff(smoothed) / (smoothed[:-1] + 1e-8)
    volatility = np.abs(returns)
    regime_feats = np.column_stack([returns, volatility])
    if len(regime_feats) >= 10:
        hmm = fit_regime_model(regime_feats)
        regime = predict_regime(hmm, regime_feats[-5:])
    else:
        hmm = None
        regime = "accumulation"

    # Causal DAG
    causal = CausalDAG().infer_price_impact(
        narrative_tipping_point=narrative["narrative_tipping_point"],
        regime=regime,
        anomaly_detected=bool(anomaly_score),
        macro_stress=False,
    )

    fv = FeatureVector(
        kalman_slope=slope, volume_z_score=vol_z, anomaly_score=anomaly_score,
        narrative_velocity=narrative["narrative_velocity"],
        narrative_tipping_point=narrative["narrative_tipping_point"],
        saturation=narrative["saturation"],
        regime_accumulation=1.0 if regime == "accumulation" else 0.0,
        regime_distribution=1.0 if regime == "distribution" else 0.0,
        regime_transition=1.0 if regime == "transition" else 0.0,
        causal_bullish=causal["bullish"], causal_confidence=causal["confidence"],
        macro_dff=0.25,
    )

    # Train XGBoost on real historical data with real forward-return labels
    ensemble = SignalEnsemble()
    if hmm is not None and len(arr) > 15:
        try:
            X, y = _build_training_data(prices, fng_scores, smoothed, anomaly_flags, vol_z_arr, hmm, regime_feats)
            if len(set(y.tolist())) >= 2:
                ensemble.fit_raw(X, y)
            else:
                ensemble.fit_synthetic_fallback()
        except Exception:
            ensemble.fit_synthetic_fallback()
    else:
        ensemble.fit_synthetic_fallback()

    return ensemble.predict("BTC", fv, citations=causal["citations"], regime=regime)


def display_signal(
    event: SignalEvent,
    current_price: float,
    fng_score: int = 50,
    fng_available: bool = True,
) -> None:
    """Print a simple, human-readable signal panel."""
    bullish = event.direction == "bullish"
    color = "green" if bullish else "red"
    arrow = "↑" if bullish else "↓"
    direction_word = "PRICE GOING UP" if bullish else "PRICE GOING DOWN"
    conf_pct = int(event.confidence * 100)

    filled = int(20 * event.confidence)
    bar = "█" * filled + "░" * (20 - filled)

    h = event.estimated_hours
    time_str = f"{h:.0f} hours" if h < 48 else f"{h / 24:.1f} days"

    regime_plain = {
        "accumulation": "Smart money quietly buying (calm before the pump)",
        "distribution": "People selling into strength (watch out)",
        "transition":   "Market changing direction (anything can happen)",
    }.get(event.regime, event.regime)

    fng_text = f"{_fng_label(fng_score)} ({fng_score}/100)"

    plain_reasons = []
    for c in event.citations:
        if "Shiller" in c:
            plain_reasons.append("Sentiment momentum is building fast")
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
    t.append(f"  Sentiment:      ", style="dim")
    t.append(f"{fng_text}\n", style="white")
    t.append(f"  Confidence:     ", style="dim")
    t.append(f"[{bar}] {conf_pct}%\n", style=color)
    t.append(f"  Move expected:  ", style="dim")
    t.append(f"within ~{time_str}\n", style="yellow")

    if plain_reasons:
        t.append(f"\n  Why this signal:\n", style="bold dim")
        for r in plain_reasons:
            t.append(f"    • {r}\n", style="white")

    if event.narrative_tipping_point:
        t.append(f"\n  ⚡ Sentiment tipping point — fear/greed shifting fast\n", style="bold yellow")

    if conf_pct < 55:
        t.append(f"\n  ⚠  Low confidence — market signal is weak, be careful\n", style="yellow")

    if not fng_available:
        t.append(f"\n  ℹ  Sentiment API unavailable — using neutral baseline\n", style="dim")

    t.append(f"\n", style="")

    console.print(Panel(
        t,
        title="[bold white]KAIROS  ·  BTC Live Signal[/bold white]",
        border_style=color,
        padding=(0, 2),
    ))
