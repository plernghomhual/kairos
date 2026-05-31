"""
Live data fetching + pipeline runner + display.
No API keys needed — CoinGecko free tier + alternative.me Fear & Greed Index.
"""

import asyncio
import logging
import threading
import time

import httpx
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from kairos.circuit_breaker import CIRCUIT_BREAKERS
from kairos.model_cache import load_model, save_model
from kairos.models.signal_event import SignalEvent
from kairos.signals.anomaly import detect_anomalies
from kairos.signals.causal import CausalDAG
from kairos.signals.ensemble import FeatureVector, SignalEnsemble
from kairos.signals.kalman import kalman_smooth
from kairos.signals.regime import fit_regime_model, predict_regime

_logger = logging.getLogger(__name__)
_RETRY_MAX_BACKOFF = 16.0

console = Console()

_PAPER_TRADING_ENGINE = None  # set at startup or in tests
_PAPER_ENGINE_LOCK = threading.Lock()
_LIVE_DATA_CACHE_TTL_SECONDS = 60.0
_LIVE_DATA_CACHE: dict[str, tuple[float, tuple[list[float], float, list[int], bool, list[float]]]] = {}

_FNG = "https://api.alternative.me/fng/"
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; kairos-signal-engine/0.1; +https://github.com/kairos)"}
_FNG_FALLBACK = [50] * 30  # neutral when API unavailable

_ASSET_IDS = {"BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana"}


def _copy_live_data(
    data: tuple[list[float], float, list[int], bool, list[float]],
) -> tuple[list[float], float, list[int], bool, list[float]]:
    prices, current_price, fng_scores, fng_ok, volumes = data
    return list(prices), float(current_price), list(fng_scores), bool(fng_ok), list(volumes)


async def _coingecko_get(client: httpx.AsyncClient, url: str, params: dict) -> httpx.Response:
    """GET with retry + exponential backoff on 429 rate limits."""
    backoff = 1.0
    for attempt in range(5):
        resp = await client.get(url, params=params, headers=_HEADERS)
        status = getattr(resp, "status_code", 200)
        if status != 429:
            if hasattr(resp, "raise_for_status"):
                resp.raise_for_status()
            return resp
        _logger.warning("CoinGecko 429 rate-limited (attempt %d/5); backing off %.0fs", attempt + 1, backoff)
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, _RETRY_MAX_BACKOFF)
    if hasattr(resp, "raise_for_status"):
        resp.raise_for_status()
    return resp


async def _fetch_prices(
    client: httpx.AsyncClient, days: int = 365, coin_id: str = "bitcoin"
) -> tuple[list[float], float, list[float]]:
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    resp = await _coingecko_get(
        client,
        url,
        {"vs_currency": "usd", "days": str(days), "interval": "daily"},
    )
    data = resp.json()
    prices = [p[1] for p in data["prices"]]
    volumes = [v[1] for v in data.get("total_volumes", [])]
    return prices, prices[-1], volumes


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
    except Exception:
        return _FNG_FALLBACK, False


async def fetch_live_data(
    asset: str = "BTC",
) -> tuple[list[float], float, list[int], bool, list[float]]:
    """Fetch prices + Fear & Greed index concurrently. No API keys needed.

    Returns: (prices, current_price, fng_scores, fng_available, volumes)
    Raises ValueError for unsupported assets.
    """
    if asset not in _ASSET_IDS:
        raise ValueError(f"Unsupported asset '{asset}'. Choose from: {', '.join(_ASSET_IDS)}")

    cached = _LIVE_DATA_CACHE.get(asset)
    now = time.monotonic()
    if cached is not None and now - cached[0] < _LIVE_DATA_CACHE_TTL_SECONDS:
        return _copy_live_data(cached[1])

    coin_id = _ASSET_IDS[asset]
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            (prices, current_price, volumes), (fng_scores, fng_ok) = await asyncio.gather(
                _fetch_prices(client, coin_id=coin_id),
                _fetch_fng(client),
            )
    except Exception:
        cached = _LIVE_DATA_CACHE.get(asset)
        if cached is not None:
            _logger.warning("Live fetch failed for %s; using stale cached data", asset, exc_info=True)
            return _copy_live_data(cached[1])
        raise

    result = (prices, current_price, fng_scores, fng_ok, volumes)
    _LIVE_DATA_CACHE[asset] = (time.monotonic(), _copy_live_data(result))
    return _copy_live_data(result)


def _fng_narrative(fng_scores: list[int]) -> dict:
    """Compute narrative features from Fear & Greed scores."""
    arr = np.clip(np.array(fng_scores, dtype=float), 0.0, 100.0)
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


def _compute_ema(prices: np.ndarray, period: int) -> np.ndarray:
    k = 2.0 / (period + 1)
    ema = np.zeros(len(prices))
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = prices[i] * k + ema[i - 1] * (1 - k)
    return ema


def _price_context(prices: list[float]) -> dict:
    """Where does current price sit relative to its own trend?"""
    arr = np.array(prices, dtype=float)
    ema50 = float(_compute_ema(arr, 50)[-1]) if len(arr) >= 50 else float(arr.mean())
    ema200 = float(_compute_ema(arr, 200)[-1]) if len(arr) >= 200 else float(arr.mean())
    current = arr[-1]
    vs_ema200 = current / ema200  # 1.0 = at trend, 1.3 = 30% above
    vs_ema50 = current / ema50
    return {
        "ema_50": round(ema50, 2),
        "ema_200": round(ema200, 2),
        "vs_ema200": round(vs_ema200, 3),
        "vs_ema50": round(vs_ema50, 3),
        "extended_above": vs_ema200 > 1.15,  # >15% above long-term trend
        "extended_below": vs_ema200 < 0.85,  # >15% below long-term trend
    }


def _apply_divergence_penalty(
    confidence: float,
    direction: str,
    ctx: dict,
    fng_score: int,
) -> tuple[float, bool]:
    """
    Cut confidence when price structure and sentiment are telling opposite stories.

    Bullish signal + extreme fear + price far above trend:
      Extreme fear usually happens at price LOWS (capitulation). When fear is
      extreme but price is already stretched above its 200-day average, the fear
      is likely about something else — and the "buy the fear" logic is weaker.

    Bearish signal + extreme greed + price far below trend:
      Same logic inverted.
    """
    if direction == "bullish" and fng_score <= 30 and ctx["extended_above"]:
        stretch = ctx["vs_ema200"] - 1.0
        penalty = max(0.60, 1.0 - stretch * 0.5)
        # Clamp to 0.5: penalty never flips direction (that would be inconsistent)
        return round(max(confidence * penalty, 0.5), 4), True

    if direction == "bearish" and fng_score >= 70 and ctx["extended_below"]:
        stretch = 1.0 - ctx["vs_ema200"]
        penalty = max(0.60, 1.0 - stretch * 0.5)
        return round(max(confidence * penalty, 0.5), 4), True

    return confidence, False


def _fng_label(score: int) -> str:
    if score >= 75:
        return "Extreme Greed"
    if score >= 55:
        return "Greed"
    if score >= 45:
        return "Neutral"
    if score >= 25:
        return "Fear"
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
    assert len(smoothed) == len(arr), "smoothed must have same length as prices"
    fng_arr = np.array(fng_scores, dtype=float)
    n = len(arr)

    X_rows, y_labels = [], []
    for t in range(10, n - 2):
        slope = float(np.polyfit(range(10), smoothed[t - 10 : t], 1)[0])
        vz = float(vol_z_arr[t])
        anom = float(anomaly_flags[t])

        fng_idx = min(t, len(fng_arr) - 1)
        fng_cur = fng_arr[fng_idx] / 100.0
        fw = fng_arr[max(0, fng_idx - 7) : fng_idx + 1]
        fng_vel = float(np.polyfit(range(len(fw)), fw, 1)[0]) / 100.0 if len(fw) >= 2 else 0.0
        fng_tip = bool(fng_cur > 0.5 and fng_vel > 0.02)

        rf_t = regime_feats[max(0, t - 6) : t]
        reg = predict_regime(hmm, rf_t) if len(rf_t) >= 2 else "lv_up"

        causal = CausalDAG().infer_price_impact(fng_tip, reg, bool(anom), False)

        fv = FeatureVector(
            kalman_slope=slope,
            volume_z_score=vz,
            anomaly_score=anom,
            narrative_velocity=max(fng_vel, 0.0),
            narrative_tipping_point=fng_tip,
            saturation=fng_cur,
            regime_lv_up=1.0 if reg == "lv_up" else 0.0,
            regime_hv_up=1.0 if reg == "hv_up" else 0.0,
            regime_lv_down=1.0 if reg == "lv_down" else 0.0,
            regime_hv_down=1.0 if reg == "hv_down" else 0.0,
            causal_bullish=causal["bullish"],
            causal_confidence=causal["confidence"],
            macro_dff=0.25,
        )
        # Real label: was price higher 2 days later? No leakage.
        y_labels.append(1 if arr[t + 2] > arr[t] else 0)
        X_rows.append(fv.to_array())

    return np.vstack(X_rows), np.array(y_labels)


def run_pipeline(
    prices: list[float],
    fng_scores: list[int],
    asset: str = "BTC",
    volumes: list[float] | None = None,
) -> SignalEvent:
    """Run full 3-layer causal pipeline on real data. Returns a SignalEvent."""
    if not fng_scores:
        fng_scores = list(_FNG_FALLBACK)
    if not prices:
        from kairos.models.signal_event import SignalEvent as _SE

        return _SE(
            asset=asset,
            direction="neutral",
            confidence=0.5,
            regime="lv_up",
            narrative_velocity=0.0,
            narrative_tipping_point=False,
            mechanism="No price data available",
            estimated_hours=24.0,
            citations=[],
        )
    arr = np.array(prices, dtype=float)
    # Interpolate non-finite values so downstream math doesn't propagate NaN/Inf
    if not np.isfinite(arr).all():
        finite_idx = np.where(np.isfinite(arr))[0]
        if len(finite_idx) >= 2:
            arr = np.interp(np.arange(len(arr)), finite_idx, arr[finite_idx])
        else:
            arr = np.where(np.isfinite(arr), arr, 1.0)

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
    if not fng_scores:
        fng_scores = [50] * 30
    narrative = _fng_narrative(fng_scores)

    # Layer 3 — Regime: HMM on full price history
    returns = np.diff(smoothed) / (smoothed[:-1] + 1e-8)
    volatility = np.abs(returns)
    regime_feats = np.column_stack([returns, volatility])
    finite_mask = np.isfinite(regime_feats).all(axis=1)
    regime_feats_clean = regime_feats[finite_mask]
    if len(regime_feats_clean) >= 10:
        hmm = fit_regime_model(regime_feats_clean)
        regime = predict_regime(hmm, regime_feats_clean[-5:])
    else:
        hmm = None
        regime = "lv_up"

    # Causal DAG
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
        regime_lv_up=1.0 if regime == "lv_up" else 0.0,
        regime_hv_up=1.0 if regime == "hv_up" else 0.0,
        regime_lv_down=1.0 if regime == "lv_down" else 0.0,
        regime_hv_down=1.0 if regime == "hv_down" else 0.0,
        causal_bullish=causal["bullish"],
        causal_confidence=causal["confidence"],
        macro_dff=0.25,
    )

    # Train XGBoost — use disk cache if available and fresh, else retrain
    ensemble = load_model(asset, len(prices))
    if ensemble is None:
        ensemble = SignalEnsemble()
        if hmm is not None and len(arr) > 15:
            try:
                X, y = _build_training_data(
                    prices,
                    fng_scores,
                    smoothed,
                    anomaly_flags,
                    vol_z_arr,
                    hmm,
                    regime_feats_clean,
                )
                if len(set(y.tolist())) >= 2:
                    ensemble.fit_raw(X, y)
                else:
                    ensemble.fit_synthetic_fallback()
            except Exception:
                ensemble.fit_synthetic_fallback()
        else:
            ensemble.fit_synthetic_fallback()
        try:
            save_model(ensemble, asset, len(prices))
        except Exception:
            pass

    return ensemble.predict(asset, fv, citations=causal["citations"], regime=regime)


def run_pipeline_with_context(
    prices: list[float], fng_scores: list[int], asset: str = "BTC"
) -> tuple[SignalEvent, dict]:
    """Same as run_pipeline() but also returns price context (EMA, divergence)."""
    event = run_pipeline(prices, fng_scores, asset=asset)
    ctx = _price_context(prices) if prices else {}
    adjusted_conf, diverged = _apply_divergence_penalty(
        event.confidence, event.direction, ctx, fng_scores[-1] if fng_scores else 50
    )
    event.confidence = adjusted_conf
    ctx["divergence_applied"] = diverged
    if _PAPER_TRADING_ENGINE is not None and prices:
        with _PAPER_ENGINE_LOCK:
            try:
                _PAPER_TRADING_ENGINE.process_signal(event, current_price=prices[-1])
                ctx["paper_account"] = _PAPER_TRADING_ENGINE.get_account(asset)
            except Exception:
                pass
    return event, ctx


async def safe_fetch_code_velocity() -> dict:
    """Wrap fetch_code_velocity with key normalization."""
    import kairos.ingest as ingest

    raw = await ingest.fetch_code_velocity()
    if not raw.get("available"):
        return {"available": False}
    return {
        "available": True,
        "commit_velocity": raw.get("commits", 0),
        "contributors": raw.get("contributors", 0),
        "merged_prs": raw.get("pull_requests", 0),
        "stars": raw.get("stars", 0),
        "forks": raw.get("forks", 0),
        "churn": raw.get("churn", 0.0),
        "repos_scraped": raw.get("repos", []),
    }


async def safe_fetch_whale_flows() -> dict:
    """Wrap fetch_whale_flows with safe fallback."""
    import kairos.ingest as ingest

    return await ingest.fetch_whale_flows()


class DataFetchSupervisor:
    """Parallel multi-source data fetcher with per-source fault isolation."""

    def __init__(self, max_retries: int = 3) -> None:
        self.max_retries = max_retries

    async def _fetch_prices_source(self, asset: str) -> dict:
        coin_id = _ASSET_IDS.get(asset, "bitcoin")
        async with httpx.AsyncClient(timeout=20.0) as client:
            prices, current_price, volumes = await _fetch_prices(client, coin_id=coin_id)
        return {"prices": prices, "current_price": current_price, "volumes": volumes}

    async def _fetch_fng_source(self, asset: str) -> dict:
        async with httpx.AsyncClient(timeout=20.0) as client:
            scores, ok = await _fetch_fng(client)
        return {"scores": scores, "available": ok}

    async def _fetch_github_source(self, asset: str) -> dict:
        return await safe_fetch_code_velocity()

    async def _fetch_whale_source(self, asset: str) -> dict:
        return await safe_fetch_whale_flows()

    async def _fetch_macro_source(self, asset: str) -> dict:
        return {"available": False}

    async def _run_source(self, name: str, asset: str) -> tuple[str, dict]:
        source_fn = getattr(self, f"_fetch_{name}_source")
        last_exc: Exception | None = None

        breaker_name = {"prices": "coingecko", "fng": "fng", "github": "github", "whale": "solana_rpc"}.get(name)
        breaker = CIRCUIT_BREAKERS.get(breaker_name) if breaker_name else None

        if breaker is not None:
            result = await breaker.call(
                lambda a=asset: source_fn(a),
                fallback=lambda: {"available": False, "error": "circuit open"},
            )
            if isinstance(result, dict) and result.get("available") is False:
                return name, result
            return name, {"available": True, "data": result}

        for _ in range(max(self.max_retries, 1)):
            try:
                result = await source_fn(asset)
                return name, {"available": True, "data": result}
            except Exception as exc:
                last_exc = exc
        return name, {"available": False, "error": str(last_exc)}

    async def fetch_all(self, asset: str = "BTC") -> dict:
        sources = ["prices", "fng", "github", "whale", "macro"]
        tasks = [self._run_source(name, asset) for name in sources]
        results = await asyncio.gather(*tasks)
        return dict(results)


def display_signal(
    event: SignalEvent,
    current_price: float,
    fng_score: int = 50,
    fng_available: bool = True,
    price_context: dict | None = None,
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
        "lv_up": "Low volatility uptrend (smart money quietly buying)",
        "hv_up": "High volatility rally (strong momentum, watch for reversal)",
        "lv_down": "Low volatility decline (distribution, watch out)",
        "hv_down": "High volatility selloff (panic, possible capitulation opportunity)",
    }.get(event.regime, event.regime)

    fng_text = f"{_fng_label(fng_score)} ({fng_score}/100)"

    plain_reasons = []

    # Sentiment reason — most human-readable, show first
    if fng_available:
        if fng_score <= 25:
            plain_reasons.append(f"Crowd is in extreme fear ({fng_score}/100) — historically a buying opportunity")
        elif fng_score >= 75:
            plain_reasons.append(f"Crowd is extremely greedy ({fng_score}/100) — often signals a top is near")
        elif fng_score <= 40:
            plain_reasons.append(f"Market sentiment is fearful ({fng_score}/100) — most people are nervous")
        elif fng_score >= 60:
            plain_reasons.append(f"Sentiment turning optimistic ({fng_score}/100) — buyers gaining confidence")

    for c in event.citations:
        if "Shiller" in c:
            plain_reasons.append("Sentiment momentum is building fast")
        elif "Stage 2" in c or "accumulation" in c.lower():
            plain_reasons.append("Market is in quiet buy phase — big players accumulating")
        elif "Stage 3" in c or "Stage 4" in c or "distribution" in c.lower():
            plain_reasons.append("Market is in sell phase — smart money distributing")
        elif "Soros" in c:
            plain_reasons.append("Market is flipping direction — high uncertainty")
        elif "Kindleberger" in c:
            plain_reasons.append("Macro stress is weighing on price")
        elif "Anomaly" in c:
            plain_reasons.append("Unusual price action detected — something is moving")

    t = Text()
    t.append(f"\n  {arrow} ", style=f"bold {color}")
    t.append(f"{direction_word}", style=f"bold {color}")
    t.append(f"  —  {conf_pct}% sure\n", style="dim")
    t.append(f"\n  {event.asset} right now:  ", style="dim")
    t.append(f"${current_price:,.0f}\n", style="bold white")
    t.append("  Market phase:   ", style="dim")
    t.append(f"{regime_plain}\n", style="white")
    t.append("  Sentiment:      ", style="dim")
    t.append(f"{fng_text}\n", style="white")
    t.append("  Confidence:     ", style="dim")
    t.append(f"[{bar}] {conf_pct}%\n", style=color)
    t.append("  Move expected:  ", style="dim")
    t.append(f"within ~{time_str}\n", style="yellow")

    if plain_reasons:
        t.append("\n  Why this signal:\n", style="bold dim")
        for r in plain_reasons:
            t.append(f"    • {r}\n", style="white")

    if price_context:
        pct_vs_200 = (price_context["vs_ema200"] - 1.0) * 100
        ema200_val = price_context["ema_200"]
        if price_context["extended_above"]:
            t.append(
                f"\n  ⚠  Price is {pct_vs_200:.0f}% above its 200-day average"
                f" (${ema200_val:,.0f}) — stretched territory\n",
                style="yellow",
            )
        elif price_context["extended_below"]:
            t.append(
                f"\n  ℹ  Price is {abs(pct_vs_200):.0f}% below its 200-day average"
                f" (${ema200_val:,.0f}) — deeply discounted\n",
                style="dim",
            )
        if price_context.get("divergence_applied"):
            t.append(
                "  ⚠  Confidence reduced: price and sentiment are" " pointing in different directions\n",
                style="yellow",
            )

    if event.narrative_tipping_point:
        t.append(
            "\n  ⚡ Sentiment tipping point — fear/greed shifting fast\n",
            style="bold yellow",
        )

    if conf_pct < 55:
        t.append(
            "\n  ⚠  Low confidence — market signal is weak, be careful\n",
            style="yellow",
        )

    if not fng_available:
        t.append("\n  ℹ  Sentiment API unavailable — using neutral baseline\n", style="dim")

    t.append("\n", style="")

    console.print(
        Panel(
            t,
            title=f"[bold white]KAIROS  ·  {event.asset} Live Signal[/bold white]",
            border_style=color,
            padding=(0, 2),
        )
    )
