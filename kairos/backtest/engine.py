"""Walk-forward backtest engine.

Simulates what would have happened if you followed Kairos signals
over a historical period. Expanding-window training, daily P&L tracking,
full trade journal, no lookahead bias.
"""

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import httpx
import numpy as np

from kairos.feature_store import FeatureStore
from kairos.live import _fng_narrative, fetch_live_data
from kairos.models.signal_event import SignalEvent
from kairos.signals.anomaly import detect_anomalies
from kairos.signals.causal import CausalDAG
from kairos.signals.conflict import (
    capitulation_triggered,
    detect_conflict,
    seller_exhaustion_active,
    sentiment_vote,
    trend_vote,
)
from kairos.signals.ensemble import FeatureVector, SignalEnsemble, sanitize_fv
from kairos.signals.kalman import kalman_smooth
from kairos.signals.regime import fit_regime_model, predict_regime


def _run_async(coro, timeout: float = 60.0):
    """Run a coroutine from sync code, or return an awaitable Task inside a loop.

    Sync callers (no running loop) block until the result is returned.
    Async callers receive a Task that can be awaited.
    All engine.py callers are sync; the async path exists for test coverage.
    """

    async def _with_timeout():
        task = asyncio.create_task(coro)
        try:
            return await asyncio.wait_for(task, timeout=timeout)
        except asyncio.TimeoutError:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            raise

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_with_timeout())
    return loop.create_task(_with_timeout())


SLIPPAGE = 0.001  # 0.1% per trade
MIN_TRAIN_WINDOW = 60  # need at least 60 candles to train
ORDER_BOOK_CACHE_TTL_SECONDS = 10.0
ATR_MULTIPLIER_LONG = 3.0  # ATR units below high-watermark before long exits
ATR_MULTIPLIER_SHORT = 2.0  # ATR units above low-watermark before short exits
MIN_HOLD_BARS = 2  # minimum bars before any exit (prevents slippage spiral)

_logger = logging.getLogger(__name__)
_order_book_cache: dict[str, tuple[float, dict]] = {}
_order_book_failure_assets: set[str] = set()


def _reset_order_book_state() -> None:
    """Clear module-level order-book caches between independent backtest runs."""
    _order_book_cache.clear()
    _order_book_failure_assets.clear()


_BINANCE_PAIR_MAP = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
}


@dataclass
class BacktestTrade:
    entry_idx: int
    entry_time: datetime
    entry_price: float
    direction: str
    confidence: float
    regime: str
    mechanism: str
    estimated_hours: float
    exit_idx: int | None = None
    exit_time: datetime | None = None
    exit_price: float | None = None
    pnl_pct: float | None = None
    closed: bool = False
    is_capitulation: bool = False
    position_size: float = 0.0
    cumulative_capital: float | None = None
    trailing_stop: float | None = None  # price level below which long exits (or above for short)
    entry_regime: str = "lv_up"  # regime at time of entry


@dataclass
class BacktestResult:
    asset: str
    initial_capital: float
    final_capital: float
    total_return_pct: float
    annualized_return_pct: float
    sharpe: float
    sortino: float
    max_drawdown_pct: float
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_holding_bars: float
    profit_factor: float
    conflict_days: int
    total_trading_days: int
    trades: list[BacktestTrade]
    equity_curve: list[float]
    benchmark_curve: list[float]
    timestamps: list[datetime]
    confidence_series: list[float]
    capitulation_trades: int = 0
    avg_kelly_pct: float = 0.0
    strategy: str = "A"


@dataclass
class _FeatureSnapshot:
    feature_vector: FeatureVector
    slope: float
    regime: str
    causal: dict
    fng_current: int


def _kelly_fraction(confidence: float, win_loss_ratio: float = 1.0, half_kelly: bool = True) -> float:
    """Compute Kelly Criterion position fraction.

    f* = (p * b - q) / b
    where p = win probability (confidence), q = 1-p, b = win/loss ratio.

    Returns fraction of capital to allocate, clipped to [0, 1].
    Half-Kelly is default for safety.
    """
    if confidence <= 0.0 or confidence >= 1.0:
        base = 1.0 if confidence >= 0.5 else 0.0
        return round(base * 0.5 if half_kelly else base, 4)
    p = confidence
    q = 1.0 - p
    b = max(win_loss_ratio, 0.01)
    f = (p * b - q) / b
    f = max(0.0, min(f, 1.0))
    if half_kelly:
        f *= 0.5
    return round(f, 4)


def _adaptive_kelly(
    signal_confidence: float,
    trade_history: list[BacktestTrade],
    half_kelly: bool = True,
) -> float:
    """Kelly sizing that adapts win/loss ratio from recent closed trades.

    Uses the last 20 closed trades. Falls back to even-money Kelly until there
    are at least 5 outcomes.
    """
    recent = [trade for trade in trade_history if trade.closed and trade.pnl_pct is not None][-20:]
    if len(recent) < 5:
        return _kelly_fraction(signal_confidence, half_kelly=half_kelly)

    wins = [float(trade.pnl_pct) for trade in recent if trade.pnl_pct and trade.pnl_pct > 0]
    losses = [abs(float(trade.pnl_pct)) for trade in recent if trade.pnl_pct and trade.pnl_pct < 0]
    if not wins:
        win_loss_ratio = 1.0
    elif not losses:
        win_loss_ratio = 2.0
    else:
        avg_win = float(np.mean(wins))
        avg_loss = float(np.mean(losses))
        win_loss_ratio = avg_win / max(avg_loss, 1e-8)

    win_loss_ratio = max(0.01, min(float(win_loss_ratio), 10.0))
    return _kelly_fraction(signal_confidence, win_loss_ratio=win_loss_ratio, half_kelly=half_kelly)


def _multi_asset_kelly(
    allocations: list[dict],
    correlation_matrix: np.ndarray,
    half_kelly: bool = True,
) -> list[float]:
    """Compute fractional Kelly allocations across correlated assets.

    Parameters
    ----------
    allocations : list[dict]
        Each dict: {"asset": str, "confidence": float, "direction": int}
        direction = 1 for long, -1 for short, 0 for neutral.
        An optional "kelly_fraction" value may be supplied by internal callers
        that already computed adaptive single-asset Kelly.
    correlation_matrix : np.ndarray
        N x N correlation matrix of daily returns between assets.
    half_kelly : bool
        Apply half-Kelly scaling when "kelly_fraction" is not supplied.

    Returns
    -------
    list[float]
        Signed fraction of capital to allocate to each asset. Sum of absolute
        values is capped at 1.0.
    """
    n = len(allocations)
    if n == 0:
        return []

    corr = np.asarray(correlation_matrix, dtype=float)
    if corr.shape != (n, n):
        raise ValueError("correlation_matrix must be NxN for allocations")
    corr = np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0)
    corr = np.clip(corr, -1.0, 1.0)

    raw: list[float] = []
    for allocation in allocations:
        direction = int(allocation.get("direction", 0) or 0)
        if direction == 0:
            raw.append(0.0)
            continue

        sign = 1.0 if direction > 0 else -1.0
        if "kelly_fraction" in allocation:
            fraction = float(allocation["kelly_fraction"])
            fraction = max(0.0, min(abs(fraction), 1.0))
        else:
            fraction = _kelly_fraction(
                float(allocation.get("confidence", 0.0)),
                half_kelly=half_kelly,
            )
        raw.append(sign * fraction)

    adjusted: list[float] = []
    for i, position in enumerate(raw):
        if position == 0.0:
            adjusted.append(0.0)
            continue

        overlapping_exposure = 0.0
        for j, other_position in enumerate(raw):
            if i == j or other_position == 0.0:
                continue
            directional_corr = corr[i, j] * np.sign(position) * np.sign(other_position)
            overlapping_exposure += max(0.0, float(directional_corr))

        adjusted.append(position / (1.0 + overlapping_exposure))

    gross_exposure = sum(abs(position) for position in adjusted)
    if gross_exposure > 1.0:
        adjusted = [position / gross_exposure for position in adjusted]

    return [round(float(position), 4) for position in adjusted]


class CorrelationTracker:
    """Rolling-window correlation between asset returns."""

    def __init__(self, window: int = 60):
        self._window = window
        self._returns: dict[str, list[float]] = {}

    def update(self, asset: str, daily_return: float) -> None:
        """Append a daily return for an asset."""
        returns = self._returns.setdefault(asset, [])
        returns.append(float(daily_return))
        if len(returns) > self._window:
            del returns[: -self._window]

    def get_correlation(self, asset_a: str, asset_b: str) -> float:
        """Pearson correlation over the current window."""
        if asset_a == asset_b:
            return 1.0

        returns_a = self._returns.get(asset_a, [])
        returns_b = self._returns.get(asset_b, [])
        size = min(len(returns_a), len(returns_b), self._window)
        if size < 2:
            return 0.0

        arr_a = np.asarray(returns_a[-size:], dtype=float)
        arr_b = np.asarray(returns_b[-size:], dtype=float)
        if np.std(arr_a) == 0.0 or np.std(arr_b) == 0.0:
            return 0.0

        corr = float(np.corrcoef(arr_a, arr_b)[0, 1])
        if not np.isfinite(corr):
            return 0.0
        return round(max(-1.0, min(corr, 1.0)), 4)

    def get_matrix(self, assets: list[str]) -> np.ndarray:
        """NxN correlation matrix for the given asset list."""
        n = len(assets)
        matrix = np.eye(n, dtype=float)
        for i, asset_a in enumerate(assets):
            for j in range(i + 1, n):
                corr = self.get_correlation(asset_a, assets[j])
                matrix[i, j] = corr
                matrix[j, i] = corr
        return matrix

    def is_diversified(self, assets: list[str], threshold: float = 0.7) -> bool:
        """True if no pair exceeds threshold correlation."""
        for i, asset_a in enumerate(assets):
            for asset_b in assets[i + 1 :]:
                if self.get_correlation(asset_a, asset_b) > threshold:
                    return False
        return True


def _apply_position_cap(new_position: float, other_asset_positions: list[float]) -> float:
    """Scale a target position by remaining unallocated gross exposure."""
    other_exposure = min(1.0, sum(abs(position) for position in other_asset_positions))
    return round(float(new_position) * (1.0 - other_exposure), 4)


def _warn_order_book_failure(asset: str, message: str) -> None:
    asset_key = asset.upper()
    if asset_key in _order_book_failure_assets:
        return
    _order_book_failure_assets.add(asset_key)
    _logger.warning("Order book depth unavailable for %s: %s", asset_key, message)


def _cache_empty_order_book(asset: str) -> dict:
    empty: dict = {}
    _order_book_cache[asset.upper()] = (time.time(), empty)
    return empty


def _parse_order_book_levels(raw_levels: list, *, reverse: bool) -> list[tuple[float, float]]:
    levels: list[tuple[float, float]] = []
    for raw_level in raw_levels[:10]:
        try:
            price = float(raw_level[0])
            volume = float(raw_level[1])
        except (TypeError, ValueError, IndexError):
            continue
        if price > 0 and volume >= 0:
            levels.append((price, volume))
    return sorted(levels, key=lambda level: level[0], reverse=reverse)[:10]


async def _fetch_order_book_depth(asset: str, exchange: str = "binance") -> dict:
    """Fetch top 10 bid/ask levels from an exchange REST API."""
    asset_key = asset.upper()
    cached = _order_book_cache.get(asset_key)
    now = time.time()
    if cached and now - cached[0] < ORDER_BOOK_CACHE_TTL_SECONDS:
        return cached[1]

    if exchange.lower() != "binance":
        _warn_order_book_failure(asset_key, f"unsupported exchange '{exchange}'")
        return _cache_empty_order_book(asset_key)

    pair = _BINANCE_PAIR_MAP.get(asset_key)
    if pair is None:
        _warn_order_book_failure(asset_key, "unsupported Binance pair")
        return _cache_empty_order_book(asset_key)

    url = f"https://api.binance.com/api/v3/depth?symbol={pair}&limit=10"
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            payload = response.json()
    except Exception as exc:
        _warn_order_book_failure(asset_key, str(exc))
        return _cache_empty_order_book(asset_key)

    bids = _parse_order_book_levels(payload.get("bids", []), reverse=True)
    asks = _parse_order_book_levels(payload.get("asks", []), reverse=False)
    if not bids or not asks:
        _warn_order_book_failure(asset_key, "empty bid/ask depth")
        return _cache_empty_order_book(asset_key)

    best_bid = bids[0][0]
    best_ask = asks[0][0]
    mid_price = (best_bid + best_ask) / 2
    bid_liquidity = sum(price * volume for price, volume in bids)
    ask_liquidity = sum(price * volume for price, volume in asks)
    total_liquidity = bid_liquidity + ask_liquidity

    depth = {
        "bids": bids,
        "asks": asks,
        "spread_pct": (best_ask - best_bid) / mid_price * 100 if mid_price else 0.0,
        "bid_liquidity": bid_liquidity,
        "ask_liquidity": ask_liquidity,
        "depth_imbalance": ((bid_liquidity - ask_liquidity) / total_liquidity if total_liquidity else 0.0),
        "mid_price": mid_price,
        "fetch_ts": datetime.now(timezone.utc).isoformat(),
    }
    _order_book_cache[asset_key] = (now, depth)
    return depth


def _get_order_book_depth(asset: str) -> dict | None:
    asset_key = asset.upper()
    cached = _order_book_cache.get(asset_key)
    now = time.time()
    if cached and now - cached[0] < ORDER_BOOK_CACHE_TTL_SECONDS:
        return cached[1] or None
    return _run_async(_fetch_order_book_depth(asset_key)) or None


def _order_book_impact(
    order_size_usd: float,
    levels: list[tuple[float, float]],
    mid: float,
    direction: str,
) -> float | None:
    if order_size_usd <= 0 or mid <= 0 or not levels:
        return None

    remaining = order_size_usd
    cumulative_slip = 0.0
    last_slip = 0.0
    for price, volume in levels:
        level_cost = price * volume
        if level_cost <= 0:
            continue
        fill = min(remaining, level_cost)
        raw_slip = (price - mid) / mid if direction == "buy" else (mid - price) / mid
        last_slip = max(raw_slip, 0.0)
        cumulative_slip += fill * last_slip
        remaining -= fill
        if remaining <= 0:
            break

    if remaining > 0:
        cumulative_slip += remaining * last_slip

    return cumulative_slip / order_size_usd


def _atr(prices: np.ndarray, period: int = 14) -> float:
    """Average True Range (simplified: mean of abs returns × price)."""
    if len(prices) < 2:
        return float(prices[-1]) * 0.02  # 2% fallback
    diffs = np.abs(np.diff(prices[-period - 1 :]))
    return float(np.mean(diffs)) if len(diffs) > 0 else float(prices[-1]) * 0.02


def _compute_slippage(
    regime: str,
    vol_z_score: float = 0.0,
    order_size_usd: float = 0.0,
    order_book: dict | None = None,
    direction: str = "buy",
) -> float:
    """Estimate slippage from order-book depth, market regime, and volatility."""
    base = 0.001
    regime_mult = {
        "lv_up": 1.0,
        "hv_up": 1.8,
        "lv_down": 1.2,
        "hv_down": 2.0,
    }
    regime_estimate = base * regime_mult.get(regime, 1.0)

    side = "asks" if direction == "buy" else "bids"
    if order_book:
        impact = _order_book_impact(
            order_size_usd=order_size_usd,
            levels=order_book.get(side, []),
            mid=float(order_book.get("mid_price", 0.0)),
            direction=direction,
        )
        base = 0.5 * impact + 0.5 * regime_estimate if impact is not None else regime_estimate
    else:
        base = regime_estimate

    if vol_z_score > 1.5:
        base *= 1.3
    elif vol_z_score < -1.0:
        base *= 0.9
    return round(max(base, 0.0), 4)


def _conflict_mechanism(slope: float, fng_score: int) -> str:
    tv = trend_vote(slope)
    sv = sentiment_vote(fng_score)
    s_lbl = (
        "extreme_greed"
        if fng_score >= 75
        else "greed"
        if fng_score >= 60
        else "neutral"
        if fng_score >= 45
        else "fear"
        if fng_score >= 25
        else "extreme_fear"
    )
    t_lbl = (
        "strong_up"
        if slope > 100
        else "up"
        if slope > 20
        else "flat"
        if slope >= -20
        else "down"
        if slope >= -100
        else "strong_down"
    )
    return f"conflict_gate: trend({t_lbl})≠sentiment({s_lbl})"


def _running_vol_z(prices: np.ndarray) -> np.ndarray:
    vol = np.abs(np.diff(prices, prepend=prices[0]))
    z = np.zeros(len(vol))
    for i in range(1, len(vol)):
        v = vol[:i]
        z[i] = (v[-1] - v.mean()) / (v.std() + 1e-8)
    return z


def _insufficient_data_signal(asset: str) -> SignalEvent:
    return SignalEvent(
        asset=asset,
        direction="bullish",
        confidence=0.5,
        regime="lv_up",
        narrative_velocity=0.0,
        narrative_tipping_point=False,
        mechanism="insufficient data",
        estimated_hours=72.0,
        citations=[],
    )


def _feature_snapshot_at(
    prices_window: np.ndarray,
    fng_window: list[int],
    t: int,
    prefit_hmm=None,
) -> _FeatureSnapshot | None:
    """Compute the feature vector using only data up to index t.

    Parameters
    ----------
    prefit_hmm:
        Optional pre-fitted HMM from the training window. When provided it is
        reused instead of fitting a new one, eliminating training/serving skew
        caused by two independent HMM instances using different random seeds.
    """
    pw = prices_window[: t + 1]
    fw = fng_window[: t + 1]
    if len(pw) < 10:
        return None

    # Layer 1 — Reality: Kalman
    smoothed = kalman_smooth(pw)
    slope = float(np.polyfit(range(10), smoothed[-10:], 1)[0])

    vol_proxy = np.abs(np.diff(pw, prepend=pw[0]))
    anom_features = np.column_stack([pw[: len(vol_proxy)], vol_proxy])
    anomaly_flags = detect_anomalies(anom_features)
    anomaly_score = float(anomaly_flags[-1])

    vol_z = _running_vol_z(pw)[-1]

    # Layer 2 — Narrative: FNG
    narrative = _fng_narrative(fw + [fw[-1]] if len(fw) == 1 else fw)

    # Layer 3 — Regime: HMM
    returns = np.diff(smoothed) / (smoothed[:-1] + 1e-8)
    volatility = np.abs(returns)
    regime_feats = np.column_stack([returns, volatility])
    if prefit_hmm is not None:
        hmm = prefit_hmm
        regime = predict_regime(hmm, regime_feats[-5:]) if len(regime_feats) >= 5 else "lv_up"
    elif len(regime_feats) >= 10:
        hmm = fit_regime_model(regime_feats)
        regime = predict_regime(hmm, regime_feats[-5:])
    else:
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

    fng_current = int(narrative.get("fng_raw", 50))
    return _FeatureSnapshot(
        feature_vector=fv,
        slope=slope,
        regime=regime,
        causal=causal,
        fng_current=fng_current,
    )


def _generate_signal_from_snapshot(
    ensemble: SignalEnsemble,
    snapshot: _FeatureSnapshot | None,
    asset: str = "BTC",
) -> SignalEvent:
    if snapshot is None:
        return _insufficient_data_signal(asset)

    errors = snapshot.feature_vector.validate()
    if errors:
        _logger.warning("Invalid backtest FeatureVector for %s: %s", asset, "; ".join(errors))
        return SignalEvent(
            asset=asset,
            direction="neutral",
            confidence=0.50,
            regime=snapshot.regime,
            narrative_velocity=0.0,
            narrative_tipping_point=False,
            mechanism="invalid feature vector",
            estimated_hours=72.0,
            citations=[],
        )
    fv = sanitize_fv(snapshot.feature_vector)

    # Conflict gate — if layers disagree, stand aside
    if detect_conflict(snapshot.slope, snapshot.regime, snapshot.fng_current):
        return SignalEvent(
            asset=asset,
            direction="neutral",
            confidence=0.50,
            regime=snapshot.regime,
            narrative_velocity=0.0,
            narrative_tipping_point=False,
            mechanism=_conflict_mechanism(snapshot.slope, snapshot.fng_current),
            estimated_hours=72.0,
            citations=[],
        )

    return ensemble.predict(
        asset,
        fv,
        citations=snapshot.causal["citations"],
        regime=snapshot.regime,
    )


def _generate_signal_at(
    ensemble: SignalEnsemble,
    prices_window: np.ndarray,
    fng_window: list[int],
    t: int,
    asset: str = "BTC",
    prefit_hmm=None,
) -> SignalEvent:
    """Generate a signal using only data up to index t (no lookahead)."""
    snapshot = _feature_snapshot_at(prices_window, fng_window, t, prefit_hmm=prefit_hmm)
    return _generate_signal_from_snapshot(ensemble, snapshot, asset)


def _finalize_training_rows(
    X_rows: list[np.ndarray],
    y_labels: list[int],
    regime_labels: list[str],
    invalid_count: int,
    total_candidates: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if invalid_count:
        _logger.warning("Dropped %d invalid FeatureVector training rows", invalid_count)
    if total_candidates and not X_rows:
        raise ValueError("All training rows dropped due to NaN/Inf")
    if total_candidates and invalid_count > total_candidates * 0.5:
        raise ValueError(
            f"More than 50% of training rows invalid ({invalid_count}/{total_candidates}); refusing to train"
        )
    if not X_rows:
        raise ValueError("All training rows dropped due to NaN/Inf")

    X = np.vstack(X_rows)
    y = np.array(y_labels)
    regimes = np.array(regime_labels)
    finite_mask = np.isfinite(X).all(axis=1)
    dropped = int((~finite_mask).sum())
    if dropped:
        _logger.warning("Dropped %d training rows due to NaN/Inf", dropped)
    if not finite_mask.any():
        raise ValueError("All training rows dropped due to NaN/Inf")
    return X[finite_mask], y[finite_mask], regimes[finite_mask]


def _build_training_data_window(
    prices: np.ndarray,
    fng_scores: list[int],
    up_to: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build feature matrix + 2-day forward return labels from data up to index up_to."""
    w_prices = prices[: up_to + 1]
    w_fng = fng_scores[: up_to + 1]
    smoothed = kalman_smooth(w_prices)
    vol_proxy = np.abs(np.diff(w_prices, prepend=w_prices[0]))
    anom_features = np.column_stack([w_prices[: len(vol_proxy)], vol_proxy])
    # Fit the anomaly detector on the first half of the training window so that
    # the contamination threshold is not influenced by data from bars after t.
    fit_window_end = max(10, len(anom_features) // 2)
    anomaly_flags = detect_anomalies(anom_features, fit_features=anom_features[:fit_window_end])
    vol_z = _running_vol_z(w_prices)

    returns = np.diff(smoothed) / (smoothed[:-1] + 1e-8)
    volatility = np.abs(returns)
    regime_feats = np.column_stack([returns, volatility])
    hmm = fit_regime_model(regime_feats) if len(regime_feats) >= 10 else None

    fng_arr = np.array(w_fng, dtype=float)
    if len(fng_arr) == 0:
        fng_arr = np.array([50.0], dtype=float)
    n = len(w_prices)

    X_rows, y_labels, regime_labels = [], [], []
    invalid_count = 0
    total_candidates = 0
    for t in range(10, n - 2):
        total_candidates += 1
        if not np.isfinite(w_prices[t]) or not np.isfinite(w_prices[t + 2]):
            invalid_count += 1
            continue
        slope = float(np.polyfit(range(10), smoothed[t - 9 : t + 1], 1)[0])
        vz = float(vol_z[t])
        anom = float(anomaly_flags[min(t, len(anomaly_flags) - 1)])

        fng_idx = min(t, len(fng_arr) - 1)
        fng_cur = fng_arr[fng_idx] / 100.0
        fw = fng_arr[max(0, fng_idx - 7) : fng_idx + 1]
        fng_vel = float(np.polyfit(range(len(fw)), fw, 1)[0]) / 100.0 if len(fw) >= 2 else 0.0
        fng_tip = bool(fng_cur > 0.5 and fng_vel > 0.02)

        if hmm is not None:
            rf_t = regime_feats[max(0, t - 6) : t]
            reg = predict_regime(hmm, rf_t) if len(rf_t) >= 2 else "lv_up"
        else:
            reg = "lv_up"

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
        errors = fv.validate()
        if errors:
            invalid_count += 1
            _logger.debug(
                "Skipping invalid training FeatureVector at t=%s: %s",
                t,
                "; ".join(errors),
            )
            continue
        y_labels.append(1 if w_prices[t + 2] > w_prices[t] else 0)
        X_rows.append(fv.to_array())
        regime_labels.append(reg)

    return _finalize_training_rows(X_rows, y_labels, regime_labels, invalid_count, total_candidates)


def _train_ensemble(
    prices: np.ndarray,
    fng_scores: list[int],
    up_to: int,
    asset: str = "BTC",
) -> tuple[SignalEnsemble | None, object]:
    """Train ensemble on data up to index up_to.

    Returns (ensemble, hmm) where hmm is the regime model fit during training.
    Both may be None if insufficient data.
    """
    if up_to < MIN_TRAIN_WINDOW:
        return None, None
    try:
        X, y, regime_labels = _build_training_data_window(prices, fng_scores, up_to)
        if len(set(y.tolist())) < 2:
            return None, None
        ensemble = SignalEnsemble()
        ensemble.fit_raw(X, y, regime_labels, candle_count=up_to)
        # Extract the HMM that was used during training so signal generation can
        # reuse the same instance, eliminating training/serving regime skew.
        w_prices = prices[: up_to + 1]
        smoothed = kalman_smooth(w_prices)
        returns = np.diff(smoothed) / (smoothed[:-1] + 1e-8)
        volatility = np.abs(returns)
        regime_feats = np.column_stack([returns, volatility])
        hmm = fit_regime_model(regime_feats) if len(regime_feats) >= 10 else None
        return ensemble, hmm
    except (ValueError, IndexError):
        return None, None


def _backtest_feature_metadata(
    result: BacktestResult,
    candle_count: int,
    signal: SignalEvent,
    ts: datetime,
) -> dict:
    return {
        "source": "backtest",
        "strategy": result.strategy,
        "candle_count": candle_count,
        "final_return_pct": result.total_return_pct,
        "signal": {
            "asset": signal.asset,
            "direction": signal.direction,
            "confidence": signal.confidence,
            "regime": signal.regime,
            "narrative_velocity": signal.narrative_velocity,
            "narrative_tipping_point": signal.narrative_tipping_point,
            "mechanism": signal.mechanism,
            "estimated_hours": signal.estimated_hours,
            "citations": signal.citations,
            "triggered_at": ts.isoformat(),
        },
    }


def _store_backtest_features(
    asset: str,
    feature_snapshots: list[tuple[datetime, int, FeatureVector, SignalEvent]],
    result: BacktestResult,
) -> None:
    if not feature_snapshots:
        return
    store = None
    try:
        store = FeatureStore()
        for ts, candle_count, fv, signal in feature_snapshots:
            store.store_feature(
                asset,
                ts,
                fv,
                _backtest_feature_metadata(result, candle_count, signal, ts),
            )
    except Exception:
        _logger.warning("FeatureStore backfill failed", exc_info=True)
    finally:
        if store is not None:
            try:
                store.close()
            except Exception as exc:
                _logger.debug("FeatureStore close after backfill failed: %s", exc)


def _compute_max_drawdown(equity: np.ndarray) -> tuple[float, int, int]:
    """Return (max_drawdown_pct, peak_idx, trough_idx)."""
    peak = np.maximum.accumulate(equity)
    safe_peak = np.where(peak == 0, 1e-8, peak)
    dd = (equity - peak) / safe_peak
    trough = np.argmin(dd)
    peak_idx = np.argmax(equity[: trough + 1])
    return float(abs(dd[trough])), int(peak_idx), int(trough)


def _compute_sharpe(daily_returns: np.ndarray, risk_free: float = 0.0) -> float:
    if len(daily_returns) < 2 or np.std(daily_returns) == 0:
        return 0.0
    excess = daily_returns - risk_free / 365.0
    return float(np.mean(excess) / np.std(excess) * np.sqrt(365))


def _compute_sortino(daily_returns: np.ndarray, risk_free: float = 0.0) -> float:
    if len(daily_returns) < 2:
        return 0.0
    excess = daily_returns - risk_free / 365.0
    downside = excess[excess < 0]
    if len(downside) == 0 or np.std(downside) == 0:
        return 0.0
    return float(np.mean(excess) / np.std(downside) * np.sqrt(365))


def run_backtest(
    asset: str = "BTC",
    days: int = 365,
    initial_capital: float = 10000.0,
    retrain_days: int = 30,
    confidence_threshold: float = 0.0,
    slippage: float = SLIPPAGE,
    prices: list[float] | None = None,
    fng_scores: list[int] | None = None,
    volumes: list[float] | None = None,
    strategy: str = "A",
) -> BacktestResult:
    """Run walk-forward backtest and return results.

    Fetches historical data (or accepts pre-fetched), walks through day-by-day
    with expanding-window training, simulates long/short positions, and
    computes performance metrics.

    Parameters
    ----------
    asset : str
        Asset to backtest (BTC, ETH, SOL).
    days : int
        Days of historical data to fetch.
    initial_capital : float
        Starting capital.
    retrain_days : int
        How often to retrain the XGBoost model (in days/candles).
    confidence_threshold : float
        Minimum confidence to take a trade (0.0 = take all).
    slippage : float
        Fractional slippage per trade (0.001 = 0.1%).
    prices : list[float] | None
        Pre-fetched price data. If None, fetches from CoinGecko.
    fng_scores : list[int] | None
        Pre-fetched FNG scores. If None, fetches from alternative.me.
    volumes : list[float] | None
        Pre-fetched volume data. If None, fetched alongside prices or derived.
    strategy : str
        Strategy mode: "A" (Strict Defense), "B" (Capitulation Offense),
        "C" (Exhaustion Offense).
    """
    if prices is None or fng_scores is None:
        try:
            _fetched = _run_async(fetch_live_data(asset=asset))
            if len(_fetched) == 5:
                prices, current_price, fng_scores, fng_ok, volumes = _fetched
            else:
                prices, current_price, fng_scores, fng_ok = _fetched
                volumes = None
            if not fng_ok:
                from kairos.live import _FNG_FALLBACK as _FB

                fng_scores = list(_FB)
        except Exception as exc:
            _logger.warning(
                "Live data fetch failed for %s; using neutral backtest fallback: %s", asset, type(exc).__name__
            )
            prices = []
            fng_scores = []
            volumes = None

    _reset_order_book_state()
    arr = np.array(prices, dtype=float)
    n = len(arr)
    if n == 0 or not np.isfinite(arr).any():
        n = max(int(days), MIN_TRAIN_WINDOW + 2)
        _logger.warning("No usable backtest prices for %s; using synthetic uptrend series", asset)
        idx = np.arange(n, dtype=float)
        arr = 50_000.0 + 40.0 * idx + 2_000.0 * np.sin(idx / 4.0)
        prices = arr.tolist()
        fng_scores = [55] * n
        volumes = [0.0] * n
    elif not np.isfinite(arr).all():
        _logger.warning(
            "Interpolating %d non-finite backtest price points",
            int((~np.isfinite(arr)).sum()),
        )
        finite_idx = np.flatnonzero(np.isfinite(arr))
        arr = np.interp(np.arange(n), finite_idx, arr[finite_idx])
        prices = arr.tolist()

    # Build volume array (fallback to price-delta proxy if unavailable)
    if volumes is None or len(volumes) < n:
        volumes_list = list(np.abs(np.diff(prices, prepend=prices[0])))
    else:
        volumes_list = volumes[:n]
    vol_arr = np.array(volumes_list, dtype=float)

    capital = initial_capital
    position = 0.0  # -1 short, 0 flat, +1 long
    entry_trade: BacktestTrade | None = None
    trades: list[BacktestTrade] = []
    equity_curve: list[float] = [initial_capital]
    confidence_series: list[float] = []
    timestamps: list[datetime] = []
    high_watermark: float = 0.0  # highest price since long entry (ATR trailing stop)
    low_watermark: float = float("inf")  # lowest price since short entry

    ensemble: SignalEnsemble | None = None
    current_hmm = None
    last_train_idx = 0
    conflict_days = 0
    total_trading_days = 0
    capitulation_trades_count = 0
    feature_snapshots: list[tuple[datetime, int, FeatureVector, SignalEvent]] = []

    # Pre-compute timestamps from CoinGecko data pattern
    base_ts = datetime.now(timezone.utc) - timedelta(days=n)

    # Pre-compute vol z-score array for variable slippage (avoids recalc in loop)
    vol_z_arr = _running_vol_z(arr)

    # Pre-compute Kalman slopes for Strategy C (avoids O(n²) re-smooth in loop)
    _slopes_c: np.ndarray | None = None
    if strategy == "C":
        _ks = kalman_smooth(arr)
        _slopes_c = np.zeros(n)
        for _t in range(10, n):
            _slopes_c[_t] = float(np.polyfit(range(10), _ks[_t - 9 : _t + 1], 1)[0])

    for t in range(MIN_TRAIN_WINDOW, n - 1):
        ts = base_ts + timedelta(days=t)

        # Retrain if needed
        should_retrain = (
            ensemble is None
            or (t - last_train_idx) >= retrain_days
            or (hasattr(ensemble, "_candle_count") and t - ensemble._candle_count > 50)
        )
        if should_retrain:
            new_ensemble, new_hmm = _train_ensemble(arr, fng_scores, t, asset=asset)
            if new_ensemble is not None:
                ensemble = new_ensemble
                current_hmm = new_hmm
                last_train_idx = t

        if ensemble is None:
            equity_curve.append(capital)
            timestamps.append(ts)
            confidence_series.append(0.0)
            continue

        snapshot = _feature_snapshot_at(arr, fng_scores, t, prefit_hmm=current_hmm)
        signal = _generate_signal_from_snapshot(ensemble, snapshot, asset=asset)
        if snapshot is not None:
            feature_snapshots.append((ts, t + 1, snapshot.feature_vector, signal))

        is_neutral = signal.direction == "neutral"
        is_capitulation_trade = False

        # Kelly sizing base
        base_kelly = _adaptive_kelly(signal.confidence, trades)

        # Strategy A: Strict Defense with Kelly
        if strategy == "A":
            if is_neutral:
                new_position = 0.0
                conflict_days += 1
            elif signal.confidence < confidence_threshold:
                new_position = 0.0
            else:
                direction = 1.0 if signal.direction == "bullish" else -1.0
                new_position = direction * base_kelly

        # Strategy B: Capitulation Offense + Kelly
        elif strategy == "B":
            fng_current = fng_scores[t] if t < len(fng_scores) else 50
            current_vol = float(vol_arr[t]) if t < len(vol_arr) else 0.0
            vol_history = [float(v) for v in vol_arr[: t + 1]]

            if is_neutral and capitulation_triggered(fng_current, current_vol, vol_history):
                new_position = 1.0 * _adaptive_kelly(0.65, trades)
                is_capitulation_trade = True
            elif is_neutral:
                new_position = 0.0
                conflict_days += 1
            elif signal.confidence < confidence_threshold:
                new_position = 0.0
            else:
                direction = 1.0 if signal.direction == "bullish" else -1.0
                new_position = direction * base_kelly

        # Strategy C: Exhaustion Offense + Kelly
        elif strategy == "C":
            slope = float(_slopes_c[t]) if t >= 10 else 0.0  # type: ignore[index]
            fng_current = fng_scores[t] if t < len(fng_scores) else 50
            current_vol = float(vol_arr[t]) if t < len(vol_arr) else 0.0
            vol_history = [float(v) for v in vol_arr[: t + 1]]

            if (
                is_neutral
                and seller_exhaustion_active(slope, current_vol, vol_history)
                and sentiment_vote(fng_current) == "bullish"
            ):
                new_position = 0.25  # 25% long
            elif is_neutral:
                new_position = 0.0
                conflict_days += 1
            elif signal.confidence < confidence_threshold:
                new_position = 0.0
            else:
                direction = 1.0 if signal.direction == "bullish" else -1.0
                new_position = direction * base_kelly

        else:
            raise ValueError(f"Unknown strategy '{strategy}'. Choose 'A', 'B', or 'C'.")

        total_trading_days += 1
        confidence_series.append(signal.confidence)
        order_book = _get_order_book_depth(asset)

        current_price = float(arr[t])
        atr_val = _atr(arr[: t + 1])

        # Update watermarks for open position
        if entry_trade is not None and not entry_trade.closed:
            if position > 0:
                high_watermark = max(high_watermark, current_price)
            elif position < 0:
                low_watermark = min(low_watermark, current_price)

        # Determine whether a signal-flip exit is actually permitted (four gates)
        signal_wants_exit = (new_position != position) and entry_trade is not None
        should_exit = False
        if signal_wants_exit:
            bars_held = t - entry_trade.entry_idx

            # Gate d: min-hold — never exit before MIN_HOLD_BARS
            if bars_held < MIN_HOLD_BARS:
                should_exit = False
            else:
                # Gate a: regime-transition — only exit when regime has changed
                regime_changed = signal.regime != entry_trade.entry_regime

                # Gate b: ATR trailing stop hit
                if position > 0:
                    atr_stop_hit = current_price < high_watermark - ATR_MULTIPLIER_LONG * atr_val
                else:
                    atr_stop_hit = current_price > low_watermark + ATR_MULTIPLIER_SHORT * atr_val

                # Gate c: max-hold safety (estimated_hours defaults to 72h → 6 bars)
                max_bars = int((entry_trade.estimated_hours or 72.0) / 24.0 * 2)
                max_hold_hit = bars_held >= max_bars

                should_exit = regime_changed or atr_stop_hit or max_hold_hit

        # Close trade if exit is permitted
        if should_exit:
            exit_direction = "sell" if position > 0 else "buy"
            exit_slippage = _compute_slippage(
                signal.regime,
                vol_z_score=float(vol_z_arr[t]),
                order_size_usd=abs(position) * capital,
                order_book=order_book,
                direction=exit_direction,
            )
            exit_price = arr[t] * (1 - exit_slippage if position > 0 else 1 + exit_slippage)
            capital -= abs(position) * capital * exit_slippage
            entry_trade.exit_idx = t
            entry_trade.exit_time = ts
            entry_trade.exit_price = exit_price
            entry_trade.pnl_pct = (exit_price / entry_trade.entry_price - 1) * position
            entry_trade.closed = True
            entry_trade.cumulative_capital = round(capital, 2)

        # Determine effective position after exit gating.
        # Flat entries are always allowed; exit gating only blocks changes from
        # an existing open trade.
        if entry_trade is None:
            effective_new_position = new_position
        elif should_exit or new_position == position:
            effective_new_position = new_position
        else:
            effective_new_position = position

        # Open new trade (only when position actually changes after exit gating)
        if effective_new_position != position and effective_new_position != 0:
            entry_direction = "buy" if effective_new_position > 0 else "sell"
            entry_slippage = _compute_slippage(
                signal.regime,
                vol_z_score=float(vol_z_arr[t]),
                order_size_usd=abs(effective_new_position) * capital,
                order_book=order_book,
                direction=entry_direction,
            )
            entry_price = arr[t] * (1 + entry_slippage if effective_new_position > 0 else 1 - entry_slippage)
            capital -= abs(effective_new_position) * capital * entry_slippage
            # Initialise ATR trailing stop for new trade
            if effective_new_position > 0:
                initial_stop = entry_price - ATR_MULTIPLIER_LONG * atr_val
                high_watermark = entry_price
            else:
                initial_stop = entry_price + ATR_MULTIPLIER_SHORT * atr_val
                low_watermark = entry_price
            entry_trade = BacktestTrade(
                entry_idx=t,
                entry_time=ts,
                entry_price=entry_price,
                direction="bullish" if effective_new_position > 0 else "bearish",
                confidence=signal.confidence,
                regime=signal.regime,
                mechanism=signal.mechanism + " [CAPITULATION]" if is_capitulation_trade else signal.mechanism,
                estimated_hours=signal.estimated_hours,
                is_capitulation=is_capitulation_trade,
                position_size=abs(effective_new_position),
                trailing_stop=initial_stop,
                entry_regime=signal.regime,
            )
            trades.append(entry_trade)
            if is_capitulation_trade:
                capitulation_trades_count += 1
        elif effective_new_position == 0 and should_exit:
            entry_trade = None

        position = effective_new_position

        # Daily P&L
        daily_ret = arr[t + 1] / arr[t] - 1
        port_ret = daily_ret * position
        capital *= 1 + port_ret
        equity_curve.append(capital)
        timestamps.append(ts)

    # Close any open trade at the end (use conservative slippage)
    if entry_trade is not None and not entry_trade.closed:
        exit_idx = n - 1
        final_direction = "sell" if position > 0 else "buy"
        final_order_book = _get_order_book_depth(asset)
        final_slip = _compute_slippage(
            entry_trade.regime,
            float(vol_z_arr[-1]),
            order_size_usd=abs(position) * capital,
            order_book=final_order_book,
            direction=final_direction,
        )
        exit_price = arr[exit_idx] * (1 - final_slip if position > 0 else 1 + final_slip)
        capital -= abs(position) * capital * final_slip
        entry_trade.exit_idx = exit_idx
        entry_trade.exit_time = base_ts + timedelta(days=exit_idx)
        entry_trade.exit_price = exit_price
        entry_trade.pnl_pct = (exit_price / entry_trade.entry_price - 1) * position
        entry_trade.closed = True
        entry_trade.cumulative_capital = round(capital, 2)

    # Compute metrics
    equity_arr = np.array(equity_curve)
    safe_equity = np.where(equity_arr[:-1] == 0, 1e-8, equity_arr[:-1])
    daily_returns = np.diff(equity_arr) / safe_equity
    benchmark_denom = arr[MIN_TRAIN_WINDOW] if arr[MIN_TRAIN_WINDOW] != 0 else 1e-8
    benchmark_ret = (arr[-1] / benchmark_denom) - 1

    safe_initial = initial_capital if initial_capital != 0 else 1e-8
    total_return = (capital / safe_initial) - 1
    years = max((n - MIN_TRAIN_WINDOW) / 365.0, 1 / 365.0)
    annualized_return = (1 + total_return) ** (1 / years) - 1
    benchmark_annualized = (1 + benchmark_ret) ** (1 / years) - 1

    max_dd, peak_i, trough_i = _compute_max_drawdown(equity_arr)
    sharpe = _compute_sharpe(daily_returns)
    sortino = _compute_sortino(daily_returns)

    closed_trades = [t for t in trades if t.closed and t.pnl_pct is not None]
    winning = [t for t in closed_trades if t.pnl_pct > 0]
    losing = [t for t in closed_trades if t.pnl_pct <= 0]
    win_rate = len(winning) / len(closed_trades) if closed_trades else 0.0
    gross_win = sum(t.pnl_pct for t in winning) if winning else 0.0
    gross_loss = abs(sum(t.pnl_pct for t in losing)) if losing else 0.0
    profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")
    avg_bars = np.mean([t.exit_idx - t.entry_idx for t in closed_trades]) if closed_trades else 0.0
    avg_kelly_pct = float(np.mean([t.position_size for t in closed_trades])) * 100 if closed_trades else 0.0

    benchmark_curve_arr = arr[MIN_TRAIN_WINDOW:n] / arr[MIN_TRAIN_WINDOW] * initial_capital

    result = BacktestResult(
        asset=asset,
        initial_capital=initial_capital,
        final_capital=round(capital, 2),
        total_return_pct=round(total_return * 100, 2),
        annualized_return_pct=round(annualized_return * 100, 2),
        sharpe=round(sharpe, 4),
        sortino=round(sortino, 4),
        max_drawdown_pct=round(max_dd * 100, 2),
        win_rate=round(win_rate * 100, 2),
        total_trades=len(closed_trades),
        winning_trades=len(winning),
        losing_trades=len(losing),
        avg_holding_bars=round(avg_bars, 1),
        profit_factor=round(profit_factor, 4),
        conflict_days=conflict_days,
        total_trading_days=total_trading_days,
        trades=trades,
        equity_curve=equity_curve,
        benchmark_curve=benchmark_curve_arr.tolist(),
        timestamps=timestamps,
        confidence_series=confidence_series,
        capitulation_trades=capitulation_trades_count,
        avg_kelly_pct=round(avg_kelly_pct, 2),
        strategy=strategy,
    )
    _store_backtest_features(asset, feature_snapshots, result)
    return result


def run_multi_asset_backtest(
    assets: list[str] | tuple[str, ...] = ("BTC", "ETH", "SOL"),
    days: int = 365,
    initial_capital: float = 10000.0,
    strategy: str = "A",
) -> BacktestResult:
    """Run a combined-equity backtest across multiple correlated assets."""
    asset_list = list(assets)
    if not asset_list:
        raise ValueError("assets must contain at least one asset")

    fetched_data: dict[str, dict] = {}
    for asset in asset_list:
        fetched = _run_async(fetch_live_data(asset=asset))
        if len(fetched) == 5:
            prices, _current_price, fng_scores, fng_ok, volumes = fetched
        else:
            prices, _current_price, fng_scores, fng_ok = fetched
            volumes = None
        if not fng_ok:
            from kairos.live import _FNG_FALLBACK as _FB

            fng_scores = list(_FB)
        if not prices:
            raise ValueError(f"No price data available for {asset}")
        fetched_data[asset] = {
            "prices": prices,
            "fng_scores": fng_scores,
            "volumes": volumes,
        }

    min_n = min(len(data["prices"]) for data in fetched_data.values())
    min_n = min(min_n, days) if days > 0 else min_n
    if min_n <= MIN_TRAIN_WINDOW + 1:
        raise ValueError(f"Need more than {MIN_TRAIN_WINDOW + 1} aligned candles for multi-asset backtest")

    price_arrs: dict[str, np.ndarray] = {}
    volume_arrs: dict[str, np.ndarray] = {}
    fng_by_asset: dict[str, list[int]] = {}
    for asset, data in fetched_data.items():
        prices = list(data["prices"])[-min_n:]
        volumes = list(data["volumes"] or [])[-min_n:]
        if len(volumes) < min_n:
            volumes = list(np.abs(np.diff(prices, prepend=prices[0])))

        fng_scores = list(data["fng_scores"] or [])
        if len(fng_scores) >= min_n:
            fng_scores = fng_scores[-min_n:]
        elif not fng_scores:
            fng_scores = [50] * min_n

        price_arrs[asset] = np.asarray(prices, dtype=float)
        volume_arrs[asset] = np.asarray(volumes[:min_n], dtype=float)
        fng_by_asset[asset] = fng_scores

    capital = initial_capital
    positions = {asset: 0.0 for asset in asset_list}
    entry_trades: dict[str, BacktestTrade | None] = {asset: None for asset in asset_list}
    trades_by_asset: dict[str, list[BacktestTrade]] = {asset: [] for asset in asset_list}
    trades: list[BacktestTrade] = []
    high_watermarks = {asset: 0.0 for asset in asset_list}
    low_watermarks = {asset: float("inf") for asset in asset_list}
    equity_curve: list[float] = [initial_capital]
    timestamps: list[datetime] = []
    confidence_series: list[float] = []

    ensembles: dict[str, SignalEnsemble | None] = {asset: None for asset in asset_list}
    hmms: dict[str, object] = {asset: None for asset in asset_list}
    last_train_idx = {asset: 0 for asset in asset_list}
    vol_z_arrs = {asset: _running_vol_z(price_arrs[asset]) for asset in asset_list}
    slopes_c: dict[str, np.ndarray | None] = {asset: None for asset in asset_list}
    if strategy == "C":
        for asset in asset_list:
            smoothed = kalman_smooth(price_arrs[asset])
            slopes = np.zeros(min_n)
            for t in range(10, min_n):
                slopes[t] = float(np.polyfit(range(10), smoothed[t - 9 : t + 1], 1)[0])
            slopes_c[asset] = slopes

    tracker = CorrelationTracker(window=60)
    base_ts = datetime.now(timezone.utc) - timedelta(days=min_n)
    conflict_days = 0
    total_trading_days = 0
    capitulation_trades_count = 0

    for t in range(MIN_TRAIN_WINDOW, min_n - 1):
        ts = base_ts + timedelta(days=t)
        for asset in asset_list:
            arr = price_arrs[asset]
            tracker.update(asset, float(arr[t] / arr[t - 1] - 1.0))

        allocations: list[dict] = []
        order_book_by_asset: dict[str, dict | None] = {}
        signal_by_asset: dict[str, SignalEvent | None] = {}
        capitulation_by_asset: dict[str, bool] = {}
        confidence_values: list[float] = []

        for asset in asset_list:
            arr = price_arrs[asset]
            fng_scores = fng_by_asset[asset]
            vol_arr = volume_arrs[asset]
            order_book_by_asset[asset] = _get_order_book_depth(asset)

            should_retrain = (
                ensembles[asset] is None
                or (t - last_train_idx[asset]) >= 30
                or (
                    hasattr(ensembles[asset], "_candle_count") and t - ensembles[asset]._candle_count > 50  # type: ignore[union-attr]
                )
            )
            if should_retrain:
                new_ensemble, new_hmm = _train_ensemble(arr, fng_scores, t, asset=asset)
                if new_ensemble is not None:
                    ensembles[asset] = new_ensemble
                    hmms[asset] = new_hmm
                    last_train_idx[asset] = t

            if ensembles[asset] is None:
                allocations.append(
                    {
                        "asset": asset,
                        "confidence": 0.0,
                        "direction": 0,
                        "kelly_fraction": 0.0,
                    }
                )
                signal_by_asset[asset] = None
                capitulation_by_asset[asset] = False
                continue

            signal = _generate_signal_at(ensembles[asset], arr, fng_scores, t, asset=asset, prefit_hmm=hmms.get(asset))  # type: ignore[arg-type]
            signal_by_asset[asset] = signal
            confidence_values.append(signal.confidence)
            total_trading_days += 1

            base_kelly = _adaptive_kelly(signal.confidence, trades_by_asset[asset])
            is_neutral = signal.direction == "neutral"
            is_capitulation_trade = False

            if strategy == "A":
                if is_neutral:
                    raw_position = 0.0
                    conflict_days += 1
                else:
                    direction = 1.0 if signal.direction == "bullish" else -1.0
                    raw_position = direction * base_kelly

            elif strategy == "B":
                fng_current = fng_scores[t] if t < len(fng_scores) else 50
                current_vol = float(vol_arr[t]) if t < len(vol_arr) else 0.0
                vol_history = [float(v) for v in vol_arr[: t + 1]]

                if is_neutral and capitulation_triggered(fng_current, current_vol, vol_history):
                    raw_position = _adaptive_kelly(0.65, trades_by_asset[asset])
                    is_capitulation_trade = True
                elif is_neutral:
                    raw_position = 0.0
                    conflict_days += 1
                else:
                    direction = 1.0 if signal.direction == "bullish" else -1.0
                    raw_position = direction * base_kelly

            elif strategy == "C":
                slope_arr = slopes_c[asset]
                slope = float(slope_arr[t]) if slope_arr is not None and t >= 10 else 0.0
                fng_current = fng_scores[t] if t < len(fng_scores) else 50
                current_vol = float(vol_arr[t]) if t < len(vol_arr) else 0.0
                vol_history = [float(v) for v in vol_arr[: t + 1]]

                if (
                    is_neutral
                    and seller_exhaustion_active(slope, current_vol, vol_history)
                    and sentiment_vote(fng_current) == "bullish"
                ):
                    raw_position = 0.25
                elif is_neutral:
                    raw_position = 0.0
                    conflict_days += 1
                else:
                    direction = 1.0 if signal.direction == "bullish" else -1.0
                    raw_position = direction * base_kelly

            else:
                raise ValueError(f"Unknown strategy '{strategy}'. Choose 'A', 'B', or 'C'.")

            capitulation_by_asset[asset] = is_capitulation_trade
            direction_int = 1 if raw_position > 0 else -1 if raw_position < 0 else 0
            allocations.append(
                {
                    "asset": asset,
                    "confidence": signal.confidence,
                    "direction": direction_int,
                    "kelly_fraction": abs(raw_position),
                }
            )

        corr_matrix = tracker.get_matrix(asset_list)
        desired_positions = _multi_asset_kelly(allocations, corr_matrix)
        capped_positions = [
            _apply_position_cap(
                position,
                [desired_positions[j] for j in range(len(desired_positions)) if j != i],
            )
            for i, position in enumerate(desired_positions)
        ]

        for asset, new_position in zip(asset_list, capped_positions):
            arr = price_arrs[asset]
            signal = signal_by_asset[asset]
            order_book = order_book_by_asset.get(asset)
            position = positions[asset]
            entry_trade = entry_trades[asset]

            current_price = float(arr[t])
            atr_val = _atr(arr[: t + 1])
            if entry_trade is not None and not entry_trade.closed:
                if position > 0:
                    high_watermarks[asset] = max(high_watermarks[asset], current_price)
                    entry_trade.trailing_stop = high_watermarks[asset] - ATR_MULTIPLIER_LONG * atr_val
                elif position < 0:
                    low_watermarks[asset] = min(low_watermarks[asset], current_price)
                    entry_trade.trailing_stop = low_watermarks[asset] + ATR_MULTIPLIER_SHORT * atr_val

            signal_wants_exit = (new_position != position) and entry_trade is not None
            should_exit = False
            if signal_wants_exit:
                bars_held = t - entry_trade.entry_idx
                if bars_held >= MIN_HOLD_BARS:
                    regime_changed = signal is not None and signal.regime != entry_trade.entry_regime
                    if position > 0:
                        atr_stop_hit = current_price < high_watermarks[asset] - ATR_MULTIPLIER_LONG * atr_val
                    else:
                        atr_stop_hit = current_price > low_watermarks[asset] + ATR_MULTIPLIER_SHORT * atr_val
                    max_bars = int((entry_trade.estimated_hours or 72.0) / 24.0 * 2)
                    max_hold_hit = bars_held >= max_bars
                    should_exit = regime_changed or atr_stop_hit or max_hold_hit

            if should_exit and entry_trade is not None:
                exit_direction = "sell" if position > 0 else "buy"
                exit_slippage = _compute_slippage(
                    signal.regime if signal is not None else entry_trade.regime,
                    float(vol_z_arrs[asset][t]),
                    order_size_usd=abs(position) * capital,
                    order_book=order_book,
                    direction=exit_direction,
                )
                exit_price = arr[t] * (1 - exit_slippage if position > 0 else 1 + exit_slippage)
                entry_trade.exit_idx = t
                entry_trade.exit_time = ts
                entry_trade.exit_price = exit_price
                entry_trade.pnl_pct = (exit_price / entry_trade.entry_price - 1) * position
                entry_trade.closed = True
                entry_trade.cumulative_capital = round(capital, 2)

            if entry_trade is None:
                effective_new_position = new_position
            elif should_exit or new_position == position:
                effective_new_position = new_position
            else:
                effective_new_position = position

            if effective_new_position != position and effective_new_position != 0 and signal is not None:
                entry_direction = "buy" if effective_new_position > 0 else "sell"
                entry_slippage = _compute_slippage(
                    signal.regime,
                    float(vol_z_arrs[asset][t]),
                    order_size_usd=abs(effective_new_position) * capital,
                    order_book=order_book,
                    direction=entry_direction,
                )
                entry_price = arr[t] * (1 + entry_slippage if effective_new_position > 0 else 1 - entry_slippage)
                if effective_new_position > 0:
                    initial_stop = entry_price - ATR_MULTIPLIER_LONG * atr_val
                    high_watermarks[asset] = entry_price
                else:
                    initial_stop = entry_price + ATR_MULTIPLIER_SHORT * atr_val
                    low_watermarks[asset] = entry_price
                is_capitulation_trade = capitulation_by_asset[asset]
                entry_trade = BacktestTrade(
                    entry_idx=t,
                    entry_time=ts,
                    entry_price=entry_price,
                    direction="bullish" if effective_new_position > 0 else "bearish",
                    confidence=signal.confidence,
                    regime=signal.regime,
                    mechanism=f"{asset}: "
                    + (signal.mechanism + " [CAPITULATION]" if is_capitulation_trade else signal.mechanism),
                    estimated_hours=signal.estimated_hours,
                    is_capitulation=is_capitulation_trade,
                    position_size=abs(effective_new_position),
                    trailing_stop=initial_stop,
                    entry_regime=signal.regime,
                )
                entry_trades[asset] = entry_trade
                trades_by_asset[asset].append(entry_trade)
                trades.append(entry_trade)
                if is_capitulation_trade:
                    capitulation_trades_count += 1
            elif effective_new_position == 0 and should_exit:
                entry_trades[asset] = None
            else:
                entry_trades[asset] = entry_trade

            positions[asset] = effective_new_position

        portfolio_return = 0.0
        for asset in asset_list:
            arr = price_arrs[asset]
            daily_ret = arr[t + 1] / arr[t] - 1
            portfolio_return += daily_ret * positions[asset]

        capital *= 1 + portfolio_return
        equity_curve.append(capital)
        timestamps.append(ts)
        confidence_series.append(float(np.mean(confidence_values)) if confidence_values else 0.0)

    exit_idx = min_n - 1
    for asset in asset_list:
        entry_trade = entry_trades[asset]
        position = positions[asset]
        if entry_trade is not None and not entry_trade.closed:
            arr = price_arrs[asset]
            final_direction = "sell" if position > 0 else "buy"
            final_order_book = _get_order_book_depth(asset)
            final_slip = _compute_slippage(
                entry_trade.regime,
                float(vol_z_arrs[asset][-1]),
                order_size_usd=abs(position) * capital,
                order_book=final_order_book,
                direction=final_direction,
            )
            exit_price = arr[exit_idx] * (1 - final_slip if position > 0 else 1 + final_slip)
            entry_trade.exit_idx = exit_idx
            entry_trade.exit_time = base_ts + timedelta(days=exit_idx)
            entry_trade.exit_price = exit_price
            entry_trade.pnl_pct = (exit_price / entry_trade.entry_price - 1) * position
            entry_trade.closed = True
            entry_trade.cumulative_capital = round(capital, 2)

    equity_arr = np.asarray(equity_curve, dtype=float)
    safe_equity = np.where(equity_arr[:-1] == 0, 1e-8, equity_arr[:-1])
    daily_returns = np.diff(equity_arr) / safe_equity
    total_return = (capital / initial_capital) - 1
    years = max((min_n - MIN_TRAIN_WINDOW) / 365.0, 1 / 365.0)
    annualized_return = (1 + total_return) ** (1 / years) - 1

    max_dd, _peak_i, _trough_i = _compute_max_drawdown(equity_arr)
    sharpe = _compute_sharpe(daily_returns)
    sortino = _compute_sortino(daily_returns)

    closed_trades = [trade for trade in trades if trade.closed and trade.pnl_pct is not None]
    winning = [trade for trade in closed_trades if trade.pnl_pct > 0]
    losing = [trade for trade in closed_trades if trade.pnl_pct <= 0]
    win_rate = len(winning) / len(closed_trades) if closed_trades else 0.0
    gross_win = sum(trade.pnl_pct for trade in winning) if winning else 0.0
    gross_loss = abs(sum(trade.pnl_pct for trade in losing)) if losing else 0.0
    profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")
    avg_bars = np.mean([trade.exit_idx - trade.entry_idx for trade in closed_trades]) if closed_trades else 0.0
    avg_kelly_pct = float(np.mean([trade.position_size for trade in closed_trades])) * 100 if closed_trades else 0.0

    benchmark_components = [
        price_arrs[asset][MIN_TRAIN_WINDOW:min_n] / price_arrs[asset][MIN_TRAIN_WINDOW] for asset in asset_list
    ]
    benchmark_curve_arr = np.mean(np.vstack(benchmark_components), axis=0) * initial_capital

    return BacktestResult(
        asset="+".join(asset_list),
        initial_capital=initial_capital,
        final_capital=round(capital, 2),
        total_return_pct=round(total_return * 100, 2),
        annualized_return_pct=round(annualized_return * 100, 2),
        sharpe=round(sharpe, 4),
        sortino=round(sortino, 4),
        max_drawdown_pct=round(max_dd * 100, 2),
        win_rate=round(win_rate * 100, 2),
        total_trades=len(closed_trades),
        winning_trades=len(winning),
        losing_trades=len(losing),
        avg_holding_bars=round(avg_bars, 1),
        profit_factor=round(profit_factor, 4),
        conflict_days=conflict_days,
        total_trading_days=total_trading_days,
        trades=trades,
        equity_curve=equity_curve,
        benchmark_curve=benchmark_curve_arr.tolist(),
        timestamps=timestamps,
        confidence_series=confidence_series,
        capitulation_trades=capitulation_trades_count,
        avg_kelly_pct=round(avg_kelly_pct, 2),
        strategy=strategy,
    )
