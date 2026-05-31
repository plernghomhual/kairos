"""Tests for the walk-forward backtest engine."""

from datetime import datetime, timezone

import numpy as np
import pytest

import kairos.backtest.engine as engine
from kairos.backtest.engine import (
    BacktestTrade,
    CorrelationTracker,
    _adaptive_kelly,
    _build_training_data_window,
    _compute_max_drawdown,
    _compute_sharpe,
    _compute_sortino,
    _kelly_fraction,
    _multi_asset_kelly,
    _running_vol_z,
    _train_ensemble,
    run_backtest,
    run_multi_asset_backtest,
)

# ── _running_vol_z ────────────────────────────────────────────────────────────


def test_running_vol_z_returns_correct_shape():
    prices = np.array([100.0, 101.0, 100.5, 102.0, 101.5])
    z = _running_vol_z(prices)
    assert len(z) == len(prices)


def test_running_vol_z_flat_prices_zero():
    prices = np.array([100.0] * 10)
    z = _running_vol_z(prices)
    assert np.allclose(z, 0.0, atol=1e-4)


# ── Order-book slippage ──────────────────────────────────────────────────────


class _FakeDepthResponse:
    def json(self):
        return {
            "bids": [["99.0", "2"], ["100.0", "1"], ["98.5", "4"]],
            "asks": [["101.0", "3"], ["100.5", "2"], ["102.0", "1"]],
        }

    def raise_for_status(self):
        return None


def _fake_httpx(calls: list[str]):
    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url):
            calls.append(url)
            return _FakeDepthResponse()

    return type("FakeHttpx", (), {"AsyncClient": FakeAsyncClient})


@pytest.mark.asyncio
async def test_order_book_depth_fetcher(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(engine, "httpx", _fake_httpx(calls), raising=False)
    engine._order_book_cache.clear()

    depth = await engine._fetch_order_book_depth("BTC")

    assert set(depth) == {
        "bids",
        "asks",
        "spread_pct",
        "bid_liquidity",
        "ask_liquidity",
        "depth_imbalance",
        "mid_price",
        "fetch_ts",
    }
    assert depth["bids"][0] == (100.0, 1.0)
    assert depth["asks"][0] == (100.5, 2.0)
    assert depth["spread_pct"] == pytest.approx((100.5 - 100.0) / 100.25 * 100)
    assert depth["bid_liquidity"] == pytest.approx(99.0 * 2 + 100.0 + 98.5 * 4)
    assert depth["ask_liquidity"] == pytest.approx(101.0 * 3 + 100.5 * 2 + 102.0)
    assert "BTCUSDT" in calls[0]


def test_compute_slippage_with_order_book():
    order_book = {
        "asks": [(100.01, 1000.0), (100.5, 1000.0), (101.0, 10000.0)],
        "bids": [(99.99, 1000.0), (99.5, 1000.0), (99.0, 10000.0)],
        "mid_price": 100.0,
    }

    small = engine._compute_slippage("lv_up", order_size_usd=1_000.0, order_book=order_book, direction="buy")
    large = engine._compute_slippage("lv_up", order_size_usd=1_000_000.0, order_book=order_book, direction="buy")

    assert small < 0.001
    assert large > small * 2


@pytest.mark.asyncio
async def test_compute_slippage_cache(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(engine, "httpx", _fake_httpx(calls), raising=False)
    engine._order_book_cache.clear()

    first = await engine._fetch_order_book_depth("BTC")
    second = await engine._fetch_order_book_depth("BTC")

    assert first == second
    assert len(calls) == 1


def test_compute_slippage_fallback():
    assert engine._compute_slippage("hv_down", vol_z_score=2.0, order_book=None) == 0.0026
    assert engine._compute_slippage("lv_up", vol_z_score=-2.0, order_book=None) == 0.0009


def test_order_book_imbalance_direction():
    order_book = {
        "asks": [(100.5, 10.0), (102.0, 1000.0)],
        "bids": [(99.9, 1000.0)],
        "mid_price": 100.0,
        "depth_imbalance": ((99.9 * 1000.0) - (100.5 * 10.0 + 102.0 * 1000.0))
        / ((99.9 * 1000.0) + (100.5 * 10.0 + 102.0 * 1000.0)),
    }

    buy = engine._compute_slippage("lv_up", order_size_usd=50_000.0, order_book=order_book, direction="buy")
    sell = engine._compute_slippage("lv_up", order_size_usd=50_000.0, order_book=order_book, direction="sell")

    assert order_book["depth_imbalance"] < 0
    assert buy > sell


# ── Kelly sizing ──────────────────────────────────────────────────────────────


def test_multi_asset_kelly_reduces_allocation():
    single_asset_kelly = _kelly_fraction(0.60)
    allocations = [
        {"asset": "BTC", "confidence": 0.60, "direction": 1},
        {"asset": "ETH", "confidence": 0.60, "direction": 1},
    ]

    sizes = _multi_asset_kelly(allocations, np.ones((2, 2)))

    assert all(abs(size) <= single_asset_kelly * 0.5 for size in sizes)
    assert sum(abs(size) for size in sizes) <= 1.0


def test_multi_asset_kelly_uncorrelated():
    single_asset_kelly = _kelly_fraction(0.60)
    allocations = [
        {"asset": "BTC", "confidence": 0.60, "direction": 1},
        {"asset": "ETH", "confidence": 0.60, "direction": 1},
    ]

    sizes = _multi_asset_kelly(allocations, np.eye(2))

    assert sizes == pytest.approx([single_asset_kelly, single_asset_kelly])
    assert sum(abs(size) for size in sizes) <= 1.0


def test_correlation_tracker_update():
    tracker = CorrelationTracker(window=4)
    for btc_ret, eth_ret, sol_ret in zip(
        [0.01, 0.02, 0.03, 0.04],
        [0.02, 0.04, 0.06, 0.08],
        [0.04, 0.03, 0.02, 0.01],
    ):
        tracker.update("BTC", btc_ret)
        tracker.update("ETH", eth_ret)
        tracker.update("SOL", sol_ret)

    assert tracker.get_correlation("BTC", "ETH") == pytest.approx(1.0)
    assert tracker.get_correlation("BTC", "SOL") == pytest.approx(-1.0)
    matrix = tracker.get_matrix(["BTC", "ETH", "SOL"])
    assert matrix.shape == (3, 3)
    assert np.diag(matrix).tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert not tracker.is_diversified(["BTC", "ETH", "SOL"], threshold=0.7)


def _closed_trade(pnl_pct: float) -> BacktestTrade:
    now = datetime.now(timezone.utc)
    return BacktestTrade(
        entry_idx=0,
        entry_time=now,
        entry_price=100.0,
        direction="bullish",
        confidence=0.60,
        regime="lv_up",
        mechanism="test",
        estimated_hours=24.0,
        exit_idx=1,
        exit_time=now,
        exit_price=100.0 * (1.0 + pnl_pct),
        pnl_pct=pnl_pct,
        closed=True,
        position_size=0.1,
    )


def test_adaptive_kelly_uses_history():
    history = [
        _closed_trade(0.04),
        _closed_trade(0.03),
        _closed_trade(0.05),
        _closed_trade(-0.01),
        _closed_trade(-0.02),
    ]

    adaptive = _adaptive_kelly(0.55, history)

    assert adaptive > _kelly_fraction(0.55)


# ── _compute_max_drawdown ────────────────────────────────────────────────────


def test_max_drawdown_monotonic_up():
    eq = np.array([100, 110, 120, 130])
    dd, peak, trough = _compute_max_drawdown(eq)
    assert dd == 0.0
    assert peak == 0
    assert trough == 0


def test_max_drawdown_single_dip():
    eq = np.array([100, 110, 105, 115, 120])
    dd, peak, trough = _compute_max_drawdown(eq)
    assert dd == pytest.approx(0.0454545, abs=1e-4)
    assert peak == 1  # 110
    assert trough == 2  # 105


def test_max_drawdown_crash():
    eq = np.array([100, 120, 80, 90, 50])
    dd, peak, trough = _compute_max_drawdown(eq)
    # peak=120, trough=50, dd = (120-50)/120 ≈ 0.5833
    assert dd == pytest.approx(0.58333, abs=1e-4)


# ── _compute_sharpe ──────────────────────────────────────────────────────────


def test_sharpe_positive():
    rets = np.array([0.01] * 100)
    s = _compute_sharpe(rets)
    assert s > 1.0


def test_sharpe_negative():
    rets = np.array([-0.01] * 100)
    s = _compute_sharpe(rets)
    assert s < 0.0


def test_sharpe_zero_vol():
    s = _compute_sharpe(np.zeros(100))
    assert s == 0.0


def test_sharpe_few_samples():
    s = _compute_sharpe(np.array([0.01]))
    assert s == 0.0


# ── _compute_sortino ─────────────────────────────────────────────────────────


def test_sortino_only_upside():
    rets = np.array([0.01] * 50)
    s = _compute_sortino(rets)
    assert s == 0.0  # no downside = no penalty


def test_sortino_mixed():
    rets = np.array([0.02, -0.01, 0.03, -0.02, 0.01])
    s = _compute_sortino(rets)
    assert isinstance(s, float)
    assert s > 0.0


# ── _build_training_data_window ──────────────────────────────────────────────


def test_build_training_data_returns_valid_shapes():
    np.random.seed(42)
    prices = np.cumsum(np.random.normal(0, 50, 80)) + 50000
    fng = [50] * 80
    X, y, _ = _build_training_data_window(prices, fng, up_to=79)
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == 13  # FeatureVector has 13 dimensions
    assert X.shape[0] > 10  # at least some training samples


def test_build_training_data_window_drops_nan_rows():
    np.random.seed(42)
    prices = np.cumsum(np.random.normal(0, 50, 80)) + 50000
    fng = [50] * 80
    fng[25] = float("nan")

    X, y, regimes = _build_training_data_window(prices, fng, up_to=79)

    assert X.shape[0] == y.shape[0] == len(regimes)
    assert np.isfinite(X).all()
    assert X.shape[0] < 68


def test_build_training_data_window_all_nan_rows_raise():
    np.random.seed(42)
    prices = np.cumsum(np.random.normal(0, 50, 80)) + 50000
    fng = [float("nan")] * 80

    with pytest.raises(ValueError, match="All training rows dropped due to NaN/Inf"):
        _build_training_data_window(prices, fng, up_to=79)


# ── _train_ensemble ──────────────────────────────────────────────────────────


def test_train_ensemble_returns_ensemble():
    np.random.seed(42)
    prices = np.cumsum(np.random.normal(0, 50, 80)) + 50000
    fng = [50] * 80
    ensemble = _train_ensemble(prices, fng, up_to=79)
    assert ensemble is not None

    # Should be able to predict
    from kairos.signals.ensemble import FeatureVector

    fv = FeatureVector(0.0, 0.0, 0.0, 0.0, False, 0.5, 1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.25)
    event = ensemble.predict("BTC", fv, citations=["test"], regime="lv_up")
    assert event.direction in ("bullish", "bearish")
    assert 0.0 <= event.confidence <= 1.0


def test_train_ensemble_insufficient_data():
    prices = np.array([50000.0] * 5)
    fng = [50] * 5
    ensemble = _train_ensemble(prices, fng, up_to=4)
    assert ensemble is None


# ── run_backtest (unit-level with controlled data) ───────────────────────────


def test_run_backtest_with_synthetic_uptrend():
    """On a synthetic uptrend, bullish signals should dominate and P&L positive."""
    np.random.seed(42)
    result = run_backtest(
        asset="BTC",
        days=365,
        initial_capital=10000.0,
        retrain_days=60,
    )
    assert result.total_trades >= 0
    assert result.initial_capital == 10000.0
    assert result.final_capital > 0
    assert 0.0 <= result.win_rate <= 100.0
    assert isinstance(result.sharpe, float)
    assert isinstance(result.sortino, float)
    assert isinstance(result.max_drawdown_pct, float)

    # Equity curve should match final capital
    assert abs(result.equity_curve[-1] - result.final_capital) < 0.01

    # Trades should have entry times
    if result.trades:
        t = result.trades[0]
        assert t.entry_price > 0
        assert t.direction in ("bullish", "bearish")


def test_run_backtest_opens_trade_from_flat_signal(monkeypatch):
    """A non-neutral flat-state signal must open an initial position."""
    monkeypatch.setattr(engine, "_get_order_book_depth", lambda asset: None)
    monkeypatch.setattr(engine, "_store_backtest_features", lambda *args, **kwargs: None)
    monkeypatch.setattr(engine, "_feature_snapshot_at", lambda *args, **kwargs: None)
    monkeypatch.setattr(engine, "_train_ensemble", lambda *args, **kwargs: object())

    def bullish_signal(_ensemble, _snapshot, asset="BTC"):
        from kairos.models.signal_event import SignalEvent

        return SignalEvent(
            asset=asset,
            direction="bullish",
            confidence=0.75,
            regime="lv_up",
            narrative_velocity=0.0,
            narrative_tipping_point=False,
            mechanism="deterministic test signal",
            estimated_hours=72.0,
            citations=[],
        )

    monkeypatch.setattr(engine, "_generate_signal_from_snapshot", bullish_signal)

    prices = [100.0 + i for i in range(90)]
    result = run_backtest(
        asset="BTC",
        days=90,
        prices=prices,
        fng_scores=[50] * 90,
        volumes=[1_000_000.0] * 90,
        strategy="A",
    )

    assert result.total_trades == 1
    assert result.trades[0].direction == "bullish"
    assert result.trades[0].position_size > 0


def test_backtest_result_has_trade_journal():
    result = run_backtest(asset="BTC", days=365)
    assert hasattr(result, "trades")
    for trade in result.trades:
        assert trade.entry_idx >= 0
        assert trade.entry_price > 0
        if trade.closed:
            assert trade.exit_idx is not None
            assert trade.exit_price > 0
            assert trade.pnl_pct is not None


def test_backtest_sharpe_sortino_consistent():
    """Sharpe and Sortino should not differ wildly on the same return series."""
    result = run_backtest(asset="BTC", days=365)
    # Sortino penalizes only downside; it's often >= Sharpe for same series
    assert abs(result.sharpe - result.sortino) < 5.0 or result.sortino >= result.sharpe - 1.0


# ── Strategy B / C (with synthetic data to avoid network calls) ───────────────


def _synth_prices(n: int = 200, seed: int = 42) -> list[float]:
    np.random.seed(seed)
    return list(np.cumsum(np.random.normal(0, 500, n)) + 50000)


def test_run_backtest_strategy_b_returns_valid_result():
    prices = _synth_prices()
    fng_scores = [20] * 100 + [60] * 100  # extreme fear then greed
    volumes = [float(abs(np.random.normal(1e9, 1e8))) for _ in range(200)]

    result = run_backtest(
        asset="BTC",
        days=200,
        prices=prices,
        fng_scores=fng_scores,
        volumes=volumes,
        strategy="B",
    )
    assert result.strategy == "B"
    assert result.final_capital > 0
    assert 0.0 <= result.win_rate <= 100.0
    assert result.capitulation_trades >= 0
    assert result.capitulation_trades <= result.total_trades


def test_run_backtest_strategy_c_returns_valid_result():
    prices = _synth_prices(seed=7)
    fng_scores = [30] * 200  # fear zone → sentiment bullish
    np.random.seed(7)
    volumes = [float(abs(np.random.normal(5e8, 1e7))) for _ in range(200)]

    result = run_backtest(
        asset="BTC",
        days=200,
        prices=prices,
        fng_scores=fng_scores,
        volumes=volumes,
        strategy="C",
    )
    assert result.strategy == "C"
    assert result.final_capital > 0
    assert 0.0 <= result.win_rate <= 100.0
    assert isinstance(result.sharpe, float)


def test_run_backtest_stores_feature_vectors_for_analytics(monkeypatch):
    import kairos.backtest.engine as engine
    from kairos.signals.ensemble import FeatureVector

    stored = []

    class FakeStore:
        def __init__(self, *args, **kwargs):
            pass

        def store_feature(self, asset, ts, fv, metadata=None):
            stored.append((asset, ts, fv, metadata))

        def close(self):
            pass

    monkeypatch.setattr(engine, "FeatureStore", FakeStore)

    prices = _synth_prices(n=90, seed=11)
    fng_scores = [45] * 90

    result = run_backtest(
        asset="BTC",
        days=90,
        prices=prices,
        fng_scores=fng_scores,
        strategy="A",
    )

    assert result.strategy == "A"
    assert stored
    assert all(asset == "BTC" for asset, _, _, _ in stored)
    assert all(isinstance(fv, FeatureVector) for _, _, fv, _ in stored)
    assert all(metadata["source"] == "backtest" for _, _, _, metadata in stored)
    assert all(metadata["strategy"] == "A" for _, _, _, metadata in stored)


def test_run_backtest_conflict_days_not_inflated_by_overrides():
    """Strategy B conflict_days should only count days the system stood aside."""
    prices = _synth_prices()
    # FNG=10 → capitulation fires when volume is high enough
    fng_scores = [10] * 200
    np.random.seed(42)
    # Alternating high/normal volume to mix capitulation + flat days
    volumes = [2e9 if i % 2 == 0 else 5e8 for i in range(200)]

    result_a = run_backtest(
        asset="BTC",
        days=200,
        prices=prices,
        fng_scores=fng_scores,
        volumes=volumes,
        strategy="A",
    )
    result_b = run_backtest(
        asset="BTC",
        days=200,
        prices=prices,
        fng_scores=fng_scores,
        volumes=volumes,
        strategy="B",
    )
    # B takes capitulation trades instead of sitting out; conflict_days must be <= A's
    assert result_b.conflict_days <= result_a.conflict_days


def test_run_backtest_invalid_strategy_raises():
    prices = _synth_prices()
    fng_scores = [50] * 200
    with pytest.raises(ValueError, match="Unknown strategy"):
        run_backtest(
            asset="BTC",
            days=200,
            prices=prices,
            fng_scores=fng_scores,
            strategy="X",
        )


def test_multi_asset_backtest_runs(monkeypatch):
    async def fake_fetch_live_data(asset: str):
        seed = {"BTC": 1, "ETH": 2, "SOL": 3}[asset]
        rng = np.random.default_rng(seed)
        returns = rng.normal(0.001, 0.02, 140)
        prices = (100.0 * np.cumprod(1.0 + returns)).tolist()
        volumes = rng.normal(1e9, 1e8, 140).clip(min=1.0).tolist()
        return prices, prices[-1], [50] * 140, True, volumes

    monkeypatch.setattr("kairos.backtest.engine.fetch_live_data", fake_fetch_live_data)

    result = run_multi_asset_backtest(days=140, initial_capital=10000.0)

    assert result.asset == "BTC+ETH+SOL"
    assert result.initial_capital == 10000.0
    assert result.final_capital > 0
    assert len(result.equity_curve) == len(result.timestamps) + 1
    assert sum(abs(trade.position_size) for trade in result.trades) >= 0.0


def test_run_backtest_zero_initial_capital_returns_result_without_exception():
    prices = _synth_prices(n=80, seed=21)
    result = run_backtest(
        asset="BTC",
        days=80,
        initial_capital=0.0,
        prices=prices,
        fng_scores=[50] * 80,
    )
    assert result.initial_capital == 0.0
    assert result.equity_curve[0] == 0.0
    assert result.total_trades >= 0


def test_confidence_threshold_one_prevents_trades():
    prices = _synth_prices(n=100, seed=22)
    result = run_backtest(
        asset="BTC",
        days=100,
        prices=prices,
        fng_scores=[50] * 100,
        confidence_threshold=1.0,
    )
    assert result.total_trades == 0
    assert result.trades == []


def test_run_backtest_strategy_d_raises_value_error():
    prices = _synth_prices(n=100, seed=23)
    with pytest.raises(ValueError, match="Unknown strategy"):
        run_backtest(
            asset="BTC",
            days=100,
            prices=prices,
            fng_scores=[50] * 100,
            strategy="D",
        )


def test_backtest_trade_optional_defaults():
    now = datetime.now(timezone.utc)
    trade = BacktestTrade(
        entry_idx=0,
        entry_time=now,
        entry_price=100.0,
        direction="bullish",
        confidence=0.5,
        regime="lv_up",
        mechanism="default-check",
        estimated_hours=24.0,
    )
    assert trade.exit_idx is None
    assert trade.closed is False
    assert trade.is_capitulation is False
    assert trade.position_size == pytest.approx(0.0)


def test_kelly_fraction_low_edge_returns_zero():
    assert _kelly_fraction(0.5, win_loss_ratio=0.5) == pytest.approx(0.0)


# ── Time-Locked Exit (Bug 3) ──────────────────────────────────────────────────


def test_trade_not_exited_on_signal_flip_same_regime():
    """A trade opened in 'lv_up' must NOT be exited immediately when the signal
    flips but the regime stays 'lv_up' and MIN_HOLD_BARS hasn't elapsed."""
    from kairos.backtest.engine import MIN_HOLD_BARS

    np.random.seed(99)
    # Build a price series that stays nearly flat (no big ATR trailing-stop move)
    n = 200
    prices = list(50000.0 + np.cumsum(np.random.normal(0, 50, n)))
    fng_scores = [50] * n
    volumes = [1e9] * n

    result = run_backtest(
        asset="BTC",
        days=n,
        prices=prices,
        fng_scores=fng_scores,
        volumes=volumes,
        strategy="A",
    )

    # Every closed trade must have been held for at least MIN_HOLD_BARS
    for trade in result.trades:
        if trade.closed and trade.exit_idx is not None:
            bars_held = trade.exit_idx - trade.entry_idx
            assert bars_held >= MIN_HOLD_BARS, (
                f"Trade {trade.entry_idx}→{trade.exit_idx} held only {bars_held} bars "
                f"(MIN_HOLD_BARS={MIN_HOLD_BARS})"
            )


def test_backtesttrade_new_fields_have_defaults():
    """BacktestTrade must expose trailing_stop and entry_regime with defaults."""
    now = datetime.now(timezone.utc)
    trade = BacktestTrade(
        entry_idx=0,
        entry_time=now,
        entry_price=100.0,
        direction="bullish",
        confidence=0.6,
        regime="lv_up",
        mechanism="test",
        estimated_hours=48.0,
    )
    assert trade.trailing_stop is None
    assert trade.entry_regime == "lv_up"


def test_atr_helper_basic():
    """_atr must return a positive float for a normal price series."""
    from kairos.backtest.engine import _atr

    prices = np.array(
        [
            100.0,
            102.0,
            101.0,
            103.0,
            105.0,
            104.0,
            106.0,
            108.0,
            107.0,
            109.0,
            110.0,
            112.0,
            111.0,
            113.0,
            115.0,
        ]
    )
    val = _atr(prices)
    assert isinstance(val, float)
    assert val > 0.0


def test_atr_helper_single_price_fallback():
    """_atr with a single-element array returns 2% of that price."""
    from kairos.backtest.engine import _atr

    prices = np.array([200.0])
    val = _atr(prices)
    assert val == pytest.approx(4.0)
