from datetime import datetime, timezone

import numpy as np
import pytest

from kairos.backtest.engine import BacktestResult, BacktestTrade
from kairos.models.signal_event import SignalEvent
from kairos.signals.ensemble import FeatureVector, SignalEnsemble
from kairos.signals.regime import fit_regime_model


@pytest.fixture(scope="session")
def sample_prices_60():
    rng = np.random.default_rng(1701)
    returns = rng.normal(0.001, 0.018, 60)
    return (50_000.0 * np.cumprod(1.0 + returns)).round(2).tolist()


@pytest.fixture(scope="session")
def sample_fng_60():
    rng = np.random.default_rng(1702)
    values = np.clip(rng.normal(52, 13, 60), 0, 100)
    return values.round().astype(int).tolist()


@pytest.fixture(scope="session")
def sample_feature_vector():
    return FeatureVector(
        kalman_slope=0.015,
        volume_z_score=0.8,
        anomaly_score=0.0,
        narrative_velocity=0.03,
        narrative_tipping_point=True,
        saturation=0.45,
        regime_lv_up=1.0,
        regime_hv_up=0.0,
        regime_lv_down=0.0,
        regime_hv_down=0.0,
        causal_bullish=0.68,
        causal_confidence=0.74,
        macro_dff=0.25,
    )


@pytest.fixture(scope="function")
def trained_ensemble():
    ensemble = SignalEnsemble()
    ensemble.fit_synthetic_fallback()
    return ensemble


@pytest.fixture(scope="session")
def fitted_hmm(sample_prices_60):
    prices = np.array(sample_prices_60, dtype=float)
    returns = np.diff(prices) / (prices[:-1] + 1e-8)
    volatility = np.abs(returns)
    return fit_regime_model(np.column_stack([returns, volatility]), n_states=4)


@pytest.fixture(scope="session")
def sample_signal_event():
    return SignalEvent(
        asset="BTC",
        direction="bullish",
        confidence=0.72,
        regime="lv_up",
        narrative_velocity=0.03,
        narrative_tipping_point=True,
        mechanism="fixture",
        estimated_hours=24.0,
        citations=["fixture"],
    )


@pytest.fixture(scope="function")
def backtest_result():
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    trade = BacktestTrade(
        entry_idx=0,
        entry_time=now,
        entry_price=100.0,
        direction="bullish",
        confidence=0.64,
        regime="lv_up",
        mechanism="fixture",
        estimated_hours=24.0,
        exit_idx=1,
        exit_time=now,
        exit_price=102.0,
        pnl_pct=0.02,
        closed=True,
        position_size=0.1,
    )
    return BacktestResult(
        asset="BTC",
        initial_capital=10_000.0,
        final_capital=10_200.0,
        total_return_pct=2.0,
        annualized_return_pct=12.0,
        sharpe=1.1,
        sortino=1.4,
        max_drawdown_pct=0.5,
        win_rate=100.0,
        total_trades=1,
        winning_trades=1,
        losing_trades=0,
        avg_holding_bars=1.0,
        profit_factor=0.0,
        conflict_days=0,
        total_trading_days=2,
        trades=[trade],
        equity_curve=[10_000.0, 10_200.0],
        benchmark_curve=[10_000.0, 10_100.0],
        timestamps=[now],
        confidence_series=[0.64],
    )
