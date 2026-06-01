import csv
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from io import StringIO

import numpy as np
import pytest

import kairos.model_cache as model_cache
from kairos.backtest.engine import _build_training_data_window, run_backtest
from kairos.backtest.report import format_csv_trades
from kairos.live import run_pipeline
from kairos.models.signal_event import SignalEvent
from kairos.signals.ensemble import FeatureVector
from kairos.signals.regime import predict_regime


def _integration_prices(n=100, seed=3001):
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.001, 0.015, n)
    return (100.0 * np.cumprod(1.0 + returns)).tolist()


def _one_hot_regime(regime):
    return {
        "regime_lv_up": 1.0 if regime == "lv_up" else 0.0,
        "regime_hv_up": 1.0 if regime == "hv_up" else 0.0,
        "regime_lv_down": 1.0 if regime == "lv_down" else 0.0,
        "regime_hv_down": 1.0 if regime == "hv_down" else 0.0,
    }


def test_pipeline_output_asset_can_feed_backtest_inputs():
    prices = _integration_prices(n=90)
    fng_scores = [50] * 90
    event = run_pipeline(prices, fng_scores=fng_scores, asset="BTC")
    result = run_backtest(
        asset=event.asset,
        days=90,
        prices=prices,
        fng_scores=fng_scores,
        confidence_threshold=1.0,
    )
    assert isinstance(event, SignalEvent)
    assert result.asset == "BTC"
    assert result.total_trades == 0


def test_feature_vector_xgboost_signal_event_roundtrip(trained_ensemble, sample_feature_vector):
    event = trained_ensemble.predict("BTC", sample_feature_vector, citations=["integration"])
    assert isinstance(event, SignalEvent)
    assert event.asset == "BTC"
    assert event.citations == ["integration"]


def test_hmm_feature_vector_ensemble_end_to_end(fitted_hmm, trained_ensemble, sample_prices_60):
    prices = np.array(sample_prices_60, dtype=float)
    returns = np.diff(prices) / (prices[:-1] + 1e-8)
    volatility = np.abs(returns)
    latest = np.column_stack([returns, volatility])[-1:]
    regime = predict_regime(fitted_hmm, latest)
    regime_flags = _one_hot_regime(regime)
    fv = FeatureVector(
        kalman_slope=float(returns[-1]),
        volume_z_score=0.0,
        anomaly_score=0.0,
        narrative_velocity=0.01,
        narrative_tipping_point=False,
        saturation=0.5,
        causal_bullish=0.55,
        causal_confidence=0.6,
        macro_dff=0.25,
        **regime_flags,
    )
    event = trained_ensemble.predict("BTC", fv, citations=["hmm"], regime=regime)
    assert event.regime == regime
    assert event.direction in ("bullish", "bearish")


def test_model_cache_save_load_predict_roundtrip(tmp_path, monkeypatch, trained_ensemble, sample_feature_vector):
    monkeypatch.setattr(model_cache, "_CACHE_DIR", tmp_path)
    before = trained_ensemble.predict("BTC", sample_feature_vector, citations=[], regime="lv_up")
    model_cache.save_model(trained_ensemble, "BTC", 100)
    loaded = model_cache.load_model("BTC", 100)
    after = loaded.predict("BTC", sample_feature_vector, citations=[], regime="lv_up")
    assert loaded is not None
    assert after.direction == before.direction
    assert after.confidence == pytest.approx(before.confidence)


def test_mock_ingest_payload_feeds_pipeline_without_crash():
    payload = {
        "prices": _integration_prices(n=85, seed=3002),
        "fng_scores": [45 + i % 10 for i in range(85)],
        "volumes": [1_000_000.0 + i * 1000.0 for i in range(85)],
    }
    event = run_pipeline(
        payload["prices"],
        fng_scores=payload["fng_scores"],
        volumes=payload["volumes"],
        asset="ETH",
    )
    assert event.asset == "ETH"
    assert event.direction in ("bullish", "bearish", "neutral")


def test_multi_asset_sequential_pipelines_do_not_pollute_state(sample_prices_60, sample_fng_60):
    events = [run_pipeline(sample_prices_60, fng_scores=sample_fng_60, asset=asset) for asset in ("BTC", "ETH", "SOL")]
    assert [event.asset for event in events] == ["BTC", "ETH", "SOL"]
    assert all(0.0 <= event.confidence <= 1.0 for event in events)


def test_concurrent_pipelines_different_assets_do_not_interfere():
    prices = [50_000.0 + i * 100 for i in range(90)]
    fng_scores = [50] * 90

    def run(asset):
        return run_pipeline(prices, fng_scores=fng_scores, asset=asset)

    with ThreadPoolExecutor(max_workers=2) as pool:
        btc, eth = list(pool.map(run, ["BTC", "ETH"]))

    assert btc.asset == "BTC"
    assert eth.asset == "ETH"
    assert btc.id != eth.id
    assert btc.direction in ("bullish", "bearish", "neutral")
    assert eth.direction in ("bullish", "bearish", "neutral")


def test_full_backtest_report_csv_is_valid(backtest_result):
    rows = list(csv.DictReader(StringIO(format_csv_trades(backtest_result))))
    assert len(rows) == 1
    assert rows[0]["direction"] == "long"
    assert rows[0]["regime_at_entry"] == "lv_up"


def test_full_backtest_metrics_json_is_valid(backtest_result):
    payload = json.dumps(asdict(backtest_result), default=str)
    decoded = json.loads(payload)
    assert decoded["asset"] == "BTC"
    assert decoded["total_trades"] == 1
    assert decoded["trades"][0]["closed"] is True


def test_training_data_consistency_with_fixed_inputs():
    prices = _integration_prices(n=90, seed=3003)
    fng_scores = [40 + i % 15 for i in range(90)]
    first = _build_training_data_window(prices, fng_scores, up_to=89)
    second = _build_training_data_window(prices, fng_scores, up_to=89)
    for left, right in zip(first, second):
        assert np.array_equal(left, right)
