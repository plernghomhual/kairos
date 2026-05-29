"""Tests for kairos.model_cache — XGBoost native-format disk persistence."""
import pytest
import kairos.model_cache as mc
from kairos.signals.ensemble import SignalEnsemble, FeatureVector


def _fitted() -> SignalEnsemble:
    fv_b = FeatureVector(0.02, 1.5, 0.0, 0.5, True, 0.3, 1.0, 0.0, 0.0, 0.75, 0.9, 0.25)
    fv_r = FeatureVector(-0.02, -1.5, 0.1, 0.1, False, 0.5, 0.0, 1.0, 0.0, 0.25, 0.7, 0.5)
    e = SignalEnsemble()
    e.fit([fv_b] * 25 + [fv_r] * 25)
    return e


def test_missing_cache_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(mc, "_CACHE_DIR", tmp_path)
    assert mc.load_model("BTC", 100) is None


def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(mc, "_CACHE_DIR", tmp_path)
    ensemble = _fitted()
    mc.save_model(ensemble, "BTC", 100)
    loaded = mc.load_model("BTC", 100)
    assert loaded is not None
    fv = FeatureVector(0.01, 1.0, 0.0, 0.3, True, 0.3, 1.0, 0.0, 0.0, 0.6, 0.8, 0.25)
    result = loaded.predict("BTC", fv, citations=[], regime="accumulation")
    assert result.direction in ("bullish", "bearish")


def test_stale_cache_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(mc, "_CACHE_DIR", tmp_path)
    mc.save_model(_fitted(), "BTC", 100)
    assert mc.load_model("BTC", 160) is None  # delta = 60 >= 50


def test_fresh_cache_returns_ensemble(tmp_path, monkeypatch):
    monkeypatch.setattr(mc, "_CACHE_DIR", tmp_path)
    mc.save_model(_fitted(), "BTC", 100)
    assert mc.load_model("BTC", 140) is not None  # delta = 40 < 50


def test_separate_cache_per_asset(tmp_path, monkeypatch):
    monkeypatch.setattr(mc, "_CACHE_DIR", tmp_path)
    mc.save_model(_fitted(), "BTC", 100)
    assert mc.load_model("ETH", 100) is None  # different asset → no cache
