"""Disk cache for trained XGBoost ensemble. Avoids retraining on every run.

Uses XGBoost's native binary format (not pickle) — safe to load, no code execution risk.
"""

import json
import os
from pathlib import Path

_CACHE_DIR = Path(os.getenv("KAIROS_CACHE_DIR", str(Path.home() / ".kairos")))


def _paths(asset: str) -> tuple[Path, Path]:
    _CACHE_DIR.mkdir(exist_ok=True)
    base = str(_CACHE_DIR / f"model_{asset.lower()}")
    return Path(base + ".ubj"), Path(base + ".meta.json")


def _regime_path(asset: str, regime: str) -> Path:
    _CACHE_DIR.mkdir(exist_ok=True)
    return _CACHE_DIR / f"model_{asset.lower()}_{regime}.ubj"


def load_model(asset: str, n_candles: int):
    """Return a ready SignalEnsemble if cache is fresh (within 50 candles), else None."""
    import xgboost as xgb

    from kairos.signals.ensemble import SignalEnsemble

    _, meta = _paths(asset)
    if not meta.exists():
        return None
    try:
        with open(meta) as f:
            m = json.load(f)
    except (json.JSONDecodeError, OSError, KeyError):
        return None
    if abs(n_candles - m.get("n_candles", 0)) >= 50:
        return None
    regimes = m.get("regimes", [])
    if not regimes:
        return None
    ensemble = SignalEnsemble()
    for reg in regimes:
        path = _regime_path(asset, reg)
        if not path.exists():
            return None
        model = xgb.XGBClassifier()
        model.load_model(str(path))
        ensemble._models[reg] = model
        ensemble._fitted[reg] = True
    return ensemble if ensemble._models else None


def save_model(ensemble, asset: str, n_candles: int) -> None:
    """Persist ensemble to XGBoost native format + JSON metadata."""
    _, meta = _paths(asset)
    regimes = list(ensemble._models.keys())
    for reg, model in ensemble._models.items():
        model.save_model(str(_regime_path(asset, reg)))
    with open(meta, "w") as f:
        json.dump({"n_candles": n_candles, "asset": asset, "regimes": regimes}, f)
