"""Disk cache for trained XGBoost ensemble. Avoids retraining on every run.

Uses XGBoost's native binary format (not pickle) — safe to load, no code execution risk.
"""
import json
from pathlib import Path

_CACHE_DIR = Path.home() / ".kairos"


def _paths(asset: str) -> tuple[Path, Path]:
    _CACHE_DIR.mkdir(exist_ok=True)
    base = str(_CACHE_DIR / f"model_{asset.lower()}")
    return Path(base + ".ubj"), Path(base + ".meta.json")


def load_model(asset: str, n_candles: int):
    """Return a ready SignalEnsemble if cache is fresh (within 50 candles), else None."""
    from kairos.signals.ensemble import SignalEnsemble
    ubj, meta = _paths(asset)
    if not ubj.exists() or not meta.exists():
        return None
    try:
        with open(meta) as f:
            m = json.load(f)
    except (json.JSONDecodeError, OSError, KeyError):
        return None
    if abs(n_candles - m.get("n_candles", 0)) >= 50:
        return None
    ensemble = SignalEnsemble()
    ensemble._model.load_model(str(ubj))
    ensemble._fitted = True
    return ensemble


def save_model(ensemble, asset: str, n_candles: int) -> None:
    """Persist ensemble to XGBoost native format + JSON metadata."""
    ubj, meta = _paths(asset)
    ensemble._model.save_model(str(ubj))
    with open(meta, "w") as f:
        json.dump({"n_candles": n_candles, "asset": asset}, f)
