"""Disk cache for trained XGBoost ensemble. Avoids retraining on every run.

Uses XGBoost's native binary format (not pickle) — safe to load, no code execution risk.
"""

import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(os.getenv("KAIROS_CACHE_DIR", str(Path.home() / ".kairos")))
# A cached model is considered stale after this many seconds regardless of candle count.
_MAX_CACHE_AGE_SECONDS = int(os.getenv("KAIROS_MODEL_CACHE_MAX_AGE", str(24 * 3600)))


def _paths(asset: str) -> tuple[Path, Path]:
    _CACHE_DIR.mkdir(exist_ok=True)
    base = str(_CACHE_DIR / f"model_{asset.lower()}")
    return Path(base + ".ubj"), Path(base + ".meta.json")


def _regime_path(asset: str, regime: str) -> Path:
    _CACHE_DIR.mkdir(exist_ok=True)
    return _CACHE_DIR / f"model_{asset.lower()}_{regime}.ubj"


def load_model(asset: str, n_candles: int):
    """Return a ready SignalEnsemble if cache is fresh, else None.

    Cache is rejected when:
    - Candle count differs by >= 50, OR
    - Wall-clock age exceeds _MAX_CACHE_AGE_SECONDS.
    """
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

    saved_at = m.get("saved_at", 0.0)
    if time.time() - saved_at > _MAX_CACHE_AGE_SECONDS:
        logger.debug("Model cache for %s expired (age %.0fs); will retrain", asset, time.time() - saved_at)
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
    """Persist ensemble to XGBoost native format + JSON metadata.

    Regime model files are written first; metadata is written last via an atomic
    rename so a crash between writes leaves no partially-valid cache state.
    """
    _, meta = _paths(asset)
    tmp_meta = meta.with_suffix(".meta.json.tmp")
    regimes = list(ensemble._models.keys())

    written: list[Path] = []
    try:
        for reg, model in ensemble._models.items():
            path = _regime_path(asset, reg)
            model.save_model(str(path))
            written.append(path)
        payload = {"n_candles": n_candles, "asset": asset, "regimes": regimes, "saved_at": time.time()}
        tmp_meta.write_text(json.dumps(payload))
        tmp_meta.replace(meta)  # atomic on POSIX; best-effort on Windows
    except Exception:
        # Clean up partial writes so a corrupt cache is not mistaken for a valid one.
        for p in written:
            p.unlink(missing_ok=True)
        if tmp_meta.exists():
            tmp_meta.unlink(missing_ok=True)
        raise
