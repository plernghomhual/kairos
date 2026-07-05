import json
import logging
import os
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np

_logger = logging.getLogger(__name__)
from hmmlearn.hmm import GaussianHMM
from scipy.optimize import linear_sum_assignment

REGIME_LABELS = {
    0: "lv_up",
    1: "hv_up",
    2: "lv_down",
    3: "hv_down",
}

CANONICAL_REGIME_PATTERNS = {
    "lv_up": np.array([1.0, -1.0]),
    "hv_up": np.array([1.0, 1.0]),
    "lv_down": np.array([-1.0, -1.0]),
    "hv_down": np.array([-1.0, 1.0]),
}

N_COMPONENT_OPTIONS = (3, 4, 5)
COVARIANCE_OPTIONS = ("diag", "spherical", "full")
N_ITER_OPTIONS = (100, 200, 500)
RANDOM_STATE_OPTIONS = (42, 0, 123)
CACHE_MAX_AGE_SECONDS = 30 * 24 * 60 * 60
SMOOTHING_THRESHOLD = 0.6
_DOMINANCE_THRESHOLD = 0.80
_DIVERSITY_REJECT_THRESHOLD = 0.70


def _cache_path() -> Path:
    cache_dir = Path(os.getenv("KAIROS_CACHE_DIR", str(Path.home() / ".kairos")))
    return cache_dir / "hmm_params.json"


def _as_feature_matrix(features: np.ndarray) -> np.ndarray:
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("features must be a 2D matrix")
    if len(matrix) == 0:
        raise ValueError("features must contain at least one row")
    if not np.isfinite(matrix).all():
        raise ValueError("features must contain only finite values")
    return matrix


def _new_model(
    n_components: int,
    covariance_type: str,
    n_iter: int,
    random_state: int,
) -> GaussianHMM:
    return GaussianHMM(
        n_components=n_components,
        covariance_type=covariance_type,
        n_iter=n_iter,
        tol=0.01,
        random_state=random_state,
    )


def fit_regime_model(features: np.ndarray, n_states: int = 4) -> GaussianHMM:
    """
    Fit Gaussian HMM to feature matrix (returns, volatility).
    Keeps the legacy fixed-parameter API for existing callers.
    """
    matrix = _as_feature_matrix(features)
    model = _new_model(
        n_components=n_states,
        covariance_type="diag",
        n_iter=200,
        random_state=42,
    )
    _fit_quietly(model, matrix)
    _apply_variance_inflation(matrix, model)
    _attach_regime_labels(model)
    return model


def _is_dominated_fit(label_sequence: list[str]) -> bool:
    """True when one label dominates >80% of predictions (degenerate model)."""
    if not label_sequence:
        return True
    counts = Counter(label_sequence)
    return counts.most_common(1)[0][1] / len(label_sequence) > _DOMINANCE_THRESHOLD


_DIVERSITY_FALLBACK_PARAMS: dict[str, Any] = {
    "n_components": 4,
    "covariance_type": "full",
    "n_iter": 200,
    "random_state": 42,
}


def optimize_regime_model(features: np.ndarray) -> GaussianHMM:
    """
    Fit a Gaussian HMM using cached hyperparameters when fresh.

    Hyperparameters are grid-searched at most every 30 days and cached at
    ~/.kairos/hmm_params.json. The returned model is always fitted on the
    supplied features. If the fitted model is degenerate (one label dominates),
    falls back to the most expressive configuration.
    """
    matrix = _as_feature_matrix(features)
    cached_params = _load_cached_params(matrix)
    if cached_params is not None:
        try:
            model = _fit_with_params(matrix, cached_params)
            if not _is_dominated_fit(_predict_label_sequence(model, matrix)):
                return model
        except Exception as exc:
            _logger.debug("Cached HMM params failed; retraining: %s", exc)

    params = _select_best_params(matrix)
    model = _fit_with_params(matrix, params)

    if _is_dominated_fit(_predict_label_sequence(model, matrix)):
        model = _fit_with_params(matrix, _DIVERSITY_FALLBACK_PARAMS)

    _write_cached_params(params, matrix)
    return model


def predict_regime(
    model: GaussianHMM,
    features: np.ndarray,
    history: list[str] | None = None,
) -> str:
    """
    Predict current market regime using volatility x direction labels.

    Existing callers can continue passing only (model, features). Supplying
    history enables confidence-aware smoothing.
    """
    regime, _confidence = predict_regime_with_confidence(model, features, history)
    return regime


def predict_regime_with_confidence(
    model: GaussianHMM,
    features: np.ndarray,
    history: list[str] | None = None,
) -> tuple[str, float]:
    matrix = _as_feature_matrix(features)
    states = model.predict(matrix)
    state = int(states[-1])
    label_map = _get_regime_label_map(model)
    regime = label_map.get(state, _fallback_label_for_state(model, state))
    confidence = _last_state_confidence(model, matrix)
    if history:
        regime = smooth_regime(history, regime, alpha=confidence)
    return regime, confidence


def smooth_regime(
    history: list[str],
    new_prediction: str,
    alpha: float = 0.7,
) -> str:
    """Hold the last stable regime when a change has less than 60% support."""
    if not history:
        return new_prediction

    previous = history[-1]
    if previous == new_prediction:
        return new_prediction

    confidence = float(np.clip(alpha, 0.0, 1.0))
    recent = history[-5:]
    history_weight = recent.count(new_prediction) / len(recent)
    transition_score = confidence + (1.0 - confidence) * history_weight
    if transition_score < SMOOTHING_THRESHOLD:
        return previous
    return new_prediction


def _fit_with_params(features: np.ndarray, params: dict[str, Any]) -> GaussianHMM:
    model = _new_model(
        n_components=int(params["n_components"]),
        covariance_type=str(params["covariance_type"]),
        n_iter=int(params["n_iter"]),
        random_state=int(params["random_state"]),
    )
    _fit_quietly(model, features)
    _apply_variance_inflation(features, model)
    _attach_regime_labels(model)
    return model


def _select_best_params(features: np.ndarray) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []

    for n_components in N_COMPONENT_OPTIONS:
        for covariance_type in COVARIANCE_OPTIONS:
            results_by_iter: dict[int, list[dict[str, Any]]] = {n_iter: [] for n_iter in N_ITER_OPTIONS}
            for random_state in RANDOM_STATE_OPTIONS:
                for result in _fit_candidate_chain(
                    features,
                    n_components=n_components,
                    covariance_type=covariance_type,
                    random_state=random_state,
                ):
                    results_by_iter[int(result["n_iter"])].append(result)

            for n_iter in N_ITER_OPTIONS:
                seed_results = results_by_iter[n_iter]
                if not seed_results:
                    continue

                _add_label_variability(seed_results, features)

                # Hard diversity rejection: drop candidates where >70% labels are same regime.
                eligible = [r for r in seed_results if r.get("label_dominance", 1.0) <= _DIVERSITY_REJECT_THRESHOLD]
                if not eligible:
                    eligible = seed_results  # fallback: use all if none survive constraint

                candidates.append(
                    min(
                        eligible,
                        key=lambda item: (
                            item.get("label_dominance", 1.0),
                            item.get("transmat_penalty", 1.0),
                            item["label_variability"],
                            not item["converged"],
                            item["bic"],
                            item["aic"],
                        ),
                    )
                )

    if not candidates:
        return {
            "n_components": 4,
            "covariance_type": "diag",
            "n_iter": 200,
            "random_state": 42,
            "aic": float("inf"),
            "bic": float("inf"),
            "label_variability": 1.0,
            "converged": False,
        }

    best = min(
        candidates,
        key=lambda item: (
            item.get("label_dominance", 1.0),
            item.get("transmat_penalty", 1.0),
            not item["converged"],
            item["bic"],
            item["aic"],
            item["label_variability"],
        ),
    )
    return _serializable_params(best)


def _fit_candidate_chain(
    features: np.ndarray,
    *,
    n_components: int,
    covariance_type: str,
    random_state: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    model = _new_model(n_components, covariance_type, N_ITER_OPTIONS[0], random_state)
    previous_target = 0

    for n_iter in N_ITER_OPTIONS:
        try:
            if previous_target == 0:
                model.n_iter = n_iter
                _fit_quietly(model, features)
            elif not model.monitor_.converged:
                model.n_iter = n_iter - previous_target
                model.init_params = ""
                _fit_quietly(model, features)

            result = _evaluate_candidate(
                model,
                features,
                n_components=n_components,
                covariance_type=covariance_type,
                n_iter=n_iter,
                random_state=random_state,
            )
            if result is not None:
                results.append(result)
            previous_target = n_iter
        except Exception as exc:
            _logger.debug("HMM candidate fit failed: %s", exc)
            break

    return results


def _evaluate_candidate(
    model: GaussianHMM,
    features: np.ndarray,
    *,
    n_components: int,
    covariance_type: str,
    n_iter: int,
    random_state: int,
) -> dict[str, Any] | None:
    try:
        aic = float(model.aic(features))
        bic = float(model.bic(features))
        if not (np.isfinite(aic) and np.isfinite(bic)):
            return None
        _attach_regime_labels(model)
        return {
            "n_components": n_components,
            "covariance_type": covariance_type,
            "n_iter": n_iter,
            "random_state": random_state,
            "aic": aic,
            "bic": bic,
            "converged": bool(model.monitor_.converged),
            "label_sequence": _predict_label_sequence(model, features),
            "transmat_penalty": _transition_matrix_penalty(model.transmat_),
        }
    except Exception as exc:
        _logger.debug("HMM candidate evaluation failed: %s", exc)
        return None


def _apply_variance_inflation(features: np.ndarray, model: GaussianHMM) -> None:
    """Reduce self-transition bias when recent volatility is elevated.

    Computes the z-score of recent (last 20) volatility vs the full history.
    When z > 0.5, decreases diagonal self-transition probabilities and
    distributes the mass evenly across off-diagonal entries, forcing the
    HMM to consider state changes during volatile periods.
    """
    vol_col = features[:, 1] if features.shape[1] > 1 else np.abs(features[:, 0])
    full_mean = float(np.mean(vol_col))
    full_std = float(np.std(vol_col))
    if full_std == 0.0:
        return

    recent = vol_col[-20:] if len(vol_col) >= 20 else vol_col
    recent_mean = float(np.mean(recent))
    vol_z = (recent_mean - full_mean) / full_std

    if vol_z < 0.5:
        return

    inflation = min((vol_z - 0.5) * 0.15, 0.3)
    transmat = np.copy(model.transmat_)
    n = transmat.shape[0]

    for i in range(n):
        diag = transmat[i, i]
        reduction = diag * inflation
        transmat[i, i] = diag - reduction
        for j in range(n):
            if j != i:
                transmat[i, j] += reduction / (n - 1)

    transmat /= transmat.sum(axis=1, keepdims=True)
    model.transmat_ = transmat


def _fit_quietly(model: GaussianHMM, features: np.ndarray) -> None:
    logger = logging.getLogger("hmmlearn.base")
    previous_level = logger.level
    logger.setLevel(logging.ERROR)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(features)
    finally:
        logger.setLevel(previous_level)


def _transition_matrix_penalty(transmat: np.ndarray) -> float:
    """Mean diagonal of transition matrix. Higher = stickier HMM (rarely switches states)."""
    return float(np.mean(np.diag(np.asarray(transmat))))


def _add_label_variability(
    seed_results: list[dict[str, Any]],
    features: np.ndarray,
) -> None:
    sequences = [result["label_sequence"] for result in seed_results]
    n_rows = len(features)
    majority = []
    for row_idx in range(n_rows):
        row_labels = [sequence[row_idx] for sequence in sequences]
        majority.append(Counter(row_labels).most_common(1)[0][0])

    for result, sequence in zip(seed_results, sequences):
        disagreements = sum(label != majority_label for label, majority_label in zip(sequence, majority))
        result["label_variability"] = disagreements / n_rows
        counts = Counter(sequence)
        result["label_dominance"] = counts.most_common(1)[0][1] / len(sequence) if sequence else 1.0


def _predict_label_sequence(model: GaussianHMM, features: np.ndarray) -> list[str]:
    states = model.predict(features)
    label_map = _get_regime_label_map(model)
    return [label_map.get(int(state), _fallback_label_for_state(model, int(state))) for state in states]


def _attach_regime_labels(model: GaussianHMM) -> None:
    model._regime_label_map = _build_hungarian_label_map(model)


def _get_regime_label_map(model: GaussianHMM) -> dict[int, str]:
    label_map = getattr(model, "_regime_label_map", None)
    if label_map is None:
        label_map = _build_hungarian_label_map(model)
        model._regime_label_map = label_map
    return label_map


def _build_hungarian_label_map(model: GaussianHMM) -> dict[int, str]:
    try:
        means = _state_mean_patterns(model)
        labels = list(REGIME_LABELS.values())
        canonical = np.vstack([CANONICAL_REGIME_PATTERNS[label] for label in labels])
        canonical_norms = np.linalg.norm(canonical, axis=1, keepdims=True)
        canonical = canonical / np.where(canonical_norms == 0.0, 1.0, canonical_norms)
        correlation = np.clip(means @ canonical.T, -1.0, 1.0)
        cost = 1.0 - correlation
        row_ind, col_ind = linear_sum_assignment(cost)
        label_map = {int(state_idx): labels[int(label_idx)] for state_idx, label_idx in zip(row_ind, col_ind)}

        for state_idx in range(model.n_components):
            if state_idx not in label_map:
                nearest_label_idx = int(np.argmin(cost[state_idx]))
                label_map[state_idx] = labels[nearest_label_idx]
        return label_map
    except Exception as exc:
        _logger.debug("HMM label-map assignment failed; using volatility sort: %s", exc)
        return _volatility_sort_label_map(model)


def _state_mean_patterns(model: GaussianHMM) -> np.ndarray:
    ret_means = model.means_[:, 0]
    vol_means = model.means_[:, 1] if model.means_.shape[1] > 1 else model.means_[:, 0]
    state_vectors = np.column_stack(
        [
            _zscore(ret_means),
            _zscore(vol_means),
        ]
    )
    norms = np.linalg.norm(state_vectors, axis=1, keepdims=True)
    return np.divide(
        state_vectors,
        np.where(norms == 0.0, 1.0, norms),
        out=np.zeros_like(state_vectors),
        where=True,
    )


def _zscore(values: np.ndarray) -> np.ndarray:
    spread = float(np.std(values))
    if spread == 0.0:
        return np.zeros_like(values)
    return (values - float(np.mean(values))) / spread


def _volatility_sort_label_map(model: GaussianHMM) -> dict[int, str]:
    vol_means = model.means_[:, 1] if model.means_.shape[1] > 1 else model.means_[:, 0]
    ret_means = model.means_[:, 0]
    vol_order = np.argsort(vol_means)
    low_cutoff = max(1, int(np.ceil(model.n_components / 2)))
    low_vol_states = set(int(state) for state in vol_order[:low_cutoff])
    ret_threshold = float(np.median(ret_means))

    label_map = {}
    for state_idx in range(model.n_components):
        vol_prefix = "lv" if state_idx in low_vol_states else "hv"
        direction = "up" if ret_means[state_idx] >= ret_threshold else "down"
        label_map[state_idx] = f"{vol_prefix}_{direction}"
    return label_map


def _fallback_label_for_state(model: GaussianHMM, state: int) -> str:
    label = _volatility_sort_label_map(model).get(state, "lv_up")
    if label == "lv_up" and state not in _volatility_sort_label_map(model):
        _logger.warning("HMM state %d unmapped; defaulting to lv_up", state)
    return label


def _last_state_confidence(model: GaussianHMM, features: np.ndarray) -> float:
    probabilities = np.asarray(model.predict_proba(features)[-1], dtype=float)
    if probabilities.size == 1:
        return 1.0

    probabilities = np.clip(probabilities, 0.0, 1.0)
    total = probabilities.sum()
    if total <= 0.0:
        return 0.0
    probabilities = probabilities / total
    ranked = np.sort(probabilities)
    return float(np.clip(ranked[-1] - ranked[-2], 0.0, 1.0))


def _load_cached_params(features: np.ndarray) -> dict[str, Any] | None:
    path = _cache_path()
    if not path.exists():
        return None

    try:
        cached = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        _logger.warning("Ignoring unreadable HMM parameter cache %s: %s", path, exc)
        return None

    if time.time() - float(cached.get("optimized_at", 0.0)) > CACHE_MAX_AGE_SECONDS:
        return None
    if int(cached.get("feature_dim", -1)) != int(features.shape[1]):
        return None

    params = cached.get("params", {})
    required = {"n_components", "covariance_type", "n_iter", "random_state"}
    if not required.issubset(params):
        return None
    return params


def _write_cached_params(params: dict[str, Any], features: np.ndarray) -> None:
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "optimized_at": time.time(),
        "feature_dim": int(features.shape[1]),
        "params": _serializable_params(params),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _serializable_params(params: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "n_components",
        "covariance_type",
        "n_iter",
        "random_state",
        "aic",
        "bic",
        "label_variability",
        "converged",
    )
    return {key: _json_value(params[key]) for key in keys if key in params}


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value
