from typing import Dict, Any
import numpy as np


def _sir_velocity(counts: np.ndarray, population: int) -> float:
    """
    Fit SIR model to observed post counts and return dI/dt at latest point.
    Infected = cumulative unique participants engaging with narrative.
    """
    if len(counts) < 3:
        return 0.0

    I = np.cumsum(counts).astype(float)
    S = population - I
    S = np.clip(S, 0, population)

    dI = np.diff(I)
    SI = S[:-1] * I[:-1] / population
    SI = np.where(SI > 0, SI, 1.0)
    betas = dI / SI
    beta = float(np.median(betas[betas > 0])) if (betas > 0).any() else 0.01

    latest_S = S[-1]
    latest_I = I[-1]
    dI_dt = beta * latest_S * latest_I / population
    # Normalize by population so velocity is per-capita rate, not raw count
    return float(dI_dt / population)


def compute_narrative_features(
    post_counts: list[int],
    population: int = 10_000,
) -> Dict[str, Any]:
    """
    Given time series of post/mention counts, compute:
    - narrative_velocity: rate of narrative spread (dI/dt normalized)
    - narrative_tipping_point: True if accelerating and < 50% saturation
    """
    counts = np.array(post_counts, dtype=float)
    velocity = _sir_velocity(counts, population)

    cumulative_infected = float(np.sum(counts))
    saturation = cumulative_infected / population
    acceleration = float(counts[-1] - counts[-2]) if len(counts) >= 2 else 0.0
    tipping_point = bool(acceleration > 0 and saturation < 0.5 and velocity > 0.1)

    return {
        "narrative_velocity": round(velocity, 4),
        "narrative_tipping_point": tipping_point,
        "saturation": round(saturation, 4),
    }
