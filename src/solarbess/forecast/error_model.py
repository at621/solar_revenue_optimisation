"""Lead-time-scaled variance and Matern-like temporal correlation.

Per the simulator PDF (slide 6):
- forecast = truth + structured error
- error variance scales with lead time (shrinks as delivery approaches)
- shocks are correlated across nearby periods (not independent noise)
- forecast updates approximate a martingale around truth
"""

from __future__ import annotations

import numpy as np


def lead_time_sigma(
    lead_hours: np.ndarray, sigma_floor: float, sigma_max: float, tau_hours: float
) -> np.ndarray:
    """sigma(dt) = sigma_floor + (sigma_max - sigma_floor) * (1 - exp(-max(dt, 0) / tau))."""
    dt = np.maximum(lead_hours, 0.0)
    return sigma_floor + (sigma_max - sigma_floor) * (1.0 - np.exp(-dt / tau_hours))


def temporal_correlation(t_hours: np.ndarray, tau_hours: float) -> np.ndarray:
    """Matern-1/2 (exponential) correlation matrix: rho_ij = exp(-|t_i - t_j| / tau)."""
    diff = np.abs(t_hours[:, None] - t_hours[None, :])
    return np.exp(-diff / max(tau_hours, 1e-6))


def cholesky_jitter(matrix: np.ndarray, jitter: float = 1e-8, max_tries: int = 5) -> np.ndarray:
    """Robust Cholesky: retry with growing diagonal jitter on failure."""
    M = matrix.copy()
    for _ in range(max_tries):
        try:
            return np.linalg.cholesky(M)
        except np.linalg.LinAlgError:
            M = M + np.eye(M.shape[0]) * jitter
            jitter *= 10.0
    raise np.linalg.LinAlgError("Failed to compute Cholesky factor after retries.")


def sample_correlated_paths(
    rng: np.random.Generator,
    sigma: np.ndarray,           # (T,)
    corr: np.ndarray,            # (T, T)
    n_scenarios: int,
) -> np.ndarray:
    """Return shape (n_scenarios, T) zero-mean correlated draws with per-time sigma."""
    L = cholesky_jitter(corr)
    z = rng.standard_normal(size=(n_scenarios, sigma.shape[0]))
    paths = z @ L.T
    return paths * sigma[None, :]
