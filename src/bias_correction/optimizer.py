#!/usr/bin/env python3
"""
Station Optimizer
Solves for optimal per-station rainfall correction factors (α) using
regularised least squares with spatial smoothness and optional prior.

Formulation (log-space)
-----------------------
    min  ||E · x - b||²  +  λ ||L x||²  +  τ ||x - m||²

where
    E   = elasticity matrix          (n_events × n_stations)
    b   = ln(Q_target) - ln(Q0)     (n_events,)
    x   = ln(α)                     (n_stations,)
    L   = spatial Laplacian
    m   = prior in log-space (zeros → prior of α = 1)
    λ   = lambda_smooth
    τ   = tau_prior

Solution: α = exp( (EᵀE + λLᵀL + τI)⁻¹ (Eᵀb + τm) )
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


class StationOptimizer:
    """Regularised least-squares solver for per-station α correction factors."""

    def __init__(self, lambda_smooth: float = 0.3, tau_prior: float = 0.2):
        """
        Parameters
        ----------
        lambda_smooth : weight for spatial smoothness regularisation (Laplacian)
        tau_prior     : weight for prior constraint
        """
        self.lambda_smooth = lambda_smooth
        self.tau_prior = tau_prior

    def solve(
        self,
        elasticity_matrix: np.ndarray,
        Q0_vector: np.ndarray,
        Q_target_vector: np.ndarray,
        prior_alpha: Optional[np.ndarray] = None,
        station_coords: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Solve for optimal α correction factors.

        Parameters
        ----------
        elasticity_matrix : E, shape (n_events, n_stations)
        Q0_vector         : baseline discharges, shape (n_events,)
        Q_target_vector   : observed peak discharges, shape (n_events,)
        prior_alpha       : prior α per station (default: all ones)
        station_coords    : UTM coordinates, shape (n_stations, 2), for
                            distance-weighted Laplacian

        Returns
        -------
        (alpha_vector, diagnostics)
            alpha_vector : shape (n_stations,)
            diagnostics  : dict with residuals, RMSE before/after, etc.
        """
        n_events, n_stations = elasticity_matrix.shape
        print(f"\n{'='*70}")
        print(f"OPTIMIZING  |  {n_events} events × {n_stations} stations  |  "
              f"λ={self.lambda_smooth}, τ={self.tau_prior}")
        print(f"{'='*70}")

        E = elasticity_matrix
        b = np.log(Q_target_vector) - np.log(Q0_vector)

        for i in range(n_events):
            ratio = Q_target_vector[i] / Q0_vector[i]
            print(f"  Event {i+1}: Q0={Q0_vector[i]:.1f} → Q_obs={Q_target_vector[i]:.1f}"
                  f"  (ratio={ratio:.3f})")

        L = self._laplacian(n_stations, station_coords)
        m = np.log(prior_alpha) if prior_alpha is not None else np.zeros(n_stations)

        M = E.T @ E + self.lambda_smooth * L.T @ L + self.tau_prior * np.eye(n_stations)
        rhs = E.T @ b + self.tau_prior * m

        try:
            x = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            print("Warning: singular matrix – falling back to least-squares")
            x = np.linalg.lstsq(M, rhs, rcond=None)[0]

        alpha = np.exp(x)
        residual = E @ x - b
        Q_pred = Q0_vector * np.exp(E @ x)

        diagnostics = {
            "alpha": alpha,
            "x_log": x,
            "data_fit": float(np.linalg.norm(residual) ** 2),
            "smoothness_penalty": float(self.lambda_smooth * np.linalg.norm(L @ x) ** 2),
            "prior_penalty": float(self.tau_prior * np.linalg.norm(x - m) ** 2),
            "residual_norm": float(np.linalg.norm(residual)),
            "Q_predicted": Q_pred,
            "Q_target": Q_target_vector,
            "Q0": Q0_vector,
            "rmse_before": float(np.sqrt(np.mean((Q0_vector - Q_target_vector) ** 2))),
            "rmse_after": float(np.sqrt(np.mean((Q_pred - Q_target_vector) ** 2))),
        }

        print(f"\nOptimal α factors:")
        for i, a in enumerate(alpha):
            print(f"  Station {i+1}: α={a:.4f}  ({(a-1)*100:+.1f}%)")

        improvement = (1 - diagnostics["rmse_after"] / diagnostics["rmse_before"]) * 100
        print(f"\nRMSE: {diagnostics['rmse_before']:.2f} → {diagnostics['rmse_after']:.2f} "
              f"m³/s  (improvement {improvement:.1f}%)")
        print(f"{'='*70}\n")

        return alpha, diagnostics

    # ------------------------------------------------------------------
    # Laplacian helpers
    # ------------------------------------------------------------------

    def _laplacian(
        self,
        n_stations: int,
        station_coords: Optional[np.ndarray],
    ) -> np.ndarray:
        if station_coords is not None:
            return self._distance_weighted_laplacian(station_coords)
        # Simple first-order difference operator
        L = np.zeros((n_stations - 1, n_stations))
        for i in range(n_stations - 1):
            L[i, i] = -1
            L[i, i + 1] = 1
        return L

    def _distance_weighted_laplacian(self, coords: np.ndarray) -> np.ndarray:
        from scipy.spatial.distance import pdist, squareform

        D = squareform(pdist(coords))
        D[D == 0] = 1e-6
        W = 1.0 / D
        np.fill_diagonal(W, 0)
        return np.diag(W.sum(axis=1)) - W


def load_prior_from_csv(
    csv_path: str,
    column_name: str = "Gauge_IMERG_Ratio",
) -> Optional[np.ndarray]:
    """
    Load prior α factors from a CSV file (e.g. long-term gauge/IMERG ratios).

    Parameters
    ----------
    csv_path    : path to CSV with one row per station
    column_name : column containing the ratio values

    Returns
    -------
    numpy array of shape (n_stations,), or None if column not found
    """
    df = pd.read_csv(csv_path)
    if column_name in df.columns:
        return df[column_name].values
    print(f"Warning: '{column_name}' not in {csv_path} – no prior loaded")
    return None
