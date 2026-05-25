#!/usr/bin/env python3
"""
Watershed Scalar Solver

Takes per-event GSSHA results (Q0, Q+, Q-) and observed peak discharge,
then computes the watershed-scale scalar correction α_k via a single
Newton-Raphson step in log-space.

Math
----
Central-difference elasticity (exact, not linearised):
    ε_e = [ln(Q+_e) − ln(Q-_e)] / [ln(1+δ) − ln(1-δ)]

Per-event correction (Newton step in log-space):
    α*_e = exp[(ln Q_obs_e − ln Q0_e) / ε_e]

Watershed scalar (geometric median — correct for log-normal quantities):
    α_k = exp(median({ln α*_e : e valid}))

Why geometric median and not arithmetic mean?
    Correction factors are multiplicative, so their natural space is
    log-space. The arithmetic mean of α*_e would over-weight large
    outliers. The geometric median is the L1-optimal estimator in
    log-space and is robust to outliers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from src.bias_correction.watershed.elasticity import EventResult

logger = logging.getLogger(__name__)


@dataclass
class SolverResult:
    alpha_k:              float                  # scalar for this iteration
    event_alphas:         Dict[str, float]       # α*_e per valid event
    event_elasticities:   Dict[str, float]       # ε_e per valid event
    n_valid:              int                    # events used in aggregation
    n_filtered:           int                    # events discarded
    log_alpha_std:        float                  # spread of ln(α*_e) — diagnostic
    filtered_reasons:     Dict[str, str] = field(default_factory=dict)


class WatershedSolver:
    """
    Aggregate per-event GSSHA perturbation runs into a single scalar α_k.
    """

    def __init__(
        self,
        delta: float = 0.10,
        min_elasticity: float = 0.10,
        alpha_bounds: Tuple[float, float] = (0.2, 5.0),
    ):
        self.delta = delta
        self.min_elasticity = min_elasticity
        self.alpha_lo, self.alpha_hi = alpha_bounds
        # Pre-compute denominator (constant for a given δ)
        self._log_denom = np.log(1.0 + delta) - np.log(1.0 - delta)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(
        self,
        event_results: List[EventResult],
        Q_obs: Dict[str, float],
    ) -> SolverResult:
        """
        Compute the watershed scalar α_k from per-event GSSHA results.

        Parameters
        ----------
        event_results : output of WatershedElasticityEstimator.compute()
        Q_obs         : {event_id: observed_peak_discharge_m3s}

        Returns
        -------
        SolverResult — α_k = 1.0 if no valid events (safe no-op).
        """
        log_alphas_valid:    list = []
        event_alphas:        Dict[str, float] = {}
        event_elasticities:  Dict[str, float] = {}
        filtered_reasons:    Dict[str, str]   = {}
        n_filtered = 0

        for er in event_results:
            skip_reason = self._check_event(er, Q_obs)
            if skip_reason:
                filtered_reasons[er.event_id] = skip_reason
                n_filtered += 1
                logger.debug("  skip %s — %s", er.event_id, skip_reason)
                continue

            q_obs = Q_obs[er.event_id]

            # Central-difference elasticity
            epsilon = (np.log(er.Q_plus) - np.log(er.Q_minus)) / self._log_denom

            if epsilon < self.min_elasticity:
                filtered_reasons[er.event_id] = f"low elasticity ε={epsilon:.3f}"
                n_filtered += 1
                logger.debug("  skip %s — low ε=%.3f", er.event_id, epsilon)
                continue

            # Newton step in log-space
            alpha_star = np.exp((np.log(q_obs) - np.log(er.Q0)) / epsilon)
            alpha_star = float(np.clip(alpha_star, self.alpha_lo, self.alpha_hi))

            event_alphas[er.event_id]       = alpha_star
            event_elasticities[er.event_id] = epsilon
            log_alphas_valid.append(np.log(alpha_star))

        if not log_alphas_valid:
            logger.error(
                "No valid events remain for aggregation — returning α_k = 1.0 (no change)."
            )
            return SolverResult(
                alpha_k=1.0,
                event_alphas={},
                event_elasticities={},
                n_valid=0,
                n_filtered=n_filtered,
                log_alpha_std=0.0,
                filtered_reasons=filtered_reasons,
            )

        arr = np.array(log_alphas_valid)
        alpha_k      = float(np.exp(np.median(arr)))   # geometric median
        log_alpha_std = float(np.std(arr))

        return SolverResult(
            alpha_k=alpha_k,
            event_alphas=event_alphas,
            event_elasticities=event_elasticities,
            n_valid=len(log_alphas_valid),
            n_filtered=n_filtered,
            log_alpha_std=log_alpha_std,
            filtered_reasons=filtered_reasons,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_event(er: EventResult, Q_obs: Dict[str, float]) -> str:
        """Return a reason string if the event must be skipped, else empty string."""
        if er.event_id not in Q_obs:
            return "no observed discharge"
        q_obs = Q_obs[er.event_id]
        if np.isnan(q_obs) or q_obs <= 0:
            return f"invalid Q_obs={q_obs}"
        if np.isnan(er.Q0) or er.Q0 <= 0:
            return f"baseline run failed (Q0={er.Q0})"
        if np.isnan(er.Q_plus) or er.Q_plus <= 0:
            return f"plus-delta run failed (Q+={er.Q_plus})"
        if np.isnan(er.Q_minus) or er.Q_minus <= 0:
            return f"minus-delta run failed (Q-={er.Q_minus})"
        return ""
