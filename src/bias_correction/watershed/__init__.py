"""
Whole-watershed scalar bias correction.

Modules
-------
elasticity  — parallel GSSHA (baseline / +δ / -δ) → Q0, Q+, Q- per event
solver      — geometric-median Newton step → watershed scalar α_k
iterator    — full iterative loop with convergence tracking and artifact saving
"""

from src.bias_correction.watershed.elasticity import WatershedElasticityEstimator, EventResult
from src.bias_correction.watershed.solver import WatershedSolver, SolverResult
from src.bias_correction.watershed.iterator import WatershedIterator

__all__ = [
    "WatershedElasticityEstimator",
    "EventResult",
    "WatershedSolver",
    "SolverResult",
    "WatershedIterator",
]
