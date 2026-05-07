#!/usr/bin/env python3
"""
Elasticity Estimator
Computes discharge elasticity with respect to per-station rainfall perturbations.

Method
------
For each station s and flood event e:

    ε_{e,s} = [ln(Q+) - ln(Q-)] / [ln(1+δ) - ln(1-δ)]

where Q+ (Q-) is the peak discharge from a GSSHA run with station s
rainfall scaled by (1+δ) resp. (1-δ), and δ = ``delta`` (default 0.10).

The result is an elasticity matrix E of shape (n_events, n_stations).
"""

from __future__ import annotations

import shutil
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from src.gssha.runner import GsshaRunner
from src.gssha.gag_editor import GagFile, create_perturbed_gag


class ElasticityEstimator:
    """Compute discharge elasticity for station-wise rainfall perturbations."""

    def __init__(
        self,
        base_model_dir: str | Path,
        delta: float = 0.10,
        threads: int = 8,
        parallel_runs: int = 1,
    ):
        """
        Parameters
        ----------
        base_model_dir : path to the GSSHA base model directory
        delta          : fractional perturbation (0.10 → ±10 %)
        threads        : OpenMP threads per GSSHA run
        parallel_runs  : simultaneous GSSHA processes (for HPC)
        """
        self.base_model_dir = Path(base_model_dir)
        self.delta = delta
        self.threads = threads
        self.parallel_runs = parallel_runs
        self.runner = GsshaRunner(str(base_model_dir))

    # ------------------------------------------------------------------
    # Single-event elasticity
    # ------------------------------------------------------------------

    def compute_event_elasticity(
        self,
        gag_file: str | Path,
        num_stations: int,
        event_name: Optional[str] = None,
    ) -> Tuple[float, np.ndarray]:
        """
        Estimate elasticity vector for a single flood event.

        Returns
        -------
        (Q0_baseline, elasticity_vector)
            Q0              : peak discharge under unperturbed rainfall [m³/s]
            elasticity_vector : shape (num_stations,), dimensionless
        """
        gag_file = Path(gag_file)
        event_name = event_name or gag_file.stem
        temp_dir = tempfile.mkdtemp(prefix=f"elas_{event_name}_")

        print(f"\n{'='*70}")
        print(f"Elasticity: {event_name}  |  ±{self.delta*100:.0f}%  |  {num_stations} stations")
        print(f"{'='*70}")

        # Baseline
        t0 = time.time()
        Q0, _ = self.runner.run_simulation(str(gag_file), cleanup=False, threads=self.threads)
        print(f"  Baseline Q0 = {Q0:.2f} m³/s  ({time.time()-t0:.1f}s)")

        # Build perturbation jobs
        jobs = []
        for s in range(num_stations):
            gag_plus = create_perturbed_gag(str(gag_file), s, 1.0 + self.delta,
                                            temp_dir, f"_S{s+1}_plus")
            gag_minus = create_perturbed_gag(str(gag_file), s, 1.0 - self.delta,
                                             temp_dir, f"_S{s+1}_minus")
            jobs.append((gag_plus,  f"S{s+1}_plus",  self.base_model_dir, self.threads))
            jobs.append((gag_minus, f"S{s+1}_minus", self.base_model_dir, self.threads))

        # Run perturbations
        t_start = time.time()
        if self.parallel_runs > 1:
            with Pool(processes=self.parallel_runs) as pool:
                raw_results = pool.map(self._run_one, jobs)
        else:
            raw_results = [self._run_one(j) for j in jobs]

        elapsed = time.time() - t_start
        print(f"  {len(jobs)} perturbations in {elapsed:.1f}s  "
              f"(~{elapsed/len(jobs):.1f}s each)")

        # Parse results
        Q_map = {}
        for sim_name, Q, error, _ in raw_results:
            if error:
                print(f"  ✗ {sim_name}: {error}")
            else:
                Q_map[sim_name] = Q

        # Compute elasticity
        elasticity = np.zeros(num_stations)
        ln_den = np.log(1.0 + self.delta) - np.log(1.0 - self.delta)
        for s in range(num_stations):
            Q_plus  = Q_map.get(f"S{s+1}_plus")
            Q_minus = Q_map.get(f"S{s+1}_minus")
            if Q_plus is None or Q_minus is None:
                print(f"  ⚠ Station {s+1}: missing simulation → ε=0")
                continue
            elasticity[s] = (np.log(Q_plus) - np.log(Q_minus)) / ln_den
            print(f"  S{s+1}: ε={elasticity[s]:.4f}  (Q-={Q_minus:.1f}, Q+={Q_plus:.1f})")

        shutil.rmtree(temp_dir, ignore_errors=True)
        return Q0, elasticity

    # ------------------------------------------------------------------
    # Multi-event elasticity matrix
    # ------------------------------------------------------------------

    def compute_multi_event_elasticity(
        self,
        gag_files: List[str | Path],
        event_names: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute elasticity for every event in *gag_files*.

        Returns
        -------
        (Q0_vector, elasticity_matrix)
            Q0_vector         : shape (n_events,)
            elasticity_matrix : shape (n_events, n_stations)
        """
        n_events = len(gag_files)
        if event_names is None:
            event_names = [Path(g).stem for g in gag_files]

        first_gag = GagFile(gag_files[0])
        n_stations = first_gag.num_stations

        print(f"\n{'#'*70}")
        print(f"# MULTI-EVENT ELASTICITY  ({n_events} events, {n_stations} stations)")
        print(f"# Total simulations: {n_events * (2 * n_stations + 1)}")
        print(f"{'#'*70}\n")

        Q0_vector = np.zeros(n_events)
        E_matrix = np.zeros((n_events, n_stations))

        for idx, (gag_file, name) in enumerate(zip(gag_files, event_names)):
            print(f"\n*** EVENT {idx+1}/{n_events}: {name} ***")
            Q0, eps = self.compute_event_elasticity(str(gag_file), n_stations, name)
            Q0_vector[idx] = Q0
            E_matrix[idx, :] = eps

        print(f"\n{'#'*70}")
        print("# ELASTICITY COMPLETE")
        print(f"# Mean ε = {E_matrix.mean():.4f}  |  range "
              f"[{E_matrix.min():.4f}, {E_matrix.max():.4f}]")
        print(f"{'#'*70}\n")

        return Q0_vector, E_matrix

    # ------------------------------------------------------------------
    # Multiprocessing helper (must be top-level-picklable)
    # ------------------------------------------------------------------

    @staticmethod
    def _run_one(args: tuple) -> tuple:
        gag_file, sim_name, base_model_dir, threads = args
        try:
            runner = GsshaRunner(str(base_model_dir))
            Q, run_dir = runner.run_simulation(gag_file, cleanup=False, threads=threads)
            return sim_name, Q, None, run_dir
        except Exception as exc:
            return sim_name, None, str(exc), None
