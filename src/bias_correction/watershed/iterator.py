#!/usr/bin/env python3
"""
Watershed Iterative Bias Correction Driver

Full Newton-Raphson loop in log-space:

    α_cum = 1.0
    for k = 1 … max_iterations:
        gags_k  = original_gags × α_cum          (always applied to originals)
        [parallel GSSHA: N baseline + N +δ + N -δ — all start simultaneously]
        Q0_e, Q+_e, Q-_e  →  ε_e, α*_e
        α_k = exp( median( ln α*_e ) )            (geometric median)
        performance_k = MAPE(Q0_e, Q_obs)         (free: from baseline runs)

        if k > 1 and performance_k > best_performance × 1.02:
            revert α_cum, STOP  (diverged)

        α_cum ×= α_k
        save iteration_0k/artifacts/{Elasticity.csv, Alpha_Scalar.json, Performance.csv}
        save iteration_0k/corrected_gags/*.gag

        if |α_k − 1| < convergence_threshold:
            STOP  (converged)

    write watershed_correction/Convergence.csv
    write watershed_correction/Summary.json

Key insight
-----------
The baseline run of iteration k (Q0_k) IS the post-correction validation
of iteration k-1 — no extra GSSHA runs needed.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.bias_correction.watershed.elasticity import WatershedElasticityEstimator, EventResult
from src.bias_correction.watershed.solver import WatershedSolver, SolverResult
from src.gssha.gag_editor import GagFile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _p(msg: str, **kw) -> None:
    """Print with immediate flush."""
    print(msg, flush=True, **kw)


def _hline(char: str = "─", width: int = 72) -> str:
    return char * width


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class WatershedIterator:
    """Drive the iterative watershed-scalar bias correction to convergence."""

    def __init__(
        self,
        base_model_dir: str | Path,
        original_gag_dir: str | Path,
        dhm_dir: str | Path,
        output_dir: str | Path,
        delta: float = 0.10,
        convergence_threshold: float = 0.02,
        max_iterations: int = 5,
        alpha_bounds: Tuple[float, float] = (0.2, 5.0),
        min_elasticity: float = 0.10,
        parallel_runs: int = 4,
        threads_per_run: int = 48,
        dry_run: bool = False,
    ):
        self.base_model_dir = Path(base_model_dir)
        self.original_gag_dir = Path(original_gag_dir)
        self.dhm_dir = Path(dhm_dir)
        self.output_dir = Path(output_dir)
        self.delta = delta
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations
        self.parallel_runs = parallel_runs
        self.threads_per_run = threads_per_run
        self.dry_run = dry_run

        self.estimator = WatershedElasticityEstimator(
            base_model_dir=base_model_dir,
            delta=delta,
            threads_per_run=threads_per_run,
            parallel_runs=parallel_runs,
        )
        self.solver = WatershedSolver(
            delta=delta,
            min_elasticity=min_elasticity,
            alpha_bounds=alpha_bounds,
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, event_ids: List[str], gag_files: List[str]) -> dict:
        """
        Execute the full iterative correction loop and write all outputs.

        Parameters
        ----------
        event_ids : flood event identifiers
        gag_files : original (uncorrected) .gag file paths, one per event_id

        Returns
        -------
        Summary dict with final_alpha_cum, n_iterations_completed, etc.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        _p("\n" + "═" * 72)
        _p("  WATERSHED ITERATIVE BIAS CORRECTION")
        _p("═" * 72)
        _p(f"  Events              : {len(event_ids)}")
        _p(f"  Max iterations      : {self.max_iterations}")
        _p(f"  Convergence (|α−1|<): {self.convergence_threshold}")
        _p(f"  Perturbation δ      : ±{int(self.delta*100)}%")
        _p(f"  Parallel GSSHA runs : {self.parallel_runs}")
        _p(f"  Threads / run       : {self.threads_per_run}")
        _p(f"  Total cores used    : {self.parallel_runs * self.threads_per_run}")
        _p(f"  GSSHA jobs/iteration: {len(event_ids)} events × 3 = {len(event_ids)*3}")
        if self.dry_run:
            _p("  MODE                : DRY-RUN (synthetic Q — no GSSHA)")
        _p(f"  Output dir          : {self.output_dir}")
        _p("═" * 72)

        # Load observed peaks once
        Q_obs = self._load_observed_peaks(event_ids)
        if not Q_obs:
            raise ValueError(
                "No observed discharge data. Run 01_extract_discharge.py first."
            )

        alpha_cum      = 1.0
        best_mape      = float("inf")
        best_alpha_cum = 1.0
        convergence_log: list = []
        stop_reason    = "max_iterations_reached"
        t_total_start  = time.time()

        for k in range(1, self.max_iterations + 1):
            t_iter_start = time.time()

            _p("\n" + "─" * 72)
            _p(f"  ITERATION {k:02d} / {self.max_iterations}"
               f"   |   α_cum = {alpha_cum:.6f}"
               f"   |   {datetime.now().strftime('%H:%M:%S')}")
            _p("─" * 72)

            # Create output directories for this iteration
            iter_dir           = self.output_dir / f"iteration_{k:02d}"
            artifacts_dir      = iter_dir / "artifacts"
            corrected_gags_dir = iter_dir / "corrected_gags"
            logs_dir           = iter_dir / "logs"
            for d in (artifacts_dir, corrected_gags_dir, logs_dir):
                d.mkdir(parents=True, exist_ok=True)

            # ---- Run GSSHA (all events × 3 scenarios simultaneously) ----
            _p(f"\n  [Step 1/3] Running GSSHA simulations …")
            sim_dir = iter_dir / "sim"   # baseline runs → permanent .otl here
            sim_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=f"wbc_iter{k:02d}_") as tmp:
                event_results = self.estimator.compute(
                    gag_files=gag_files,
                    event_ids=event_ids,
                    alpha_cum=alpha_cum,
                    work_dir=Path(tmp),
                    sim_dir=sim_dir,
                    dry_run=self.dry_run,
                )

            # Show where .otl files landed
            otl_files = list(sim_dir.glob("*/Khokana.otl"))
            if otl_files:
                _p(f"\n  Baseline .otl files ({len(otl_files)} events) → {sim_dir}")
                for otl in sorted(otl_files):
                    size_kb = otl.stat().st_size // 1024
                    _p(f"    {otl.parent.name}/Khokana.otl  ({size_kb} KB)")

            # ---- Performance of CURRENT α_cum (free from baseline runs) ----
            current_mape = self._compute_mape(event_results, Q_obs)
            failed_runs  = sum(1 for er in event_results if np.isnan(er.Q0))

            _p(f"\n  [Step 2/3] Performance @ current α_cum = {alpha_cum:.6f}")
            _p(f"  {'─'*50}")
            _p(f"  {'EVENT':<35}  {'Q_obs':>8}  {'Q_sim':>8}  {'ERR%':>7}")
            _p(f"  {'─'*35}  {'─'*8}  {'─'*8}  {'─'*7}")
            for er in event_results:
                q_obs = Q_obs.get(er.event_id, float("nan"))
                if not np.isnan(er.Q0) and not np.isnan(q_obs) and q_obs > 0:
                    err_pct = abs(er.Q0 - q_obs) / q_obs * 100.0
                    sym = "✓"
                else:
                    err_pct = float("nan")
                    sym = "✗"
                _p(f"  {sym} {er.event_id:<33}  {q_obs:8.2f}  {er.Q0:8.2f}"
                   f"  {err_pct:6.1f}%")
            _p(f"  {'─'*35}  {'─'*8}  {'─'*8}  {'─'*7}")
            _p(f"  MAPE = {current_mape:.2f}%   |   Failed runs: {failed_runs}/{len(event_ids)}")

            # ---- Divergence guard ----
            if k > 1 and current_mape > best_mape * 1.02:
                _p(f"\n  ⚠  DIVERGENCE: MAPE {current_mape:.2f}% > previous best "
                   f"{best_mape:.2f}% (>2% worse).")
                _p(f"  Reverting to α_cum = {best_alpha_cum:.6f} and stopping.")
                alpha_cum   = best_alpha_cum
                stop_reason = "divergence_detected"
                break

            best_mape      = current_mape
            best_alpha_cum = alpha_cum

            # ---- Newton step: compute α_k ----
            _p(f"\n  [Step 3/3] Computing α_k via Newton-Raphson in log-space …")
            solver_result = self.solver.solve(event_results, Q_obs)
            alpha_k        = solver_result.alpha_k
            alpha_cum_new  = alpha_cum * alpha_k

            # Print per-event elasticity table
            _p(f"\n  {'EVENT':<35}  {'ε':>7}  {'α*':>7}  {'NOTE'}")
            _p(f"  {'─'*35}  {'─'*7}  {'─'*7}  {'─'*20}")
            for er in event_results:
                eps  = solver_result.event_elasticities.get(er.event_id, float("nan"))
                ast  = solver_result.event_alphas.get(er.event_id, float("nan"))
                note = solver_result.filtered_reasons.get(er.event_id, "")
                sym  = "─" if note else "✓"
                _p(f"  {sym} {er.event_id:<33}  {eps:7.3f}  {ast:7.4f}  {note}")

            _p(f"\n  Aggregation (geometric median of {solver_result.n_valid} valid events):")
            _p(f"    α_k           = {alpha_k:.6f}")
            _p(f"    |α_k − 1|     = {abs(alpha_k - 1):.4f}  "
               f"(threshold = {self.convergence_threshold})")
            _p(f"    log(α*) std   = {solver_result.log_alpha_std:.4f}  "
               f"(spread of per-event corrections)")
            _p(f"    α_cum:  {alpha_cum:.6f}  →  {alpha_cum_new:.6f}")

            # ---- Save iteration artifacts ----
            self._save_elasticity(artifacts_dir, event_results, solver_result, Q_obs)
            self._save_alpha_scalar(artifacts_dir, k, alpha_k, alpha_cum_new, solver_result)
            self._save_performance(artifacts_dir, event_results, Q_obs, current_mape)
            self._write_corrected_gags(corrected_gags_dir, gag_files, event_ids, alpha_cum_new)

            t_iter = time.time() - t_iter_start
            convergence_log.append({
                "iteration":            k,
                "alpha_k":              round(alpha_k, 6),
                "alpha_cum":            round(alpha_cum_new, 6),
                "mape_percent":         round(current_mape, 4),
                "n_valid_events":       solver_result.n_valid,
                "n_filtered":           solver_result.n_filtered,
                "n_failed_runs":        failed_runs,
                "log_alpha_std":        round(solver_result.log_alpha_std, 6),
                "iteration_time_s":     round(t_iter, 1),
                "converged_this_iter":  abs(alpha_k - 1.0) < self.convergence_threshold,
            })

            alpha_cum = alpha_cum_new

            _p(f"\n  Iteration {k:02d} complete in {t_iter:.1f}s")

            # ---- Convergence check ----
            if abs(alpha_k - 1.0) < self.convergence_threshold:
                _p(f"\n  ✓ CONVERGED — |α_k − 1| = {abs(alpha_k - 1):.4f} "
                   f"< threshold {self.convergence_threshold}")
                stop_reason = "converged"
                break

        # ---- Write summary files ----
        _p("\n" + "─" * 72)
        _p("  Writing output files …")

        conv_df   = pd.DataFrame(convergence_log)
        conv_path = self.output_dir / "Convergence.csv"
        conv_df.to_csv(conv_path, index=False)
        _p(f"  ✓ Convergence log  → {conv_path}")

        summary = {
            "final_alpha_cum":        round(alpha_cum, 6),
            "n_iterations_completed": len(convergence_log),
            "final_mape_percent":     (
                round(convergence_log[-1]["mape_percent"], 4)
                if convergence_log else None
            ),
            "converged":              stop_reason == "converged",
            "stop_reason":            stop_reason,
            "total_time_s":           round(time.time() - t_total_start, 1),
            "timestamp_utc":          datetime.utcnow().isoformat(),
        }
        summary_path = self.output_dir / "Summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        _p(f"  ✓ Summary          → {summary_path}")

        # ---- Print convergence table ----
        _p("\n" + "═" * 72)
        _p("  CONVERGENCE SUMMARY")
        _p("─" * 72)
        _p(f"  {'ITER':>4}  {'α_k':>9}  {'α_cum':>9}  {'MAPE%':>7}  "
           f"{'VALID':>5}  {'TIME':>7}  STATUS")
        _p(f"  {'─'*4}  {'─'*9}  {'─'*9}  {'─'*7}  {'─'*5}  {'─'*7}  {'─'*12}")
        for row in convergence_log:
            status = "CONVERGED" if row["converged_this_iter"] else ""
            _p(
                f"  {row['iteration']:>4}  "
                f"{row['alpha_k']:>9.6f}  "
                f"{row['alpha_cum']:>9.6f}  "
                f"{row['mape_percent']:>7.2f}  "
                f"{row['n_valid_events']:>5}  "
                f"{row['iteration_time_s']:>6.0f}s  "
                f"{status}"
            )
        _p("─" * 72)
        _p(f"  Final α_cum  = {alpha_cum:.6f}")
        _p(f"  Stop reason  = {stop_reason}")
        _p(f"  Total time   = {summary['total_time_s']:.1f}s")
        _p("═" * 72)

        return summary

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_observed_peaks(self, event_ids: List[str]) -> Dict[str, float]:
        _p("\n--- Loading observed peak discharges ---")
        peaks: Dict[str, float] = {}
        for eid in event_ids:
            path = self.dhm_dir / f"discharge_{eid}.csv"
            if not path.exists():
                _p(f"  ⚠ {eid}: discharge file not found ({path.name})")
                continue
            df = pd.read_csv(path)
            q_col = next((c for c in df.columns if "Discharge" in c), None)
            if q_col is None:
                _p(f"  ⚠ {eid}: no Discharge column in {path.name}")
                continue
            q_peak = float(df[q_col].max())
            if np.isnan(q_peak) or q_peak <= 0:
                _p(f"  ⚠ {eid}: zero/NaN peak discharge")
                continue
            peaks[eid] = q_peak
            _p(f"  ✓ {eid}: Q_peak = {q_peak:.2f} m³/s")
        _p(f"  Loaded {len(peaks)}/{len(event_ids)} observed peaks.")
        return peaks

    @staticmethod
    def _compute_mape(
        event_results: List[EventResult],
        Q_obs: Dict[str, float],
    ) -> float:
        errors = []
        for er in event_results:
            q = Q_obs.get(er.event_id)
            if q and not np.isnan(er.Q0) and er.Q0 > 0 and q > 0:
                errors.append(abs(er.Q0 - q) / q * 100.0)
        return float(np.mean(errors)) if errors else float("inf")

    @staticmethod
    def _save_elasticity(
        artifacts_dir: Path,
        event_results: List[EventResult],
        solver_result: SolverResult,
        Q_obs: Dict[str, float],
    ) -> None:
        rows = []
        for er in event_results:
            rows.append({
                "event_id":     er.event_id,
                "Q_obs":        Q_obs.get(er.event_id, np.nan),
                "Q0":           er.Q0,
                "Q_plus":       er.Q_plus,
                "Q_minus":      er.Q_minus,
                "epsilon":      solver_result.event_elasticities.get(er.event_id, np.nan),
                "alpha_star":   solver_result.event_alphas.get(er.event_id, np.nan),
                "filtered":     er.event_id in solver_result.filtered_reasons,
                "filter_reason": solver_result.filtered_reasons.get(er.event_id, ""),
            })
        pd.DataFrame(rows).to_csv(artifacts_dir / "Elasticity.csv", index=False)

    @staticmethod
    def _save_alpha_scalar(
        artifacts_dir: Path,
        k: int,
        alpha_k: float,
        alpha_cum_new: float,
        solver_result: SolverResult,
    ) -> None:
        data = {
            "iteration":          k,
            "alpha_k":            round(alpha_k, 6),
            "alpha_cum_after":    round(alpha_cum_new, 6),
            "n_valid_events":     solver_result.n_valid,
            "n_filtered_events":  solver_result.n_filtered,
            "log_alpha_std":      round(solver_result.log_alpha_std, 6),
            "event_alphas":       {k: round(v, 4) for k, v in solver_result.event_alphas.items()},
            "event_elasticities": {k: round(v, 4) for k, v in solver_result.event_elasticities.items()},
            "filtered_reasons":   solver_result.filtered_reasons,
        }
        (artifacts_dir / "Alpha_Scalar.json").write_text(json.dumps(data, indent=2))

    @staticmethod
    def _save_performance(
        artifacts_dir: Path,
        event_results: List[EventResult],
        Q_obs: Dict[str, float],
        mape: float,
    ) -> None:
        rows = []
        for er in event_results:
            q = Q_obs.get(er.event_id, np.nan)
            err = (
                abs(er.Q0 - q) / q * 100.0
                if (q and not np.isnan(er.Q0) and q > 0)
                else np.nan
            )
            rows.append({
                "event_id":       er.event_id,
                "Q_obs_m3s":      round(q, 3) if not np.isnan(q) else np.nan,
                "Q_sim_m3s":      round(er.Q0, 3) if not np.isnan(er.Q0) else np.nan,
                "abs_pct_error":  round(err, 2) if not np.isnan(err) else np.nan,
            })
        df = pd.DataFrame(rows)
        summary_row = pd.DataFrame([{
            "event_id": "MEAN_MAPE", "Q_obs_m3s": np.nan,
            "Q_sim_m3s": np.nan, "abs_pct_error": round(mape, 4),
        }])
        pd.concat([df, summary_row], ignore_index=True).to_csv(
            artifacts_dir / "Performance.csv", index=False
        )

    @staticmethod
    def _write_corrected_gags(
        corrected_gags_dir: Path,
        gag_files: List[str],
        event_ids: List[str],
        alpha_cum: float,
    ) -> None:
        for event_id, orig_gag in zip(event_ids, gag_files):
            gag = GagFile(orig_gag)
            gag.scale_uniform(alpha_cum).write(corrected_gags_dir / f"{event_id}.gag")
        _p(f"  ✓ Saved {len(event_ids)} corrected GAG files → {corrected_gags_dir.name}/")
