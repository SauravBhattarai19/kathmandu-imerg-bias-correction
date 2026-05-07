#!/usr/bin/env python3
"""
Post-Correction Validator
Applies α correction factors to every event's GAG file, re-runs GSSHA,
and computes before/after validation metrics against observed discharge.
"""

from __future__ import annotations

import time
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.gssha.runner import GsshaRunner
from src.gssha.gag_editor import GagFile, apply_alpha_corrections


class ValidationRunner:
    """Validate bias-corrected precipitation by re-running GSSHA."""

    def __init__(
        self,
        base_model_dir: str | Path,
        threads: int = 8,
        parallel_runs: int = 1,
    ):
        self.base_model_dir = Path(base_model_dir)
        self.threads = threads
        self.parallel_runs = parallel_runs
        self.runner = GsshaRunner(str(base_model_dir))

    def validate_corrections(
        self,
        gag_files: List[str],
        alpha_factors: np.ndarray,
        observed_discharge: Dict[str, pd.DataFrame],
        event_names: List[str],
        output_dir: str | Path,
        corrected_gags_dir: Optional[str | Path] = None,
    ) -> pd.DataFrame:
        """
        Apply α factors, run GSSHA for each event, compute metrics.

        Parameters
        ----------
        gag_files          : original uncorrected GAG file paths
        alpha_factors      : per-station correction factors
        observed_discharge : {event_name → DataFrame with 'Discharge_m3/s'}
        event_names        : event identifiers matching gag_files order
        output_dir         : where to save Validation_Report.csv and Validation_Plots.png
        corrected_gags_dir : where to write corrected .gag files
                             (defaults to output_dir if not specified)

        Returns
        -------
        DataFrame with columns:
            Event, Q_Observed, Q_Baseline, Q_Corrected,
            Error_Baseline, Error_Corrected,
            Bias_Baseline_%, Bias_Corrected_%, Improvement
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        gags_dir = Path(corrected_gags_dir) if corrected_gags_dir else output_dir
        gags_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'#'*70}")
        print(f"# POST-VALIDATION  |  {len(event_names)} events  |  "
              f"{self.parallel_runs} parallel processes")
        print(f"{'#'*70}\n")

        # 1. Create corrected GAG files
        corrected_gags = []
        for gag_file, event_name in zip(gag_files, event_names):
            out_gag = gags_dir / f"{event_name}_corrected.gag"
            apply_alpha_corrections(gag_file, alpha_factors.tolist(), str(out_gag))
            corrected_gags.append(str(out_gag))
            print(f"  ✓ Corrected GAG: {out_gag.name}")

        # 2. Build simulation jobs (baseline + corrected per event)
        jobs = []
        for gag_orig, gag_corr, name in zip(gag_files, corrected_gags, event_names):
            jobs.append((gag_orig, f"{name}_baseline",  self.base_model_dir, self.threads))
            jobs.append((gag_corr, f"{name}_corrected", self.base_model_dir, self.threads))

        # 3. Run simulations
        print(f"\nRunning {len(jobs)} simulations …")
        t_start = time.time()
        if self.parallel_runs > 1:
            with Pool(processes=self.parallel_runs) as pool:
                raw = pool.map(self._run_one, jobs)
        else:
            raw = [self._run_one(j) for j in jobs]
        elapsed = time.time() - t_start
        print(f"  Done in {elapsed:.1f}s")

        sim_dict = {}
        for sim_name, Q, error, _ in raw:
            if error:
                print(f"  ❌ {sim_name}: {error}")
            else:
                sim_dict[sim_name] = Q

        # 4. Compute metrics
        records = []
        for name in event_names:
            Q_base = sim_dict.get(f"{name}_baseline")
            Q_corr = sim_dict.get(f"{name}_corrected")
            obs_df = observed_discharge.get(name)
            Q_obs = obs_df["Discharge_m3/s"].max() if obs_df is not None and len(obs_df) else np.nan

            if Q_base is None or Q_corr is None:
                continue

            err_b = Q_base - Q_obs if not np.isnan(Q_obs) else np.nan
            err_c = Q_corr - Q_obs if not np.isnan(Q_obs) else np.nan
            bias_b = (Q_base / Q_obs - 1) * 100 if not np.isnan(Q_obs) else np.nan
            bias_c = (Q_corr / Q_obs - 1) * 100 if not np.isnan(Q_obs) else np.nan

            records.append({
                "Event": name,
                "Q_Observed": Q_obs,
                "Q_Baseline": Q_base,
                "Q_Corrected": Q_corr,
                "Error_Baseline": err_b,
                "Error_Corrected": err_c,
                "Bias_Baseline_%": bias_b,
                "Bias_Corrected_%": bias_c,
                "Improvement": abs(err_b) - abs(err_c) if not np.isnan(Q_obs) else np.nan,
            })

        df = pd.DataFrame(records)

        # 5. Print summary
        if len(df):
            rmse_b = np.sqrt(np.nanmean(df["Error_Baseline"] ** 2))
            rmse_c = np.sqrt(np.nanmean(df["Error_Corrected"] ** 2))
            print(f"\nOverall RMSE: {rmse_b:.2f} → {rmse_c:.2f} m³/s  "
                  f"(reduction {(1-rmse_c/rmse_b)*100:.1f}%)")

        # 6. Save
        report = output_dir / "Validation_Report.csv"
        df.to_csv(report, index=False)
        print(f"✓ Saved: {report}")

        self._create_plots(df, output_dir)
        return df

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _create_plots(self, df: pd.DataFrame, output_dir: Path) -> None:
        events = df["Event"].values
        x = np.arange(len(events))
        width = 0.35

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Station-Based Correction – Validation", fontsize=14, fontweight="bold")

        # Peak discharge comparison
        ax = axes[0, 0]
        ax.plot(x, df["Q_Observed"], "o-", color="black", label="Observed", lw=2, ms=7)
        ax.plot(x, df["Q_Baseline"], "s--", color="steelblue", label="Baseline", alpha=0.8)
        ax.plot(x, df["Q_Corrected"], "^-", color="tomato", label="Corrected", alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(events, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Peak Discharge (m³/s)"); ax.set_title("Peak Discharge")
        ax.legend(); ax.grid(True, alpha=0.3)

        # Absolute errors
        ax = axes[0, 1]
        ax.bar(x - width/2, df["Error_Baseline"], width, color="steelblue", alpha=0.6, label="Baseline")
        ax.bar(x + width/2, df["Error_Corrected"], width, color="tomato", alpha=0.6, label="Corrected")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(events, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Error (m³/s)"); ax.set_title("Absolute Errors")
        ax.legend(); ax.grid(True, alpha=0.3)

        # Scatter: obs vs sim
        ax = axes[1, 0]
        lim = max(df["Q_Observed"].max(), df[["Q_Baseline","Q_Corrected"]].max().max()) * 1.1
        ax.scatter(df["Q_Observed"], df["Q_Baseline"], s=80, marker="s", color="steelblue", alpha=0.7, label="Baseline")
        ax.scatter(df["Q_Observed"], df["Q_Corrected"], s=80, marker="^", color="tomato", alpha=0.7, label="Corrected")
        ax.plot([0, lim], [0, lim], "k-", alpha=0.3, label="1:1")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel("Observed (m³/s)"); ax.set_ylabel("Simulated (m³/s)")
        ax.set_title("Observed vs Simulated"); ax.legend(); ax.grid(True, alpha=0.3)

        # Relative bias
        ax = axes[1, 1]
        ax.bar(x - width/2, df["Bias_Baseline_%"], width, color="steelblue", alpha=0.6, label="Baseline")
        ax.bar(x + width/2, df["Bias_Corrected_%"], width, color="tomato", alpha=0.6, label="Corrected")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(events, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Relative Bias (%)"); ax.set_title("Relative Bias")
        ax.legend(); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out = output_dir / "Validation_Plots.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved: {out}")

    # ------------------------------------------------------------------
    # Multiprocessing helper
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
