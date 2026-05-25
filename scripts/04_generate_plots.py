#!/usr/bin/env python3
"""
Step 4 – Generate Diagnostic Plots
====================================
Reads the bias correction outputs and produces publication-quality figures.

Station mode plots
------------------
    01_hydrograph_comparison.png   – observed / baseline / corrected peaks
    02_alpha_correction_map.png    – per-station α factors
    03_performance_summary.png     – RMSE, scatter, bias grid

Watershed mode plots
--------------------
    01_convergence.png             – α and MAPE per Newton-Raphson iteration
    02_watershed_performance.png   – peak discharge bars, scatter, error grid

Usage
-----
    # Watershed mode (default):
    python scripts/04_generate_plots.py

    # Station mode:
    python scripts/04_generate_plots.py --mode station

    # Specific run:
    python scripts/04_generate_plots.py --run-id 20260507_121736

Configuration
-------------
    configs/paths.yaml
    configs/stations.yaml
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import argparse
import numpy as np
import pandas as pd
import yaml

from src.visualization.plots import (
    plot_hydrographs,
    plot_alpha_map,
    plot_performance_summary,
    plot_watershed_convergence,
    plot_watershed_performance,
    plot_cluster_analysis,
)
from src.gssha.gag_editor import GagFile


def _load(name: str) -> dict:
    with open(ROOT / "configs" / name) as fh:
        return yaml.safe_load(fh)


paths_cfg   = _load("paths.yaml")
station_cfg = _load("stations.yaml")


def _p(rel: str) -> Path:
    return (ROOT / rel).resolve()


RUNS_ROOT   = _p(paths_cfg["outputs"]["runs_root"])
LATEST_LINK = _p(paths_cfg["outputs"]["latest_link"])


def _resolve_run_root(run_id: str | None, mode: str) -> Path | None:
    """
    Resolve the output folder for a run_id + mode combination.
    If run_id is None, use the outputs/latest symlink or newest timestamped run.
    Returns None if not found.
    """
    subdir = f"{mode}_correction"

    if run_id:
        p = (RUNS_ROOT / run_id / subdir).resolve()
        return p if p.exists() else None

    # Try latest symlink first
    if LATEST_LINK.exists():
        candidate = LATEST_LINK.resolve()
        # The symlink may point directly to a mode subdirectory or to a run root
        if candidate.name == subdir:
            return candidate
        mode_path = candidate / subdir
        if mode_path.exists():
            return mode_path

    # Fall back to newest timestamped run directory
    if RUNS_ROOT.exists():
        candidates = sorted(
            [p for p in RUNS_ROOT.iterdir() if p.is_dir()],
            key=lambda p: p.name,
            reverse=True,
        )
        for run_dir in candidates:
            mode_path = run_dir / subdir
            if mode_path.exists():
                return mode_path.resolve()

    return None


# ---------------------------------------------------------------------------
# Station mode
# ---------------------------------------------------------------------------

def _run_station(run_root: Path) -> None:
    artifacts_dir  = run_root / "artifacts"
    validation_dir = run_root / "validation"
    figures_dir    = run_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    validation_file = validation_dir / "Validation_Report.csv"
    alpha_file      = artifacts_dir  / "Alpha_Results.csv"

    if not validation_file.exists():
        print(f"⚠  {validation_file} not found.")
        print("   This run was likely a --dry-run (no GSSHA simulations).")
        print("   Re-run without --dry-run to produce a Validation_Report.")
        return
    if not alpha_file.exists():
        print(f"⚠  {alpha_file} not found – run Step 3 first.")
        return

    validation_df = pd.read_csv(validation_file)
    alpha_df      = pd.read_csv(alpha_file)

    alpha_factors  = alpha_df["Alpha"].values
    station_names  = [s["name"] for s in station_cfg["stations"]]
    station_coords = np.array(
        [[s["longitude"], s["latitude"]] for s in station_cfg["stations"]]
    )

    plot_hydrographs(
        validation_df,
        output_path=figures_dir / "01_hydrograph_comparison.png",
    )
    plot_alpha_map(
        alpha_factors,
        station_names,
        station_coords=station_coords,
        output_path=figures_dir / "02_alpha_correction_map.png",
    )
    plot_performance_summary(
        validation_df,
        output_path=figures_dir / "03_performance_summary.png",
    )
    print(f"\nAll figures saved → {figures_dir}")


# ---------------------------------------------------------------------------
# Watershed mode
# ---------------------------------------------------------------------------

def _plot_cluster(run_root: Path, perf_iterations: dict, figures_dir: Path) -> None:
    """Build cluster analysis plot from final iteration artifacts + IMERG GAG files."""
    if not perf_iterations:
        return
    final_iter   = max(perf_iterations.keys())
    alpha_path   = run_root / f"iteration_{final_iter:02d}" / "artifacts" / "Alpha_Scalar.json"
    imerg_gag_dir = _p(paths_cfg["data"]["processed"]["imerg_gag_dir"])

    if not alpha_path.exists():
        print(f"⚠  {alpha_path.name} not found — skipping cluster plot.")
        return

    with open(alpha_path) as fh:
        alpha_data = json.load(fh)
    event_alphas: dict = alpha_data.get("event_alphas", {})

    # Compute average IMERG total (mm) per event from GAG files
    imerg_totals: dict = {}
    for event_id in event_alphas:
        gag_path = imerg_gag_dir / f"{event_id}.gag"
        if gag_path.exists():
            try:
                gag = GagFile(gag_path)
                totals = gag.get_total_rainfall_per_station()
                imerg_totals[event_id] = float(np.mean(totals)) if totals else float("nan")
            except Exception:
                pass

    # Use final iteration Performance.csv; filter MEAN_MAPE row
    perf_df = perf_iterations[final_iter].copy()
    perf_df = perf_df[perf_df["event_id"] != "MEAN_MAPE"].copy()
    # Rename column for plot function if needed
    if "Q_obs_m3s" not in perf_df.columns and "Q_obs" in perf_df.columns:
        perf_df = perf_df.rename(columns={"Q_obs": "Q_obs_m3s"})
    # Add Q_obs_m3s from DHM discharge if not in performance CSV
    if "Q_obs_m3s" not in perf_df.columns:
        dhm_dir = _p(paths_cfg["data"]["processed"]["dhm_dir"])
        q_peaks = {}
        for _, row in perf_df.iterrows():
            eid = row["event_id"]
            f = dhm_dir / f"discharge_{eid}.csv"
            if f.exists():
                df = pd.read_csv(f)
                col = next((c for c in df.columns if "Discharge" in c), None)
                if col:
                    q_peaks[eid] = float(df[col].max())
        perf_df["Q_obs_m3s"] = perf_df["event_id"].map(q_peaks)

    if event_alphas and imerg_totals and not perf_df.empty:
        plot_cluster_analysis(
            perf_df=perf_df,
            event_alphas=event_alphas,
            imerg_totals=imerg_totals,
            output_path=figures_dir / "03_cluster_analysis.png",
        )
    else:
        print("⚠  Insufficient data for cluster plot — skipping.")


def _run_watershed(run_root: Path) -> None:
    figures_dir = run_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary_file     = run_root / "Summary.json"
    convergence_file = run_root / "Convergence.csv"

    if not convergence_file.exists():
        print(f"⚠  {convergence_file} not found.")
        print("   Run scripts/03_run_bias_correction.py --mode watershed first.")
        return

    convergence_df = pd.read_csv(convergence_file)
    summary        = json.loads(summary_file.read_text()) if summary_file.exists() else {}

    # Collect per-iteration Performance.csv files
    perf_iterations: dict = {}
    for iter_dir in sorted(run_root.glob("iteration_*")):
        perf_file = iter_dir / "artifacts" / "Performance.csv"
        if perf_file.exists():
            it_num = int(iter_dir.name.split("_")[1])
            perf_iterations[it_num] = pd.read_csv(perf_file)

    if not perf_iterations:
        print(f"⚠  No Performance.csv files found under {run_root}/iteration_*/artifacts/")
        print("   The run may have failed before completing any iteration.")
        return

    # Plot 1: Convergence
    plot_watershed_convergence(
        convergence_df,
        summary,
        output_path=figures_dir / "01_convergence.png",
    )

    # Plot 2: Performance grid
    plot_watershed_performance(
        perf_iterations,
        output_path=figures_dir / "02_watershed_performance.png",
    )

    # Plot 3: Cluster analysis (requires final Alpha_Scalar.json + IMERG GAG totals)
    _plot_cluster(run_root, perf_iterations, figures_dir)

    print(f"\nAll figures saved → {figures_dir}")

    # Print summary
    if summary:
        print(f"\n── Summary ──────────────────────────────")
        print(f"  α_final  : {summary.get('final_alpha_cum', '?'):.4f}")
        print(f"  MAPE     : {summary.get('final_mape_percent', '?'):.2f} %")
        print(f"  Converged: {summary.get('converged', '?')}")
        print(f"  Iterations: {summary.get('n_iterations_completed', '?')}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(mode: str, run_id: str | None) -> None:
    print("=" * 70)
    print(f"STEP 4 – GENERATING DIAGNOSTIC PLOTS  (mode={mode.upper()})")
    print("=" * 70)

    run_root = _resolve_run_root(run_id, mode)
    if run_root is None:
        label = f"run '{run_id}'" if run_id else "any run"
        print(f"⚠  No {mode}_correction directory found for {label} under {RUNS_ROOT}.")
        print(f"   Run scripts/03_run_bias_correction.py --mode {mode} first.")
        return

    print(f"Run directory: {run_root}\n")

    if mode == "watershed":
        _run_watershed(run_root)
    else:
        _run_station(run_root)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate diagnostic plots for a bias correction run")
    parser.add_argument(
        "--mode", choices=["watershed", "station"], default="watershed",
        help="Correction mode to plot (default: watershed)",
    )
    parser.add_argument(
        "--run-id", type=str, default=None,
        help="Run ID under outputs/runs/ (default: latest)",
    )
    args = parser.parse_args()
    main(mode=args.mode, run_id=args.run_id)
