#!/usr/bin/env python3
"""
Step 4 – Generate Diagnostic Plots
====================================
Reads the bias correction outputs and produces publication-quality figures:

    - Hydrograph comparison (observed / baseline / corrected)
    - Per-station α correction map
    - Performance summary (RMSE, scatter, bias)

Usage
-----
    python scripts/04_generate_plots.py

Configuration
-------------
    configs/paths.yaml
"""

from __future__ import annotations

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
)


def _load(name: str) -> dict:
    with open(ROOT / "configs" / name) as fh:
        return yaml.safe_load(fh)


paths_cfg  = _load("paths.yaml")
station_cfg = _load("stations.yaml")


def _p(rel: str) -> Path:
    return (ROOT / rel).resolve()


RUNS_ROOT   = _p(paths_cfg["outputs"]["runs_root"])
LATEST_LINK = _p(paths_cfg["outputs"]["latest_link"])


def _resolve_run_root(run_id: str | None) -> Path | None:
    """
    Resolve the systematic output folder for a given run_id.
    If run_id is None, try outputs/latest symlink; otherwise, pick newest run.
    Returns None if the resolved directory does not exist.
    """
    if run_id:
        p = (RUNS_ROOT / run_id / "station_correction").resolve()
        return p if p.exists() else None

    if LATEST_LINK.exists():
        return LATEST_LINK.resolve()

    if RUNS_ROOT.exists():
        candidates = sorted(
            [p for p in RUNS_ROOT.iterdir() if p.is_dir()],
            key=lambda p: p.name,
            reverse=True,
        )
        if candidates:
            return (candidates[0] / "station_correction").resolve()

    return None


def main(run_id: str | None) -> None:
    print("=" * 70)
    print("STEP 4 – GENERATING DIAGNOSTIC PLOTS")
    print("=" * 70)

    run_root = _resolve_run_root(run_id)
    if run_root is None:
        label = f"run '{run_id}'" if run_id else "any run"
        print(f"⚠  No output directory found for {label} under {RUNS_ROOT}.")
        print("   Run scripts/03_run_bias_correction.py first, then re-run this script.")
        return

    artifacts_dir = run_root / "artifacts"
    validation_dir = run_root / "validation"
    figures_dir = run_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # Load data
    # ----------------------------------------------------------------
    validation_file = validation_dir / "Validation_Report.csv"
    alpha_file      = artifacts_dir / "Alpha_Results.csv"

    if not validation_file.exists():
        print(f"⚠  {validation_file} not found.")
        print("   This run was likely a --dry-run (no GSSHA simulations).")
        print("   Re-run without --dry-run to produce a Validation_Report, then re-run this script.")
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

    # ----------------------------------------------------------------
    # Plot 1: Hydrograph comparison
    # ----------------------------------------------------------------
    plot_hydrographs(
        validation_df,
        output_path=figures_dir / "01_hydrograph_comparison.png",
    )

    # ----------------------------------------------------------------
    # Plot 2: Alpha factor map
    # ----------------------------------------------------------------
    plot_alpha_map(
        alpha_factors,
        station_names,
        station_coords=station_coords,
        output_path=figures_dir / "02_alpha_correction_map.png",
    )

    # ----------------------------------------------------------------
    # Plot 3: Performance summary
    # ----------------------------------------------------------------
    plot_performance_summary(
        validation_df,
        output_path=figures_dir / "03_performance_summary.png",
    )

    print(f"\nAll figures saved → {figures_dir}")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate diagnostic plots for a run")
    parser.add_argument("--run-id", type=str, default=None, help="Run ID under outputs/runs/ (default: latest)")
    args = parser.parse_args()
    main(run_id=args.run_id)
