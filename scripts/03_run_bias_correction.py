#!/usr/bin/env python3
"""
Step 3 – Station-Based IMERG Bias Correction
=============================================
Complete station-wise correction workflow:

    1. Load flood events and their .gag / discharge files.
    2. Compute discharge elasticity matrix (±δ perturbations via GSSHA).
    3. Solve for per-station α correction factors (regularised least squares).
    4. Apply corrections to every event's GAG file.
    5. Re-run GSSHA with corrected precipitation and validate.
    6. Save all results and diagnostic plots.

Usage
-----
    python scripts/03_run_bias_correction.py

    # Override parallelism:
    python scripts/03_run_bias_correction.py --threads 4 --parallel 12

    # Dry-run (skip GSSHA, use synthetic Q=1 data):
    python scripts/03_run_bias_correction.py --dry-run

Configuration
-------------
    configs/paths.yaml
    configs/system.yaml
    configs/stations.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yaml

from src.bias_correction.elasticity import ElasticityEstimator
from src.bias_correction.optimizer import StationOptimizer
from src.bias_correction.validator import ValidationRunner
from src.gssha.gag_editor import GagFile


# ---------------------------------------------------------------------------
# Load configs
# ---------------------------------------------------------------------------

def _load(name: str) -> dict:
    with open(ROOT / "configs" / name) as fh:
        return yaml.safe_load(fh)


paths_cfg   = _load("paths.yaml")
sys_cfg     = _load("system.yaml")
station_cfg = _load("stations.yaml")


def _p(rel: str) -> Path:
    return (ROOT / rel).resolve()


BASE_MODEL  = _p(paths_cfg["model"]["sbc"])
IMERG_DIR   = _p(paths_cfg["data"]["processed"]["imerg_gag_dir"])
DHM_DIR     = _p(paths_cfg["data"]["processed"]["dhm_dir"])
FLOOD_CSV   = _p(paths_cfg["data"]["processed"]["flood_csv"])

RUNS_ROOT   = _p(paths_cfg["outputs"]["runs_root"])
LATEST_LINK = _p(paths_cfg["outputs"]["latest_link"])


def _make_run_dirs(run_id: str) -> dict:
    """
    Create systematic output directories for a run.

    Layout:
      outputs/runs/<run_id>/
        station_correction/
          artifacts/
          corrected_gags/
          validation/
          figures/
          logs/
    """
    run_root = RUNS_ROOT / run_id / "station_correction"
    dirs = {
        "run_root": run_root,
        "artifacts": run_root / "artifacts",
        "corrected_gags": run_root / "corrected_gags",
        "validation": run_root / "validation",
        "figures": run_root / "figures",
        "logs": run_root / "logs",
    }
    for d in dirs.values():
        if isinstance(d, Path):
            d.mkdir(parents=True, exist_ok=True)

    # Point outputs/latest → this run (best-effort; symlink may fail on some FS)
    try:
        if LATEST_LINK.exists() or LATEST_LINK.is_symlink():
            LATEST_LINK.unlink()
        LATEST_LINK.symlink_to(run_root, target_is_directory=True)
    except Exception:
        pass
    return dirs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_flood_events(dry_run: bool = False):
    """Return (event_names, gag_files, n_stations, station_names, station_coords)."""
    flood_df = pd.read_csv(FLOOD_CSV).dropna(subset=["flood_id"])
    event_names = flood_df["flood_id"].tolist()

    gag_files = []
    for event in event_names:
        gag_path = IMERG_DIR / f"{event}.gag"
        if not gag_path.exists():
            print(f"  ⚠ Missing GAG: {gag_path.name}")
            continue
        gag_files.append(str(gag_path))

    # Station metadata from first GAG file
    first_gag = GagFile(gag_files[0])
    return (
        event_names,
        gag_files,
        first_gag.num_stations,
        first_gag.station_names,
        np.array(first_gag.station_coords),
    )


def load_observed_discharge(event_names: list) -> tuple:
    """Return (observed_dict, Q_target_vector)."""
    observed: dict = {}
    Q_target = []
    for event in event_names:
        discharge_file = DHM_DIR / f"discharge_{event}.csv"
        if discharge_file.exists():
            df = pd.read_csv(discharge_file)
            df["DateTime"] = pd.to_datetime(df["DateTime"])
            observed[event] = df
            Q_peak = df["Discharge_m3/s"].max()
            Q_target.append(Q_peak)
            print(f"  ✓ {event}: Q_peak = {Q_peak:.2f} m³/s")
        else:
            print(f"  ⚠ {event}: discharge file not found")
            Q_target.append(np.nan)
    return observed, np.array(Q_target)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(threads: int, parallel: int, dry_run: bool, run_id: str | None) -> bool:
    start = datetime.now()
    if run_id is None:
        run_id = start.strftime("%Y%m%d_%H%M%S")
    out = _make_run_dirs(run_id)

    print("\n" + "#" * 70)
    print("# STATION-BASED IMERG BIAS CORRECTION")
    print(f"# {start.strftime('%Y-%m-%d %H:%M:%S')}")
    if dry_run:
        print("# MODE: DRY-RUN (synthetic discharges, no GSSHA)")
    print(f"# Threads: {threads} per run | Parallel: {parallel}")
    print(f"# Run ID: {run_id}")
    print(f"# Outputs: {out['run_root']}")
    print("#" * 70)

    try:
        # ----------------------------------------------------------------
        # Load data
        # ----------------------------------------------------------------
        print("\n--- Loading flood events ---")
        event_names, gag_files, n_stations, station_names, station_coords = load_flood_events()
        print(f"Events: {len(event_names)}  |  Stations: {n_stations}")

        print("\n--- Loading observed discharge ---")
        observed_discharge, Q_target = load_observed_discharge(event_names)

        if np.all(np.isnan(Q_target)):
            print("✗ No observed discharge data found for any event.")
            print("  Run scripts/01_extract_discharge.py first.")
            return False

        # ----------------------------------------------------------------
        # Step 1: Elasticity
        # ----------------------------------------------------------------
        print("\n" + "=" * 70)
        print("STEP 1: ELASTICITY ESTIMATION")
        print("=" * 70)

        if dry_run:
            np.random.seed(42)
            Q0_vector = np.where(np.isnan(Q_target), 50.0, Q_target * 0.4)
            E_matrix  = np.random.uniform(0.1, 0.5, (len(event_names), n_stations))
        else:
            estimator = ElasticityEstimator(
                base_model_dir=BASE_MODEL,
                delta=sys_cfg["bias_correction"]["delta"],
                threads=threads,
                parallel_runs=parallel,
            )
            Q0_vector, E_matrix = estimator.compute_multi_event_elasticity(
                gag_files, event_names
            )

        # Save elasticity matrix
        elast_df = pd.DataFrame(
            E_matrix,
            index=event_names,
            columns=[f"Station_{i+1}" for i in range(n_stations)],
        )
        elast_df.insert(0, "Event", event_names)
        elast_df.insert(1, "Q0_Baseline", Q0_vector)
        elast_file = out["artifacts"] / "Elasticity_Matrix.csv"
        elast_df.to_csv(elast_file, index=False)
        print(f"✓ Elasticity matrix → {elast_file}")

        # ----------------------------------------------------------------
        # Step 2: Optimisation
        # ----------------------------------------------------------------
        print("\n" + "=" * 70)
        print("STEP 2: OPTIMISATION")
        print("=" * 70)

        optimizer = StationOptimizer(
            lambda_smooth=sys_cfg["bias_correction"]["lambda_smooth"],
            tau_prior=sys_cfg["bias_correction"]["tau_prior"],
        )

        # Filter out events without observed discharge
        valid_mask = ~np.isnan(Q_target)
        alpha_factors, diagnostics = optimizer.solve(
            elasticity_matrix=E_matrix[valid_mask],
            Q0_vector=Q0_vector[valid_mask],
            Q_target_vector=Q_target[valid_mask],
            station_coords=station_coords,
        )

        # Save alpha factors
        alpha_df = pd.DataFrame({
            "Station": station_names,
            "Station_Index": range(1, n_stations + 1),
            "Alpha": alpha_factors,
            "Correction_%": (alpha_factors - 1) * 100,
        })
        alpha_file = out["artifacts"] / "Alpha_Results.csv"
        alpha_df.to_csv(alpha_file, index=False)
        print(f"✓ Alpha factors → {alpha_file}")

        diag_save = {k: v.tolist() if isinstance(v, np.ndarray) else v
                     for k, v in diagnostics.items()}
        diag_file = out["artifacts"] / "Optimization_Diagnostics.json"
        diag_file.write_text(json.dumps(diag_save, indent=2))
        print(f"✓ Diagnostics → {diag_file}")

        # ----------------------------------------------------------------
        # Step 3: Validation
        # ----------------------------------------------------------------
        print("\n" + "=" * 70)
        print("STEP 3: POST-VALIDATION")
        print("=" * 70)

        if not dry_run:
            validator = ValidationRunner(
                base_model_dir=BASE_MODEL,
                threads=threads,
                parallel_runs=parallel,
            )
            results_df = validator.validate_corrections(
                gag_files=gag_files,
                alpha_factors=alpha_factors,
                observed_discharge=observed_discharge,
                event_names=event_names,
                output_dir=str(out["validation"]),
                corrected_gags_dir=str(out["corrected_gags"]),
            )
            print(f"\nValidation report → {out['validation'] / 'Validation_Report.csv'}")

        # ----------------------------------------------------------------
        # Summary
        # ----------------------------------------------------------------
        elapsed = datetime.now() - start
        print("\n" + "#" * 70)
        print("# WORKFLOW COMPLETE")
        print(f"# Elapsed: {elapsed}")
        print(f"# Outputs: {out['run_root']}")
        print("#" * 70)
        return True

    except Exception as exc:
        import traceback
        print(f"\n✗ WORKFLOW FAILED: {exc}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run station-based IMERG bias correction")
    parser.add_argument("--threads",  type=int, default=None, help="Threads per GSSHA run")
    parser.add_argument("--parallel", type=int, default=None, help="Parallel GSSHA runs")
    parser.add_argument("--dry-run",  action="store_true",    help="Skip GSSHA (synthetic data)")
    parser.add_argument("--run-id",   type=str, default=None, help="Run identifier (default: timestamp)")
    args = parser.parse_args()

    threads  = args.threads  or sys_cfg["gssha"]["threads_per_run"]
    parallel = args.parallel or sys_cfg["gssha"]["parallel_runs"]

    success = main(threads=threads, parallel=parallel, dry_run=args.dry_run, run_id=args.run_id)
    sys.exit(0 if success else 1)
