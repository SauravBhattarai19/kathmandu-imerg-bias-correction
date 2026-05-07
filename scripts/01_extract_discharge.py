#!/usr/bin/env python3
"""
Step 1 – Stage → Discharge Conversion
======================================
Reads the full DHM stage time-series, applies the Khokana rating curve,
and exports:

    1. A complete discharge CSV  (Results/Discharge.csv)
    2. Per-event discharge CSVs  (Data/processed/DHM_Discharge/)

These per-event files are consumed by the bias-correction workflow.

Usage
-----
    python scripts/01_extract_discharge.py

Configuration
-------------
All paths and parameters are read from:
    configs/paths.yaml
    configs/system.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow project imports without installing as a package
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml

from src.preprocessing.stage_to_discharge import (
    RatingCurve,
    convert_stage_file,
    extract_event_discharge,
)

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------

with open(ROOT / "configs" / "paths.yaml") as fh:
    paths = yaml.safe_load(fh)


def _p(rel: str) -> Path:
    return (ROOT / rel).resolve()


STAGE_CSV     = _p(paths["data"]["raw"]["stage_csv"])
RATING_CURVE  = _p(paths["data"]["raw"]["rating_curve"])
DISCHARGE_ALL = _p(paths["data"]["processed"]["discharge_full"])
FLOOD_CSV     = _p(paths["data"]["processed"]["flood_csv"])
DHM_OUT_DIR   = _p(paths["data"]["processed"]["dhm_dir"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("STEP 1 – STAGE → DISCHARGE CONVERSION")
    print("=" * 70)

    # 1. Load rating curve
    rating = RatingCurve(RATING_CURVE)

    # 2. Convert full stage file
    discharge_df = convert_stage_file(
        stage_csv=STAGE_CSV,
        rating_curve=rating,
        output_csv=DISCHARGE_ALL,
    )

    # 3. Slice per flood event
    if not FLOOD_CSV.exists():
        print(f"\n⚠  Flood event CSV not found: {FLOOD_CSV}")
        print("   Run scripts/02_identify_floods.py first, or supply Data/flood_events.csv")
        return

    events_df = pd.read_csv(FLOOD_CSV).dropna(subset=["flood_id"])
    print(f"\nExtracting discharge for {len(events_df)} flood events …")

    DHM_OUT_DIR.mkdir(parents=True, exist_ok=True)

    ok, fail = 0, 0
    for _, row in events_df.iterrows():
        event_df = extract_event_discharge(
            discharge_df=discharge_df.rename(columns={"Discharge_m3s": "Discharge_m3s"}),
            start_date=str(row["precip_start_daily"]),
            end_date=str(row["precip_end_daily"]),
            event_id=row["flood_id"],
            output_dir=DHM_OUT_DIR,
        )
        if event_df.empty:
            fail += 1
        else:
            ok += 1

    print(f"\n{'='*70}")
    print(f"Done – {ok} events extracted, {fail} skipped (no data)")
    print(f"Per-event CSVs → {DHM_OUT_DIR}")
    print(f"Full discharge  → {DISCHARGE_ALL}")


if __name__ == "__main__":
    main()
