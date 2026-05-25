#!/usr/bin/env python3
"""
Step 0b – Flood Event Identification (Monsoon Only)
====================================================
Reads the full discharge time-series and selects the top-N independent
monsoon flood peaks (June–September). Saves the result to flood_data.csv,
which is consumed by all downstream pipeline steps.

Run this before scripts 01–03.

Usage
-----
    python scripts/00b_identify_floods.py
    python scripts/00b_identify_floods.py --n-events 15
    python scripts/00b_identify_floods.py --min-discharge 100
    python scripts/00b_identify_floods.py --spacing 7  # min days between peaks
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml

from src.preprocessing.flood_identifier import FloodIdentifier


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

with open(ROOT / "configs" / "paths.yaml") as fh:
    paths_cfg = yaml.safe_load(fh)


def _p(rel: str) -> Path:
    return (ROOT / rel).resolve()


DISCHARGE_CSV = _p(paths_cfg["data"]["processed"]["discharge_full"])
FLOOD_CSV     = _p(paths_cfg["data"]["processed"]["flood_csv"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(n_events: int, min_discharge: float, spacing_days: int) -> None:
    print("=" * 70)
    print("STEP 0b – MONSOON FLOOD IDENTIFICATION")
    print("=" * 70)

    if not DISCHARGE_CSV.exists():
        sys.exit(
            f"Full discharge CSV not found: {DISCHARGE_CSV}\n"
            "Run scripts/01_extract_discharge.py first."
        )

    print(f"\nLoading discharge data from: {DISCHARGE_CSV}")
    discharge_df = pd.read_csv(DISCHARGE_CSV)
    discharge_df["DateTime"] = pd.to_datetime(discharge_df["DateTime"])
    valid = discharge_df.dropna(subset=["Discharge_m3s"])
    print(f"  Records: {len(valid):,}  "
          f"({valid['DateTime'].min().date()} → {valid['DateTime'].max().date()})")

    print(f"\nIdentifying top-{n_events} monsoon flood peaks …")
    print(f"  Min discharge : {min_discharge} m³/s")
    print(f"  Min separation: {spacing_days} days between peaks")

    identifier = FloodIdentifier(
        min_peak_spacing_days=spacing_days,
        event_window_days=3,
        min_discharge=min_discharge,
    )

    events_df = identifier.identify(
        discharge_df=discharge_df,
        n_events=n_events,
        discharge_col="Discharge_m3s",
        datetime_col="DateTime",
        monsoon_only=True,
    )

    if events_df.empty:
        print("\n✗ No events identified. Check discharge data and thresholds.")
        sys.exit(1)

    # Show season breakdown
    print(f"\nSeason breakdown:")
    for season, grp in events_df.groupby("season"):
        print(f"  {season}: {len(grp)} events")

    print(f"\nYear breakdown:")
    for year, grp in events_df.groupby("year"):
        print(f"  {year}: {len(grp)} event(s)  "
              f"(peak Q: {grp['peak_discharge_m3s'].max():.0f} m³/s)")

    # Save — overwrite existing flood_data.csv
    identifier.save_event_list(events_df, FLOOD_CSV)

    print(f"\n{'='*70}")
    print(f"Done — {len(events_df)} monsoon flood events saved to:")
    print(f"  {FLOOD_CSV}")
    print("\nNext steps:")
    print("  python scripts/01_extract_discharge.py  # extract per-event discharge")
    print("  python scripts/02_download_imerg.py     # download IMERG for all events")
    print("  python scripts/03_run_bias_correction.py --mode watershed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Identify top-N monsoon flood events and save to flood_data.csv"
    )
    parser.add_argument(
        "--n-events", type=int, default=20,
        help="Number of flood events to select (default: 20)"
    )
    parser.add_argument(
        "--min-discharge", type=float, default=100.0,
        help="Minimum peak discharge threshold in m³/s (default: 100)"
    )
    parser.add_argument(
        "--spacing", type=int, default=10,
        help="Minimum days between two distinct event peaks (default: 10)"
    )
    args = parser.parse_args()

    main(
        n_events=args.n_events,
        min_discharge=args.min_discharge,
        spacing_days=args.spacing,
    )
