#!/usr/bin/env python3
"""
Step 2 – Download IMERG and Generate GAG Files
===============================================
For every flood event in the event list:

    1. Download 30-min IMERG V07 precipitation at 7 station locations
       (via Google Earth Engine).
    2. Convert the CSV to a GSSHA-format .gag file with UTM coordinates.

The resulting .gag files are the direct rainfall input to GSSHA simulations.

Usage
-----
    python scripts/02_download_imerg.py

    # Process only specific events:
    python scripts/02_download_imerg.py --events event_001 event_002

Prerequisites
-------------
    - GEE authenticated:  earthengine authenticate
    - Python packages:    earthengine-api, pyproj

Configuration
-------------
    configs/paths.yaml   (paths)
    configs/system.yaml  (GEE project)
    configs/stations.yaml (station coordinates)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml

from src.data_acquisition.imerg_downloader import IMERGDownloader
from src.data_acquisition.gag_converter import (
    convert_coordinates_to_utm,
    batch_convert_csvs,
)


# ---------------------------------------------------------------------------
# Load configs
# ---------------------------------------------------------------------------

def _load(name: str) -> dict:
    with open(ROOT / "configs" / name) as fh:
        return yaml.safe_load(fh)


paths_cfg    = _load("paths.yaml")
sys_cfg      = _load("system.yaml")
station_cfg  = _load("stations.yaml")


def _p(rel: str) -> Path:
    return (ROOT / rel).resolve()


FLOOD_CSV    = _p(paths_cfg["data"]["processed"]["flood_csv"])
IMERG_DIR    = _p(paths_cfg["data"]["processed"]["imerg_gag_dir"])
TEMP_CSV_DIR = _p("Data/processed/temp_imerg_csv")
# Optional: only needed if you want to override the downloader default.
# (Older configs may not have a `gee:` section.)
GEE_PROJECT  = (sys_cfg.get("gee") or {}).get("project")

# Build stations DataFrame from YAML
stations_df = pd.DataFrame(station_cfg["stations"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(filter_events: list[str] | None = None) -> None:
    print("=" * 70)
    print("STEP 2 – DOWNLOAD IMERG & GENERATE GAG FILES")
    print("=" * 70)

    # Load flood events
    if not FLOOD_CSV.exists():
        sys.exit(f"Flood event list not found: {FLOOD_CSV}\n"
                 "Run scripts/01_extract_discharge.py first.")

    events_df = pd.read_csv(FLOOD_CSV).dropna(subset=["flood_id"])
    if filter_events:
        events_df = events_df[events_df["flood_id"].isin(filter_events)]

    print(f"\nEvents to process: {len(events_df)}")

    # Prepare UTM coordinates
    utm_coords = convert_coordinates_to_utm(
        stations_df,
        station_cfg["utm_projection"],
    )

    # Download IMERG CSVs
    TEMP_CSV_DIR.mkdir(parents=True, exist_ok=True)
    downloader = IMERGDownloader(gee_project=GEE_PROJECT) if GEE_PROJECT else IMERGDownloader()
    if not downloader.initialise_gee():
        sys.exit("GEE initialisation failed. Run: earthengine authenticate")

    print(f"\nDownloading to temporary CSV dir: {TEMP_CSV_DIR}")
    download_results = downloader.download_all_events(
        events_df=events_df,
        stations_df=stations_df,
        output_dir=TEMP_CSV_DIR,
    )

    # Convert CSVs → GAG files
    # mm/hr → mm per timestep conversion is handled inside batch_convert_csvs
    print(f"\nConverting CSVs to GAG format …")
    IMERG_DIR.mkdir(parents=True, exist_ok=True)
    created = batch_convert_csvs(TEMP_CSV_DIR, utm_coords, IMERG_DIR)

    # Clean up temporary CSVs
    import shutil
    if TEMP_CSV_DIR.exists():
        shutil.rmtree(TEMP_CSV_DIR, ignore_errors=True)

    print(f"\n{'='*70}")
    n_ok = sum(download_results.values())
    print(f"Downloaded: {n_ok}/{len(events_df)} events")
    print(f"GAG files created: {len(created)}")
    print(f"Output directory: {IMERG_DIR}")

    if created:
        print("\nCreated GAG files:")
        for g in sorted(created):
            print(f"  {Path(g).name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download IMERG and generate GAG files")
    parser.add_argument(
        "--events", nargs="+",
        help="Event IDs to process (default: all events in flood_data.csv)"
    )
    args = parser.parse_args()
    main(filter_events=args.events)
