#!/usr/bin/env python3
"""
CSV → GAG Converter
Reads an IMERG CSV (one column per station, one row per 30-min timestep)
and writes a GSSHA-format .gag file with UTM coordinates.

Unified from:
    OLD Files/legacy_code/Stations Based Correction/Codes/Get_IMERG_gag.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from pyproj import Transformer


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------

def convert_coordinates_to_utm(
    stations_df: pd.DataFrame,
    target_epsg: str = "EPSG:32645",
) -> Dict[str, dict]:
    """
    Convert WGS84 lat/lon to UTM easting/northing.

    Parameters
    ----------
    stations_df  : DataFrame with columns [station_id, latitude, longitude, name]
    target_epsg  : target CRS (default: UTM Zone 45N for Kathmandu)

    Returns
    -------
    {station_id: {easting, northing, name}}
    """
    transformer = Transformer.from_crs("EPSG:4326", target_epsg, always_xy=True)
    utm: Dict[str, dict] = {}
    for _, row in stations_df.iterrows():
        e, n = transformer.transform(row["longitude"], row["latitude"])
        utm[row["station_id"]] = {
            "easting": e,
            "northing": n,
            "name": row.get("name", row["station_id"]),
        }
    return utm


# ---------------------------------------------------------------------------
# CSV → GAG
# ---------------------------------------------------------------------------

def csv_to_gag(
    csv_path: str | Path,
    utm_coords: Dict[str, dict],
    output_dir: str | Path,
    flood_id: str,
) -> Optional[str]:
    """
    Convert one IMERG CSV to a GSSHA .gag file.

    The CSV must have:
        - A 'Date and Time' column (parseable datetime)
        - One column per station, named starting with the station_id prefix
          (e.g. 'TP001 - Nagarjun (Prativa's house)')

    Parameters
    ----------
    csv_path   : path to the IMERG CSV
    utm_coords : output of :func:`convert_coordinates_to_utm`
    output_dir : directory to write the .gag file
    flood_id   : used as the base filename (→ ``{flood_id}.gag``)

    Returns
    -------
    Full path of the written GAG file, or None on failure.
    """
    try:
        df = pd.read_csv(csv_path)
        df["Date and Time"] = pd.to_datetime(df["Date and Time"])

        # Map CSV column names to station_ids
        station_cols = [c for c in df.columns if c != "Date and Time"]
        col_to_station: Dict[str, str] = {}
        for col in station_cols:
            for sid in utm_coords:
                if col.startswith(sid):
                    col_to_station[col] = sid
                    break

        if not col_to_station:
            print(f"  ✗ {flood_id}: no matching stations found in CSV")
            return None

        # Build GAG content
        lines: List[str] = []
        start = df["Date and Time"].min()
        end   = df["Date and Time"].max()
        desc  = (f'EVENT "IMERG {flood_id} '
                 f'{start.strftime("%Y-%m-%d")} to {end.strftime("%Y-%m-%d")}"')
        lines.append(desc)
        lines.append(f"NRPDS {len(df)}")
        lines.append(f"NRGAG {len(col_to_station)}")

        for col, sid in col_to_station.items():
            c = utm_coords[sid]
            lines.append(f'COORD {c["easting"]:.2f} {c["northing"]:.2f} "{c["name"]}"')

        for _, row in df.iterrows():
            dt = row["Date and Time"]
            date_str = f"{dt.year:04d} {dt.month:02d} {dt.day:02d} {dt.hour:02d} {dt.minute:02d}"
            vals = "  ".join(
                f"{0.0 if pd.isna(row[col]) else row[col]:.2f}"
                for col in col_to_station
            )
            lines.append(f"GAGES {date_str}  {vals}")

        out_path = Path(output_dir) / f"{flood_id}.gag"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(lines) + "\n")
        print(f"  ✓ {out_path.name}")
        return str(out_path)

    except Exception as exc:
        print(f"  ✗ {flood_id}: {exc}")
        return None


def batch_convert_csvs(
    csv_dir: str | Path,
    utm_coords: Dict[str, dict],
    output_dir: str | Path,
) -> List[str]:
    """
    Convert every ``imerg_*.csv`` file in *csv_dir* to a GAG file.

    Returns
    -------
    List of successfully created GAG file paths.
    """
    csv_dir = Path(csv_dir)
    created = []
    for csv_file in sorted(csv_dir.glob("imerg_*.csv")):
        flood_id = csv_file.stem.replace("imerg_", "")
        gag_path = csv_to_gag(str(csv_file), utm_coords, output_dir, flood_id)
        if gag_path:
            created.append(gag_path)
    return created
