#!/usr/bin/env python3
"""
GAG File Editor
Reads, modifies, and writes GSSHA rainfall GAG files with station-wise scaling.

GAG format reference:
    EVENT "description"
    NRPDS <n_timesteps>
    NRGAG <n_stations>
    COORD <easting> <northing> "<name>"   (repeated n_stations times)
    GAGES YYYY MM DD HH MM  v1  v2 ...   (repeated n_timesteps times)
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import List, Optional, Tuple


class GagFile:
    """Read, manipulate, and write a GSSHA GAG rainfall file."""

    def __init__(self, gag_path: str | Path):
        self.gag_path = Path(gag_path)
        self.event_description: str = ""
        self.num_periods: int = 0
        self.num_stations: int = 0
        self.station_coords: List[Tuple[float, float]] = []
        self.station_names: List[str] = []
        self.gages_data: List[Tuple[str, List[float]]] = []
        self._parse()

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse(self) -> None:
        with open(self.gag_path, "r") as fh:
            lines = fh.readlines()

        coord_start = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("EVENT"):
                self.event_description = stripped
            elif stripped.startswith("NRPDS"):
                self.num_periods = int(stripped.split()[1])
            elif stripped.startswith("NRGAG"):
                self.num_stations = int(stripped.split()[1])
                coord_start = i + 1
                break

        for i in range(self.num_stations):
            raw = lines[coord_start + i]
            parts = raw.split('"')
            coords = parts[0].split()[1:3]
            name = parts[1] if len(parts) > 1 else f"Station_{i + 1}"
            self.station_coords.append((float(coords[0]), float(coords[1])))
            self.station_names.append(name)

        gages_start = coord_start + self.num_stations
        for line in lines[gages_start:]:
            if line.startswith("GAGES"):
                parts = line.strip().split()
                timestamp = " ".join(parts[1:6])
                values = [float(v) for v in parts[6:]]
                self.gages_data.append((timestamp, values))

    # ------------------------------------------------------------------
    # Scaling helpers
    # ------------------------------------------------------------------

    def scale_station(self, station_idx: int, scale_factor: float) -> "GagFile":
        """Return a new GagFile with one station scaled."""
        new = self._shallow_copy()
        new.gages_data = [
            (ts, [v * scale_factor if j == station_idx else v for j, v in enumerate(vals)])
            for ts, vals in self.gages_data
        ]
        return new

    def scale_all_stations(self, scale_factors: List[float]) -> "GagFile":
        """Return a new GagFile with every station scaled by its own factor."""
        if len(scale_factors) != self.num_stations:
            raise ValueError(
                f"Expected {self.num_stations} scale factors, got {len(scale_factors)}"
            )
        new = self._shallow_copy()
        new.gages_data = [
            (ts, [v * sf for v, sf in zip(vals, scale_factors)])
            for ts, vals in self.gages_data
        ]
        return new

    def scale_uniform(self, scalar: float) -> "GagFile":
        """Return a new GagFile with all stations scaled by the same factor."""
        return self.scale_all_stations([scalar] * self.num_stations)

    def _shallow_copy(self) -> "GagFile":
        obj = GagFile.__new__(GagFile)
        obj.gag_path = self.gag_path
        obj.event_description = self.event_description
        obj.num_periods = self.num_periods
        obj.num_stations = self.num_stations
        obj.station_coords = self.station_coords.copy()
        obj.station_names = self.station_names.copy()
        obj.gages_data = []
        return obj

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def write(self, output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as fh:
            fh.write(f"{self.event_description}\n")
            fh.write(f"NRPDS {self.num_periods}\n")
            fh.write(f"NRGAG {self.num_stations}\n")
            for (x, y), name in zip(self.station_coords, self.station_names):
                fh.write(f'COORD {x:.2f} {y:.2f} "{name}"\n')
            for ts, vals in self.gages_data:
                vals_str = "  ".join(f"{v:.2f}" for v in vals)
                fh.write(f"GAGES {ts}  {vals_str}\n")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_total_rainfall_per_station(self) -> List[float]:
        """Cumulative rainfall [mm] for each station across the event."""
        totals = [0.0] * self.num_stations
        for _, vals in self.gages_data:
            for i, v in enumerate(vals):
                totals[i] += v
        return totals

    def __repr__(self) -> str:
        return (
            f"GagFile({self.gag_path.name}, "
            f"{self.num_stations} stations, {self.num_periods} periods)"
        )


# ------------------------------------------------------------------
# Module-level convenience functions
# ------------------------------------------------------------------

def create_perturbed_gag(
    input_gag_path: str | Path,
    station_idx: int,
    scale: float,
    output_dir: str | Path,
    suffix: Optional[str] = None,
) -> str:
    """
    Write a copy of *input_gag_path* with one station scaled.

    Parameters
    ----------
    input_gag_path : path to the original GAG file
    station_idx    : 0-based index of the station to perturb
    scale          : multiplicative factor (e.g. 1.10 for +10 %)
    output_dir     : directory to receive the new file
    suffix         : optional filename suffix (auto-generated if None)

    Returns
    -------
    Absolute path of the written perturbed GAG file.
    """
    gag = GagFile(input_gag_path)
    perturbed = gag.scale_station(station_idx, scale)
    base_name = Path(input_gag_path).stem
    if suffix is None:
        suffix = f"_S{station_idx + 1}_{scale:.2f}"
    out_path = Path(output_dir) / f"{base_name}{suffix}.gag"
    perturbed.write(out_path)
    return str(out_path)


def apply_alpha_corrections(
    input_gag_path: str | Path,
    alpha_factors: List[float],
    output_path: str | Path,
) -> str:
    """
    Apply station-wise correction factors *alpha_factors* and write result.

    Parameters
    ----------
    input_gag_path : original (uncorrected) GAG file
    alpha_factors  : one factor per station
    output_path    : destination for the corrected GAG file

    Returns
    -------
    The resolved output path string.
    """
    gag = GagFile(input_gag_path)
    corrected = gag.scale_all_stations(alpha_factors)
    corrected.write(output_path)
    return str(output_path)
