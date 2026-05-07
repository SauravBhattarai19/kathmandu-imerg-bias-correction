#!/usr/bin/env python3
"""
Stage → Discharge Conversion
Loads a DHM rating curve, interpolates/extrapolates discharge from stage
measurements, and extracts event-specific discharge time-series.

Unified from:
    Notebooks/Discharge/stage_to_discharge.py
    OLD Files/legacy_code/Stations Based Correction/Codes/extract_discharge_data.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


# ---------------------------------------------------------------------------
# Rating Curve
# ---------------------------------------------------------------------------

class RatingCurve:
    """Interpolates discharge from water level using a station rating curve."""

    def __init__(self, rating_file: str | Path):
        """
        Parameters
        ----------
        rating_file : plain-text rating table with comma-separated
                      (stage, discharge) pairs, one per line.
                      Lines not starting with a digit are ignored.
        """
        self.rating_file = Path(rating_file)
        self.stages, self.discharges = self._load(rating_file)
        self._interp = interp1d(
            self.stages,
            self.discharges,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        print(
            f"✓ Rating curve loaded: {len(self.stages)} points | "
            f"stage {self.stages.min():.2f}–{self.stages.max():.2f} m | "
            f"Q {self.discharges.min():.1f}–{self.discharges.max():.1f} m³/s"
        )

    def convert(self, stage: float) -> float:
        """
        Convert a single stage value to discharge.
        Returns NaN for missing (NaN / −9 999 990) inputs.
        """
        if pd.isna(stage) or stage == -9_999_990:
            return np.nan
        return max(0.0, float(self._interp(stage)))

    def convert_series(self, stage_series: pd.Series) -> pd.Series:
        """Vectorised conversion for a pandas Series."""
        result = self._interp(stage_series.values).astype(float)
        # Clamp negative extrapolation and mask sentinel values
        invalid = (stage_series.isna()) | (stage_series == -9_999_990)
        result[result < 0] = 0.0
        result[invalid.values] = np.nan
        return pd.Series(result, index=stage_series.index)

    @staticmethod
    def _load(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
        stages, discharges = [], []
        with open(path, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line or not line[0].isdigit():
                    continue
                if "," not in line:
                    continue
                try:
                    s, q = line.split(",")[:2]
                    stages.append(float(s))
                    discharges.append(float(q))
                except ValueError:
                    pass
        return np.array(stages), np.array(discharges)


# ---------------------------------------------------------------------------
# File-level helpers
# ---------------------------------------------------------------------------

def convert_stage_file(
    stage_csv: str | Path,
    rating_curve: RatingCurve,
    output_csv: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Read a DHM stage CSV, convert to discharge, optionally save.

    Expected columns: ``dateTime`` (M/D/YYYY HH:MM:SS), ``value`` (stage in m).

    Parameters
    ----------
    stage_csv    : path to the raw stage CSV
    rating_curve : pre-loaded RatingCurve instance
    output_csv   : if given, write the result here

    Returns
    -------
    DataFrame with columns [DateTime, Stage_m, Discharge_m3s]
    """
    df = pd.read_csv(stage_csv)
    df.columns = df.columns.str.strip()

    # Parse datetime — try multiple common formats
    for fmt in ("%m/%d/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M:%S"):
        try:
            df["DateTime"] = pd.to_datetime(df["dateTime"], format=fmt)
            break
        except Exception:
            pass
    else:
        df["DateTime"] = pd.to_datetime(df["dateTime"], infer_datetime_format=True, errors="coerce")

    df["Stage_m"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["DateTime"]).sort_values("DateTime").reset_index(drop=True)
    df["Discharge_m3s"] = rating_curve.convert_series(df["Stage_m"])

    result = df[["DateTime", "Stage_m", "Discharge_m3s"]]

    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_csv, index=False)
        print(f"✓ Saved discharge data → {output_csv}")

    print(
        f"✓ {len(result):,} records | "
        f"Q range {result['Discharge_m3s'].min():.1f}–{result['Discharge_m3s'].max():.1f} m³/s"
    )
    return result


def extract_event_discharge(
    discharge_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    event_id: str,
    output_dir: str | Path,
) -> pd.DataFrame:
    """
    Slice a full discharge time-series for one flood event and save to CSV.

    Parameters
    ----------
    discharge_df : full discharge DataFrame (columns: DateTime, Discharge_m3s)
    start_date   : 'YYYY-MM-DD' inclusive start
    end_date     : 'YYYY-MM-DD' inclusive end
    event_id     : identifier string used in the output filename
    output_dir   : directory to write ``discharge_{event_id}.csv``

    Returns
    -------
    Sliced DataFrame with columns [DateTime, Discharge_m3/s]
    """
    start_dt = pd.to_datetime(start_date + " 00:00:00")
    end_dt   = pd.to_datetime(end_date   + " 23:59:59")

    mask = (discharge_df["DateTime"] >= start_dt) & (discharge_df["DateTime"] <= end_dt)
    event_df = discharge_df.loc[mask, ["DateTime", "Discharge_m3s"]].copy()
    event_df.rename(columns={"Discharge_m3s": "Discharge_m3/s"}, inplace=True)

    if event_df.empty:
        print(f"  ⚠ {event_id}: no data in {start_date} – {end_date}")
        return event_df

    out_path = Path(output_dir) / f"discharge_{event_id}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    event_df.to_csv(out_path, index=False)

    q_peak = event_df["Discharge_m3/s"].max()
    print(f"  ✓ {event_id}: {len(event_df)} records | Q_peak = {q_peak:.2f} m³/s → {out_path.name}")
    return event_df
