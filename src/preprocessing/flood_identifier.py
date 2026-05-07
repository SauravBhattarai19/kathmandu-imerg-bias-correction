#!/usr/bin/env python3
"""
Flood Event Identifier
Selects the top-N distinct flood peaks from a discharge time-series
and returns structured event metadata (dates, duration, season).

Unified from:
    Notebooks/IMERG/flood_event_detector.py
    Notebooks/Discharge/flood_identification.ipynb  (logic extracted)
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


MONSOON_MONTHS = {6, 7, 8, 9}
PRE_MONSOON_MONTHS = {3, 4, 5}
POST_MONSOON_MONTHS = {10, 11}
WINTER_MONTHS = {12, 1, 2}


def _season(month: int) -> str:
    if month in MONSOON_MONTHS:
        return "Monsoon"
    if month in PRE_MONSOON_MONTHS:
        return "Pre-Monsoon"
    if month in POST_MONSOON_MONTHS:
        return "Post-Monsoon"
    return "Winter"


class FloodIdentifier:
    """
    Identify the top-N independent flood events from a discharge time-series.

    Parameters
    ----------
    min_peak_spacing_days : minimum separation between two distinct event peaks
    event_window_days     : half-window around the peak to define event bounds
    min_discharge         : minimum discharge threshold [m³/s] to count as flood
    """

    def __init__(
        self,
        min_peak_spacing_days: int = 10,
        event_window_days: int = 3,
        min_discharge: float = 0.0,
    ):
        self.min_peak_spacing_days = min_peak_spacing_days
        self.event_window_days = event_window_days
        self.min_discharge = min_discharge

    def identify(
        self,
        discharge_df: pd.DataFrame,
        n_events: int = 25,
        discharge_col: str = "Discharge_m3s",
        datetime_col: str = "DateTime",
    ) -> pd.DataFrame:
        """
        Select the top-*n_events* independent flood peaks.

        Parameters
        ----------
        discharge_df   : full discharge time-series
        n_events       : number of events to select
        discharge_col  : column name for discharge values
        datetime_col   : column name for timestamps

        Returns
        -------
        DataFrame with one row per event:
            flood_id, flood_rank, peak_date, peak_discharge_m3s,
            start_date, end_date, precip_start_daily, precip_end_daily,
            duration_days, year, month, season
        """
        df = discharge_df[[datetime_col, discharge_col]].copy()
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        df = df.sort_values(datetime_col).reset_index(drop=True)

        # Remove sentinel / impossible values
        df = df[(df[discharge_col] >= self.min_discharge) &
                (df[discharge_col] < 1e5)].copy()

        min_gap = timedelta(days=self.min_peak_spacing_days)
        half_win = timedelta(days=self.event_window_days)

        events = []
        remaining = df.copy()

        for rank in range(1, n_events + 1):
            if remaining.empty:
                break
            idx_peak = remaining[discharge_col].idxmax()
            peak_row = remaining.loc[idx_peak]
            peak_dt = peak_row[datetime_col]
            peak_q  = peak_row[discharge_col]

            start_dt = peak_dt - half_win
            end_dt   = peak_dt + half_win

            # Sequential event ID using rank
            flood_id = f"event_{rank:03d}_{peak_dt.strftime('%Y%m%d')}_flood"

            events.append({
                "flood_id":             flood_id,
                "flood_rank":           rank,
                "peak_date":            peak_dt,
                "peak_discharge_m3s":   round(float(peak_q), 3),
                "start_date":           start_dt,
                "end_date":             end_dt,
                "precip_start_daily":   start_dt.date(),
                "precip_end_daily":     end_dt.date(),
                "duration_days":        (end_dt - start_dt).days,
                "year":                 peak_dt.year,
                "month":                peak_dt.month,
                "season":               _season(peak_dt.month),
            })

            # Suppress all records within min_gap of this peak
            mask = (remaining[datetime_col] >= (peak_dt - min_gap)) & \
                   (remaining[datetime_col] <= (peak_dt + min_gap))
            remaining = remaining[~mask]

        result = pd.DataFrame(events)
        print(f"✓ Identified {len(result)} flood events")
        for _, row in result.iterrows():
            print(f"  {row['flood_id']}: {row['peak_date'].date()}  "
                  f"Q={row['peak_discharge_m3s']:.1f} m³/s  ({row['season']})")
        return result

    def save_event_list(
        self,
        events_df: pd.DataFrame,
        output_path: str | Path,
    ) -> None:
        """Write the event list to CSV."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        events_df.to_csv(out, index=False)
        print(f"✓ Event list saved → {out}")
