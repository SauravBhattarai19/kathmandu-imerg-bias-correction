#!/usr/bin/env python3
"""
IMERG Downloader
Downloads 30-minute NASA GPM IMERG V07 precipitation at point locations
via Google Earth Engine (GEE) and saves per-event CSV files.

Unified from:
    Notebooks/IMERG/enhanced_imerg_downloader.py
    Notebooks/IMERG/imerg_data_download.py

Key fix: Nepal is UTC+5:45; all downloads use UTC timestamps.

Usage
-----
    from src.data_acquisition.imerg_downloader import IMERGDownloader
    dl = IMERGDownloader(gee_project="ee-sauravbhattarai1999")
    dl.download_event("event_001_20130523_flood", "2013-05-21 00:00:00",
                      "2013-05-25 23:45:00", stations_df, output_dir)
"""

from __future__ import annotations

import logging
import time
from datetime import timedelta
from pathlib import Path
from typing import Optional

import pandas as pd


def nepal_to_utc(nepal_datetime: pd.Timestamp) -> pd.Timestamp:
    """Convert Nepal Standard Time (UTC+5:45) to UTC."""
    return nepal_datetime - timedelta(hours=5, minutes=45)


class IMERGDownloader:
    """
    Download IMERG half-hourly precipitation for a set of point locations.

    Parameters
    ----------
    gee_project : GEE project identifier (e.g. 'ee-sauravbhattarai1999')
    log_level   : Python logging level
    """

    IMERG_COLLECTION = "NASA/GPM_L3/IMERG_V07"
    BAND = "precipitation"

    def __init__(
        self,
        gee_project: str = "ee-sauravbhattarai1999",
        log_level: int = logging.INFO,
    ):
        self.gee_project = gee_project
        self._gee_initialised = False
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s  %(levelname)s  %(message)s",
        )
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # GEE initialisation
    # ------------------------------------------------------------------

    def initialise_gee(self) -> bool:
        """Authenticate and initialise the Earth Engine API."""
        try:
            import ee  # deferred import – ee not required at import time
            ee.Initialize(project=self.gee_project)
            self._gee_initialised = True
            self.logger.info(f"✓ GEE initialised ({self.gee_project})")
            return True
        except Exception as exc:
            self.logger.error(f"✗ GEE initialisation failed: {exc}")
            return False

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download_event(
        self,
        event_id: str,
        start_nepal: str,
        end_nepal: str,
        stations_df: pd.DataFrame,
        output_dir: str | Path,
        retry_delay: float = 2.0,
    ) -> pd.DataFrame:
        """
        Download IMERG precipitation for one flood event.

        Parameters
        ----------
        event_id     : e.g. 'event_001_20130523_flood'
        start_nepal  : 'YYYY-MM-DD HH:MM:SS' in Nepal time
        end_nepal    : 'YYYY-MM-DD HH:MM:SS' in Nepal time
        stations_df  : DataFrame with columns [station_id, latitude, longitude]
        output_dir   : directory to save ``imerg_{event_id}.csv``
        retry_delay  : seconds to sleep between consecutive API calls

        Returns
        -------
        DataFrame with timestamp column and one column per station_id
        (empty DataFrame on failure)
        """
        import ee

        if not self._gee_initialised:
            if not self.initialise_gee():
                return pd.DataFrame()

        # Convert Nepal → UTC
        start_utc = nepal_to_utc(pd.to_datetime(start_nepal))
        end_utc   = nepal_to_utc(pd.to_datetime(end_nepal))

        self.logger.info(f"{'='*60}")
        self.logger.info(f"Event: {event_id}")
        self.logger.info(f"Nepal : {start_nepal}  →  {end_nepal}")
        self.logger.info(f"UTC   : {start_utc}  →  {end_utc}")

        # Build GEE station feature collection
        features = []
        for _, row in stations_df.iterrows():
            pt = ee.Feature(
                ee.Geometry.Point([row["longitude"], row["latitude"]]),
                {"station_id": row["station_id"]},
            )
            features.append(pt)
        fc = ee.FeatureCollection(features)

        # Load IMERG collection
        collection = (
            ee.ImageCollection(self.IMERG_COLLECTION)
            .filterDate(
                start_utc.strftime("%Y-%m-%d"),
                (end_utc + timedelta(days=1)).strftime("%Y-%m-%d"),
            )
            .select(self.BAND)
        )

        n_images = collection.size().getInfo()
        if n_images == 0:
            self.logger.warning(f"No IMERG images found for {event_id}")
            return pd.DataFrame()

        self.logger.info(f"IMERG images: {n_images}")

        # Sample each image at all stations
        records = []

        def _sample_image(image):
            timestamp = image.date().format("YYYY-MM-dd HH:mm:ss")
            sampled = image.sampleRegions(
                collection=fc,
                scale=11132,
                geometries=False,
            )
            return sampled.map(
                lambda f: f.set("timestamp", timestamp)
            )

        sampled_col = collection.map(_sample_image).flatten()

        try:
            data_list = sampled_col.getInfo()["features"]
        except Exception as exc:
            self.logger.error(f"GEE data retrieval failed: {exc}")
            return pd.DataFrame()

        for feat in data_list:
            props = feat["properties"]
            records.append({
                "Date and Time": props.get("timestamp"),
                props.get("station_id", "unknown"): props.get(self.BAND, 0.0),
            })

        if not records:
            self.logger.warning(f"Empty result for {event_id}")
            return pd.DataFrame()

        # Pivot to wide format: one row per timestamp, one column per station
        df_long = pd.DataFrame(records)
        # GEE returns UTC timestamps — convert to Nepal Standard Time (UTC+5:45)
        # before filtering and saving so timestamps in the GAG file are Nepal time.
        df_long["Date and Time"] = (
            pd.to_datetime(df_long["Date and Time"])
            + pd.Timedelta(hours=5, minutes=45)
        )
        df_wide = df_long.groupby("Date and Time").first().reset_index()

        # Trim to exact Nepal-time window (comparison now valid: both Nepal time)
        mask = (df_wide["Date and Time"] >= pd.to_datetime(start_nepal)) & \
               (df_wide["Date and Time"] <= pd.to_datetime(end_nepal))
        df_wide = df_wide[mask].reset_index(drop=True)

        # Replace NaN with 0
        df_wide.fillna(0.0, inplace=True)

        # Save
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"imerg_{event_id}.csv"
        df_wide.to_csv(out_path, index=False)
        self.logger.info(f"✓ Saved {len(df_wide)} rows → {out_path}")

        time.sleep(retry_delay)
        return df_wide

    def download_all_events(
        self,
        events_df: pd.DataFrame,
        stations_df: pd.DataFrame,
        output_dir: str | Path,
    ) -> dict:
        """
        Download IMERG data for every event in *events_df*.

        Parameters
        ----------
        events_df  : DataFrame with columns [flood_id, precip_start_daily,
                     precip_end_daily]
        stations_df: station metadata [station_id, latitude, longitude]
        output_dir : directory for CSV outputs

        Returns
        -------
        {flood_id: success_bool}
        """
        results = {}
        n = len(events_df)
        for i, row in events_df.iterrows():
            flood_id = row["flood_id"]
            start = str(row["precip_start_daily"]) + " 00:00:00"
            end   = str(row["precip_end_daily"])   + " 23:45:00"

            self.logger.info(f"\n[{i+1}/{n}] {flood_id}")
            df = self.download_event(flood_id, start, end, stations_df, output_dir)
            results[flood_id] = not df.empty

        n_ok = sum(results.values())
        self.logger.info(f"\nCompleted: {n_ok}/{n} events downloaded successfully")
        return results
