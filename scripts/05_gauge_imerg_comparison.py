#!/usr/bin/env python3
"""
Step 5 – Citizen Science Gauge vs IMERG Direct Comparison
==========================================================
Compares tipping-bucket rain gauge observations (TP001–TP008, 15-min resolution)
against IMERG satellite rainfall from the GAG files at the same station locations.

This provides a ground-truth IMERG bias estimate that is independent of GSSHA
model errors — separating instrument/satellite bias from hydrological model uncertainty.

Key output
----------
    outputs/gauge_imerg_comparison.csv  — per-event × per-station comparison
    outputs/gauge_imerg_summary.csv     — per-cluster and per-station averages
    outputs/figures/05_gauge_imerg_comparison.png

Interpretation
--------------
    bias_ratio = CS_mm / IMERG_mm

    ratio > 1  →  IMERG UNDERESTIMATES (gauge sees more rain than satellite)
    ratio ≈ 1  →  IMERG is accurate
    ratio < 1  →  IMERG OVERESTIMATES (gauge sees less rain than satellite)

Clusters are assigned using the final watershed correction α* values from the
most recent run (outputs/runs/<latest>/watershed_correction/...). Events before
2018 (no citizen science network) are skipped automatically.

Usage
-----
    python scripts/05_gauge_imerg_comparison.py
    python scripts/05_gauge_imerg_comparison.py --run-id 20260522_130703
    python scripts/05_gauge_imerg_comparison.py --gap-threshold 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ---------------------------------------------------------------------------
# Load configs
# ---------------------------------------------------------------------------

def _load(name: str) -> dict:
    with open(ROOT / "configs" / name) as fh:
        return yaml.safe_load(fh)


paths_cfg   = _load("paths.yaml")
station_cfg = _load("stations.yaml")


def _p(rel: str) -> Path:
    return (ROOT / rel).resolve()


FLOOD_CSV    = _p(paths_cfg["data"]["processed"]["flood_csv"])
IMERG_DIR    = _p(paths_cfg["data"]["processed"]["imerg_gag_dir"])
CS_DIR       = ROOT / "Data" / "raw" / "Citizen Science" / "KV-Data"
OUTPUTS_DIR  = _p(paths_cfg["outputs"]["base"])
RUNS_ROOT    = _p(paths_cfg["outputs"]["runs_root"])
LATEST_LINK  = _p(paths_cfg["outputs"]["latest_link"])


# ---------------------------------------------------------------------------
# Station mapping: Excel filename → station_id (matching GAG column order)
# ---------------------------------------------------------------------------

STATION_FILES = {
    "TP001": "TP001 (Nagarjun).xlsx",
    "TP002": "TP002 (Tokha).xlsx",
    "TP003": "TP003 (Lapsifedi).xlsx",
    "TP005": "TP005 (Kusunti).xlsx",
    "TP006": "TP006 (Liwali).xlsx",
    "TP007": "TP007 (Bhardev).xlsx",
    "TP008": "TP008 (Okhreni).xlsx",
}

# Map station_id → index in GAG file (0-based, matches station_names order)
GAG_STATION_ORDER = ["TP001", "TP002", "TP003", "TP005", "TP006", "TP007", "TP008"]


# ---------------------------------------------------------------------------
# Helper: load citizen science data for one station over a date range
# ---------------------------------------------------------------------------

def _load_cs_station(
    station_id: str,
    start_date: str,
    end_date: str,
    gap_threshold: float = 0.30,
) -> tuple[float, float, str]:
    """
    Return (total_mm, data_coverage_fraction, quality_flag) for one station
    over the event window [start_date, end_date] (inclusive).

    quality_flag: 'ok', 'gap_warn', 'no_data', 'file_missing'
    """
    filename = STATION_FILES.get(station_id)
    if filename is None:
        return float("nan"), 0.0, "no_mapping"

    filepath = CS_DIR / filename
    if not filepath.exists():
        return float("nan"), 0.0, "file_missing"

    try:
        df = pd.read_excel(filepath, usecols=["Date and Time", "Precipitation (mm)"])
        df.columns = ["datetime", "precip_mm"]
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)

        # Filter to event window
        start = pd.to_datetime(start_date)
        end   = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        mask  = (df["datetime"] >= start) & (df["datetime"] <= end)
        subset = df[mask].copy()

        if subset.empty:
            return float("nan"), 0.0, "no_data"

        # Expected number of 15-min records in window
        window_minutes = (end - start).total_seconds() / 60
        expected = int(window_minutes / 15)
        actual   = len(subset)
        coverage = actual / expected if expected > 0 else 0.0

        # Replace negative values with 0
        subset["precip_mm"] = subset["precip_mm"].clip(lower=0)
        total_mm = float(subset["precip_mm"].sum())

        flag = "gap_warn" if coverage < (1 - gap_threshold) else "ok"
        return total_mm, coverage, flag

    except Exception as exc:
        return float("nan"), 0.0, f"error: {exc}"


# ---------------------------------------------------------------------------
# Helper: load IMERG total from GAG file for one station
# ---------------------------------------------------------------------------

def _load_imerg_station(event_id: str, station_idx: int) -> float:
    """Return total IMERG rainfall (mm) for one station in one event's GAG file."""
    from src.gssha.gag_editor import GagFile

    gag_path = IMERG_DIR / f"{event_id}.gag"
    if not gag_path.exists():
        return float("nan")
    try:
        gag    = GagFile(gag_path)
        totals = gag.get_total_rainfall_per_station()
        if station_idx < len(totals):
            return float(totals[station_idx])
    except Exception:
        pass
    return float("nan")


# ---------------------------------------------------------------------------
# Helper: resolve latest run and load event alphas
# ---------------------------------------------------------------------------

def _load_event_alphas(run_id: str | None) -> dict:
    """Return {event_id: alpha*} from the latest (or specified) watershed run."""
    subdir = "watershed_correction"

    if run_id:
        run_root = RUNS_ROOT / run_id / subdir
    elif LATEST_LINK.exists():
        candidate = LATEST_LINK.resolve()
        run_root  = candidate if candidate.name == subdir else candidate / subdir
    else:
        # Find newest run
        candidates = sorted(
            [p for p in RUNS_ROOT.iterdir() if p.is_dir()],
            key=lambda p: p.name,
            reverse=True,
        )
        run_root = None
        for rd in candidates:
            if (rd / subdir).exists():
                run_root = rd / subdir
                break

    if run_root is None or not run_root.exists():
        print("⚠  No watershed correction run found — cluster labels will be missing.")
        return {}

    # Find last iteration
    iter_dirs = sorted(run_root.glob("iteration_*"))
    if not iter_dirs:
        return {}
    last_alpha = iter_dirs[-1] / "artifacts" / "Alpha_Scalar.json"
    if not last_alpha.exists():
        return {}

    with open(last_alpha) as fh:
        data = json.load(fh)
    return data.get("event_alphas", {})


def _cluster_label(alpha: float) -> str:
    if np.isnan(alpha):
        return "unknown"
    if alpha < 0.70:
        return "overestimation"
    if alpha > 1.20:
        return "underestimation"
    return "balanced"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(run_id: str | None, gap_threshold: float) -> None:
    print("=" * 70)
    print("STEP 5 – CITIZEN SCIENCE GAUGE vs IMERG COMPARISON")
    print("=" * 70)

    if not FLOOD_CSV.exists():
        sys.exit(f"Flood event list not found: {FLOOD_CSV}")

    events_df = pd.read_csv(FLOOD_CSV).dropna(subset=["flood_id"])
    print(f"\nTotal events: {len(events_df)}")

    event_alphas = _load_event_alphas(run_id)
    if event_alphas:
        print(f"Loaded α* for {len(event_alphas)} events from watershed correction run.")
    else:
        print("No α* values found — cluster labels will be 'unknown'.")

    # Filter events before 2018 (no citizen science network)
    events_df["year"] = pd.to_datetime(events_df["precip_start_daily"]).dt.year
    pre2018 = events_df[events_df["year"] < 2018]
    if not pre2018.empty:
        print(f"\nSkipping {len(pre2018)} pre-2018 events (no citizen science network):")
        for _, r in pre2018.iterrows():
            print(f"  {r['flood_id']} ({r['year']})")
    events_df = events_df[events_df["year"] >= 2018].reset_index(drop=True)
    print(f"\nProcessing {len(events_df)} events (2018–present) × {len(GAG_STATION_ORDER)} stations …\n")

    records = []
    for _, ev in events_df.iterrows():
        event_id  = ev["flood_id"]
        start     = str(ev["precip_start_daily"])
        end       = str(ev["precip_end_daily"])
        year      = int(ev["year"])
        month     = int(pd.to_datetime(start).month)
        alpha_val = event_alphas.get(event_id, float("nan"))
        cluster   = _cluster_label(alpha_val)

        print(f"  {event_id}  [cluster={cluster}, α={alpha_val:.3f}]")

        for s_idx, station_id in enumerate(GAG_STATION_ORDER):
            cs_mm, coverage, flag = _load_cs_station(
                station_id, start, end, gap_threshold=gap_threshold
            )
            imerg_mm = _load_imerg_station(event_id, s_idx)

            bias_ratio = (cs_mm / imerg_mm) if (imerg_mm > 0 and not np.isnan(cs_mm)) else float("nan")

            status = "✓" if flag == "ok" else ("⚠" if "warn" in flag else "✗")
            print(f"    {status} {station_id}: CS={cs_mm:6.1f}mm  IMERG={imerg_mm:6.1f}mm  "
                  f"ratio={bias_ratio:5.2f}  cov={coverage:.0%}  [{flag}]")

            records.append({
                "event_id":     event_id,
                "station_id":   station_id,
                "year":         year,
                "month":        month,
                "cluster":      cluster,
                "alpha_star":   round(alpha_val, 4) if not np.isnan(alpha_val) else None,
                "CS_mm":        round(cs_mm, 2)    if not np.isnan(cs_mm)    else None,
                "IMERG_mm":     round(imerg_mm, 2) if not np.isnan(imerg_mm) else None,
                "bias_ratio":   round(bias_ratio, 4) if not np.isnan(bias_ratio) else None,
                "cs_coverage":  round(coverage, 3),
                "cs_flag":      flag,
            })

    result_df = pd.DataFrame(records)

    # Save outputs
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUTPUTS_DIR / "gauge_imerg_comparison.csv"
    result_df.to_csv(out_csv, index=False)
    print(f"\n✓ Detailed results → {out_csv}")

    # Print cluster summary
    _print_cluster_summary(result_df)

    # Per-station summary
    _print_station_summary(result_df)

    # Save summary CSVs
    _save_summaries(result_df, OUTPUTS_DIR)

    # Plot
    _plot(result_df, OUTPUTS_DIR)


def _print_cluster_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("  CLUSTER SUMMARY  (bias_ratio = CS_mm / IMERG_mm)")
    print("  Note: CS=0mm when IMERG>20mm excluded (likely gauge malfunction)")
    print("─" * 70)
    print(f"  {'CLUSTER':<18}  {'N':>4}  {'Mean ratio':>10}  {'Std':>6}  INTERPRETATION")
    print(f"  {'─'*18}  {'─'*4}  {'─'*10}  {'─'*6}  {'─'*35}")

    # Exclude zero-CS readings during active IMERG events (likely gauge malfunction)
    good = df[df["cs_flag"] == "ok"].dropna(subset=["bias_ratio"])
    good = good[~((good["CS_mm"] == 0) & (good["IMERG_mm"] > 20))]
    for cluster in ["overestimation", "balanced", "underestimation", "unknown"]:
        sub = good[good["cluster"] == cluster]
        if sub.empty:
            continue
        mean_r = sub["bias_ratio"].mean()
        std_r  = sub["bias_ratio"].std()
        if cluster == "balanced":
            note = "IMERG ~accurate for these events"
        elif cluster == "underestimation":
            note = "Gauge > IMERG → IMERG misses rain" if mean_r > 1 else "Gauge ≈ IMERG → model error"
        elif cluster == "overestimation":
            note = "Gauge < IMERG → IMERG too high" if mean_r < 1 else "Gauge ≈ IMERG → model error"
        else:
            note = ""
        print(f"  {cluster:<18}  {len(sub):>4}  {mean_r:>10.3f}  {std_r:>6.3f}  {note}")
    print("─" * 70)
    all_good = good.dropna(subset=["bias_ratio"])
    if not all_good.empty:
        print(f"  {'Overall':<18}  {len(all_good):>4}  {all_good['bias_ratio'].mean():>10.3f}  "
              f"{all_good['bias_ratio'].std():>6.3f}  (gauge-active readings only)")
    print("=" * 70)

    # Coverage warning
    total_records = len(df)
    no_data = (df["cs_flag"] == "no_data").sum()
    zero_malfunc = ((df["cs_flag"] == "ok") & (df["CS_mm"].fillna(-1) == 0) & (df["IMERG_mm"].fillna(0) > 20)).sum()
    print(f"\n  Data coverage: {total_records} records total")
    print(f"    no_data (CS files end 2020-12-31): {no_data} ({100*no_data/total_records:.0f}%)")
    print(f"    zero-during-rain (excluded above): {zero_malfunc}")
    print(f"    gauge-active readings used:        {len(all_good)}")
    print(f"  ⚠ CS data only covers 2018–2020 — 2021–2024 events have no gauge data.")


def _print_station_summary(df: pd.DataFrame) -> None:
    print("\n  PER-STATION MEAN BIAS RATIO  (CS/IMERG, gauge-active only)")
    print("─" * 50)
    good = df[df["cs_flag"] == "ok"].dropna(subset=["bias_ratio"])
    good = good[~((good["CS_mm"] == 0) & (good["IMERG_mm"] > 20))]
    for sid in GAG_STATION_ORDER:
        sub = good[good["station_id"] == sid]
        if sub.empty:
            continue
        mean_r = sub["bias_ratio"].mean()
        bar    = "█" * int(min(mean_r, 3.0) * 10)
        print(f"  {sid}:  {mean_r:5.3f}  {bar}")
    print("─" * 50)


def _save_summaries(df: pd.DataFrame, out_dir: Path) -> None:
    good = df[df["cs_flag"] == "ok"].dropna(subset=["bias_ratio"])
    good = good[~((good["CS_mm"] == 0) & (good["IMERG_mm"] > 20))]

    cluster_summary = (
        good.groupby("cluster")["bias_ratio"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "mean_bias_ratio", "std": "std_bias_ratio", "count": "n"})
        .reset_index()
    )
    cluster_summary.to_csv(out_dir / "gauge_imerg_cluster_summary.csv", index=False)

    station_summary = (
        good.groupby("station_id")["bias_ratio"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "mean_bias_ratio", "std": "std_bias_ratio", "count": "n"})
        .reset_index()
    )
    station_summary.to_csv(out_dir / "gauge_imerg_station_summary.csv", index=False)
    print(f"✓ Cluster summary → {out_dir / 'gauge_imerg_cluster_summary.csv'}")
    print(f"✓ Station summary → {out_dir / 'gauge_imerg_station_summary.csv'}")


def _plot(df: pd.DataFrame, out_dir: Path) -> None:
    """Two-panel plot: scatter CS vs IMERG (coloured by cluster) + per-station ratio bars."""
    good = df[df["cs_flag"] == "ok"].dropna(subset=["CS_mm", "IMERG_mm", "bias_ratio"])
    good = good[~((good["CS_mm"] == 0) & (good["IMERG_mm"] > 20))]
    if good.empty:
        print("⚠  No valid data for plotting.")
        return

    COLOURS = {
        "overestimation": "#e84040",
        "balanced":       "#2ca02c",
        "underestimation":"#1f77b4",
        "unknown":        "#aaaaaa",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ---- Panel A: scatter CS vs IMERG ----
    ax = axes[0]
    for cluster, colour in COLOURS.items():
        sub = good[good["cluster"] == cluster]
        if sub.empty:
            continue
        ax.scatter(sub["IMERG_mm"], sub["CS_mm"], c=colour, label=cluster,
                   s=50, alpha=0.7, edgecolors="k", lw=0.3)

    max_val = max(good["IMERG_mm"].max(), good["CS_mm"].max()) * 1.05
    ax.plot([0, max_val], [0, max_val], "k--", lw=1, label="1:1 line")
    ax.set_xlabel("IMERG rainfall (mm)", fontsize=10)
    ax.set_ylabel("Citizen science gauge (mm)", fontsize=10)
    ax.set_title("Gauge vs IMERG rainfall\n(per event × station, quality-filtered)", fontsize=11)
    ax.legend(fontsize=8, title="Cluster")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)

    # ---- Panel B: per-station mean bias ratio ----
    ax = axes[1]
    station_means = (
        good.groupby("station_id")["bias_ratio"].mean()
        .reindex(GAG_STATION_ORDER)
        .dropna()
    )
    x = np.arange(len(station_means))
    colours_bar = ["#e84040" if v < 0.90 else "#2ca02c" if v < 1.10 else "#1f77b4"
                   for v in station_means.values]
    ax.bar(x, station_means.values, color=colours_bar, alpha=0.85, edgecolor="k", lw=0.5)
    ax.axhline(1.0, color="k", lw=1.2, ls="--", label="ratio = 1 (no bias)")
    ax.set_xticks(x)
    ax.set_xticklabels(station_means.index, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mean CS / IMERG ratio", fontsize=10)
    ax.set_title("Per-station IMERG bias\n(ratio > 1 = IMERG underestimates)", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    patches = [
        mpatches.Patch(color="#e84040", label="IMERG overestimates (ratio<0.90)"),
        mpatches.Patch(color="#2ca02c", label="IMERG accurate (0.90–1.10)"),
        mpatches.Patch(color="#1f77b4", label="IMERG underestimates (ratio>1.10)"),
    ]
    ax.legend(handles=patches, fontsize=7, loc="upper right")

    fig.suptitle("Citizen Science vs IMERG — Kathmandu Valley", fontsize=12, y=1.01)
    fig.tight_layout()

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "05_gauge_imerg_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"✓ Comparison plot → {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare citizen science gauges against IMERG satellite rainfall"
    )
    parser.add_argument(
        "--run-id", type=str, default=None,
        help="Watershed correction run ID to read α* values from (default: latest)"
    )
    parser.add_argument(
        "--gap-threshold", type=float, default=0.30,
        help="Max allowed data gap fraction per station/event (default: 0.30 = 30%%)"
    )
    args = parser.parse_args()
    main(run_id=args.run_id, gap_threshold=args.gap_threshold)
