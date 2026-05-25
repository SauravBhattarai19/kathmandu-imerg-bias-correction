#!/usr/bin/env python3
"""
Step 5 – Citizen Science Gauge vs IMERG Direct Comparison
==========================================================
Compares tipping-bucket rain gauge observations (TP001–TP008, 15-min resolution)
against IMERG satellite rainfall from the GAG files at the same station locations.

This provides a ground-truth IMERG bias estimate that is independent of GSSHA
model errors — separating instrument/satellite bias from hydrological model
uncertainty.

Data source
-----------
    Data/raw/Citizen Science/precipitation_data.csv
        Combined 15-min record for TP001–TP008, 2018-01-01 → 2025-08-29.
        TP004 (Chandragiri) is present in the CSV but absent from the IMERG
        GAG files and is therefore excluded from the comparison.

Known data gaps
---------------
    event_011_20150817  Pre-2018, no CS network → skipped
    event_007_20190712  July 2019 absent from combined CSV → skipped

Key output
----------
    outputs/gauge_imerg_comparison.csv      – per-event × per-station
    outputs/gauge_imerg_cluster_summary.csv – per-cluster means
    outputs/gauge_imerg_station_summary.csv – per-station means
    outputs/figures/05_gauge_imerg_comparison.png

Interpretation
--------------
    bias_ratio = CS_mm / IMERG_mm

    ratio > 1  →  IMERG UNDERESTIMATES (gauge sees more rain than satellite)
    ratio ≈ 1  →  IMERG is accurate
    ratio < 1  →  IMERG OVERESTIMATES (gauge sees less rain than satellite)

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


paths_cfg = _load("paths.yaml")


def _p(rel: str) -> Path:
    return (ROOT / rel).resolve()


FLOOD_CSV   = _p(paths_cfg["data"]["processed"]["flood_csv"])
IMERG_DIR   = _p(paths_cfg["data"]["processed"]["imerg_gag_dir"])
CS_CSV      = ROOT / "Data" / "raw" / "Citizen Science" / "precipitation_data.csv"
OUTPUTS_DIR = _p(paths_cfg["outputs"]["base"])
RUNS_ROOT   = _p(paths_cfg["outputs"]["runs_root"])
LATEST_LINK = _p(paths_cfg["outputs"]["latest_link"])

# Stations that appear in both the IMERG GAG files and the CS CSV.
# TP004 (Chandragiri) is in the CSV but NOT in the GAG files → excluded.
# Order matches GAG file column order (index 0–6).
GAG_STATION_ORDER = ["TP001", "TP002", "TP003", "TP005", "TP006", "TP007", "TP008"]

# Column name prefix in precipitation_data.csv: "TP00X_..."
CS_COL_PREFIX = {
    "TP001": "TP001_Nagarjun",
    "TP002": "TP002_Tokha",
    "TP003": "TP003_Lapsephedi",
    "TP005": "TP005_Kusunti",
    "TP006": "TP006_Bhaktapur",
    "TP007": "TP007_Bhardev",
    "TP008": "TP008_Okhreni",
}


# ---------------------------------------------------------------------------
# Load the combined CS dataset once at import time
# ---------------------------------------------------------------------------

def _load_cs_dataframe() -> pd.DataFrame | None:
    """Load precipitation_data.csv, return DataFrame indexed by datetime."""
    if not CS_CSV.exists():
        print(f"⚠  CS combined CSV not found: {CS_CSV}")
        return None

    df = pd.read_csv(CS_CSV)
    df["dt"] = pd.to_datetime(df["Date and Time"])
    df = df.set_index("dt").drop(columns=["Date and Time"])

    # Build a clean mapping: station_id → full column name
    col_map = {}
    for sid, prefix in CS_COL_PREFIX.items():
        matches = [c for c in df.columns if c.startswith(prefix[:8])]
        if matches:
            col_map[sid] = matches[0]

    # Keep only the 7 GAG stations, rename to station IDs
    keep = {v: k for k, v in col_map.items() if v in df.columns}
    df = df[[c for c in df.columns if c in keep]].rename(columns=keep)
    return df


_CS_DF: pd.DataFrame | None = None   # module-level cache


def _get_cs_df() -> pd.DataFrame | None:
    global _CS_DF
    if _CS_DF is None:
        _CS_DF = _load_cs_dataframe()
    return _CS_DF


# ---------------------------------------------------------------------------
# Helper: per-station CS total over event window
# ---------------------------------------------------------------------------

def _load_cs_station(
    station_id: str,
    start_date: str,
    end_date: str,
    gap_threshold: float = 0.30,
) -> tuple[float, float, str]:
    """
    Return (total_mm, coverage_fraction, quality_flag).

    quality_flag: 'ok', 'gap_warn', 'no_data', 'no_csv'
    """
    cs = _get_cs_df()
    if cs is None:
        return float("nan"), 0.0, "no_csv"
    if station_id not in cs.columns:
        return float("nan"), 0.0, "no_column"

    start = pd.to_datetime(start_date)
    end   = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    sub   = cs.loc[start:end, station_id].dropna()

    if sub.empty:
        return float("nan"), 0.0, "no_data"

    # Coverage: expected 15-min slots vs actual non-NaN rows
    window_min = (end - start).total_seconds() / 60
    expected   = int(window_min / 15)
    coverage   = len(sub) / expected if expected > 0 else 0.0

    total_mm = float(sub.clip(lower=0).sum())
    flag     = "gap_warn" if coverage < (1 - gap_threshold) else "ok"
    return total_mm, coverage, flag


# ---------------------------------------------------------------------------
# Helper: per-station IMERG total from GAG file
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
# Helper: load per-event α* from latest watershed run
# ---------------------------------------------------------------------------

def _load_event_alphas(run_id: str | None) -> dict:
    """Return {event_id: alpha*} from the latest (or specified) watershed run."""
    subdir = "watershed_correction"

    if run_id:
        run_root = RUNS_ROOT / run_id / subdir
    else:
        # Always scan timestamped run directories newest-first, regardless of symlink
        run_root = None
        if RUNS_ROOT.exists():
            candidates = sorted(
                [p for p in RUNS_ROOT.iterdir() if p.is_dir() and (p / subdir).exists()],
                key=lambda p: p.name,
                reverse=True,
            )
            if candidates:
                run_root = candidates[0] / subdir

    if run_root is None or not run_root.exists():
        print("⚠  No watershed correction run found — cluster labels will be missing.")
        return {}

    # Find the latest iteration that has a completed Alpha_Scalar.json
    iter_dirs = sorted(run_root.glob("iteration_*"), reverse=True)
    last_alpha = None
    for d in iter_dirs:
        candidate = d / "artifacts" / "Alpha_Scalar.json"
        if candidate.exists():
            last_alpha = candidate
            break

    if last_alpha is None:
        # Try parent runs in reverse order (handles incomplete newest run)
        if RUNS_ROOT.exists() and run_id is None:
            all_runs = sorted(
                [p for p in RUNS_ROOT.iterdir() if p.is_dir()],
                key=lambda p: p.name, reverse=True,
            )
            for rd in all_runs:
                for alpha_f in sorted((rd / subdir).glob("iteration_*/artifacts/Alpha_Scalar.json"), reverse=True):
                    last_alpha = alpha_f
                    break
                if last_alpha:
                    break
    if last_alpha is None:
        print("⚠  No Alpha_Scalar.json found — cluster labels will be missing.")
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
# Quality filter used consistently across summary / plot functions
# ---------------------------------------------------------------------------

def _quality_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only ok-flagged, non-NaN ratios; drop CS=0 when IMERG>20mm (malfunction)."""
    good = df[df["cs_flag"] == "ok"].dropna(subset=["bias_ratio"])
    good = good[~((good["CS_mm"] == 0) & (good["IMERG_mm"] > 20))]
    return good


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(run_id: str | None, gap_threshold: float) -> None:
    print("=" * 70)
    print("STEP 5 – CITIZEN SCIENCE GAUGE vs IMERG COMPARISON")
    print(f"  CS source: {CS_CSV.name}  (2018-01-01 → 2025-08-29)")
    print("=" * 70)

    if not FLOOD_CSV.exists():
        sys.exit(f"Flood event list not found: {FLOOD_CSV}")

    events_df = pd.read_csv(FLOOD_CSV).dropna(subset=["flood_id"])
    print(f"\nTotal events: {len(events_df)}")

    # Pre-load CS dataframe (gives date-range info)
    cs = _get_cs_df()
    if cs is not None:
        print(f"CS data loaded: {len(cs):,} rows, "
              f"{cs.index.min().date()} → {cs.index.max().date()}, "
              f"{len(cs.columns)} stations")

    event_alphas = _load_event_alphas(run_id)
    if event_alphas:
        print(f"Loaded α* for {len(event_alphas)} events from watershed correction run.")
    else:
        print("No α* values found — cluster labels will be 'unknown'.")

    # Filter pre-2018 events (no CS network)
    events_df["year"] = pd.to_datetime(events_df["precip_start_daily"]).dt.year
    pre2018 = events_df[events_df["year"] < 2018]
    if not pre2018.empty:
        print(f"\nSkipping {len(pre2018)} pre-2018 events (no CS network):")
        for _, r in pre2018.iterrows():
            print(f"  {r['flood_id']} ({r['year']})")
    events_df = events_df[events_df["year"] >= 2018].reset_index(drop=True)

    print(f"\nProcessing {len(events_df)} events × {len(GAG_STATION_ORDER)} stations …\n")

    records = []
    for _, ev in events_df.iterrows():
        event_id  = ev["flood_id"]
        start     = str(ev["precip_start_daily"])
        end       = str(ev["precip_end_daily"])
        year      = int(ev["year"])
        month     = int(pd.to_datetime(start).month)
        alpha_val = event_alphas.get(event_id, float("nan"))
        cluster   = _cluster_label(alpha_val)

        print(f"  {event_id}  [cluster={cluster}, α={'nan' if np.isnan(alpha_val) else f'{alpha_val:.3f}'}]")

        for s_idx, station_id in enumerate(GAG_STATION_ORDER):
            cs_mm, coverage, flag = _load_cs_station(
                station_id, start, end, gap_threshold=gap_threshold
            )
            imerg_mm = _load_imerg_station(event_id, s_idx)

            bias_ratio = (
                (cs_mm / imerg_mm)
                if (imerg_mm > 0 and not np.isnan(cs_mm))
                else float("nan")
            )

            status = "✓" if flag == "ok" else ("⚠" if "warn" in flag else "✗")
            print(f"    {status} {station_id}: CS={cs_mm:6.1f}mm  IMERG={imerg_mm:6.1f}mm  "
                  f"ratio={bias_ratio:5.2f}  cov={coverage:.0%}  [{flag}]")

            records.append({
                "event_id":   event_id,
                "station_id": station_id,
                "year":       year,
                "month":      month,
                "cluster":    cluster,
                "alpha_star": round(alpha_val, 4) if not np.isnan(alpha_val) else None,
                "CS_mm":      round(cs_mm, 2)     if not np.isnan(cs_mm)     else None,
                "IMERG_mm":   round(imerg_mm, 2)  if not np.isnan(imerg_mm)  else None,
                "bias_ratio": round(bias_ratio, 4) if not np.isnan(bias_ratio) else None,
                "cs_coverage": round(coverage, 3),
                "cs_flag":    flag,
            })

    result_df = pd.DataFrame(records)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUTPUTS_DIR / "gauge_imerg_comparison.csv"
    result_df.to_csv(out_csv, index=False)
    print(f"\n✓ Detailed results → {out_csv}")

    _print_cluster_summary(result_df)
    _print_station_summary(result_df)
    _save_summaries(result_df, OUTPUTS_DIR)
    _plot(result_df, OUTPUTS_DIR)


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _print_cluster_summary(df: pd.DataFrame) -> None:
    good = _quality_filter(df)

    print("\n" + "=" * 70)
    print("  CLUSTER SUMMARY  (bias_ratio = CS_mm / IMERG_mm)")
    print("  Quality filter: ok coverage, CS>0 when IMERG>20mm, bias_ratio not NaN")
    print("─" * 70)
    print(f"  {'CLUSTER':<18}  {'N':>4}  {'Mean ratio':>10}  {'Std':>6}  {'Median':>7}  INTERPRETATION")
    print(f"  {'─'*18}  {'─'*4}  {'─'*10}  {'─'*6}  {'─'*7}  {'─'*35}")

    for cluster in ["overestimation", "balanced", "underestimation", "unknown"]:
        sub = good[good["cluster"] == cluster]
        if sub.empty:
            continue
        mean_r   = sub["bias_ratio"].mean()
        std_r    = sub["bias_ratio"].std()
        median_r = sub["bias_ratio"].median()
        if cluster == "balanced":
            note = "IMERG accurate for these events" if 0.9 < mean_r < 1.1 else "IMERG biased even for balanced cluster"
        elif cluster == "underestimation":
            note = "Gauge > IMERG → IMERG misses rain" if mean_r > 1.0 else "Gauge < IMERG → model error not IMERG"
        elif cluster == "overestimation":
            note = "Gauge < IMERG → IMERG too high" if mean_r < 1.0 else "Gauge ≈ IMERG → confirms model error"
        else:
            note = ""
        print(f"  {cluster:<18}  {len(sub):>4}  {mean_r:>10.3f}  {std_r:>6.3f}  {median_r:>7.3f}  {note}")

    print("─" * 70)
    if not good.empty:
        print(f"  {'Overall':<18}  {len(good):>4}  {good['bias_ratio'].mean():>10.3f}  "
              f"{good['bias_ratio'].std():>6.3f}  {good['bias_ratio'].median():>7.3f}  "
              f"(all quality-filtered readings)")
    print("=" * 70)

    # Coverage breakdown
    total     = len(df)
    no_data   = (df["cs_flag"] == "no_data").sum()
    gap_warn  = (df["cs_flag"] == "gap_warn").sum()
    zero_rain = ((df["cs_flag"] == "ok") & (df["CS_mm"].fillna(-1) == 0)
                 & (df["IMERG_mm"].fillna(0) > 20)).sum()
    print(f"\n  Coverage breakdown ({total} station×event pairs):")
    print(f"    no_data (Jul 2019 gap / pre-2018) : {no_data:>4} ({100*no_data/total:.0f}%)")
    print(f"    gap_warn (>30% missing)           : {gap_warn:>4} ({100*gap_warn/total:.0f}%)")
    print(f"    CS=0 during IMERG rain (excluded) : {zero_rain:>4}")
    print(f"    quality-filtered readings used    : {len(good):>4}")


def _print_station_summary(df: pd.DataFrame) -> None:
    good = _quality_filter(df)
    print("\n  PER-STATION MEAN BIAS RATIO  (CS/IMERG, quality-filtered)")
    print("─" * 60)
    for sid in GAG_STATION_ORDER:
        sub = good[good["station_id"] == sid]
        if sub.empty:
            print(f"  {sid}:  no data")
            continue
        mean_r = sub["bias_ratio"].mean()
        n      = len(sub)
        bar    = "█" * int(min(mean_r, 3.0) * 10)
        print(f"  {sid} (n={n:2d}):  {mean_r:5.3f}  {bar}")
    print("─" * 60)


def _save_summaries(df: pd.DataFrame, out_dir: Path) -> None:
    good = _quality_filter(df)

    cluster_summary = (
        good.groupby("cluster")["bias_ratio"]
        .agg(["mean", "median", "std", "count"])
        .rename(columns={"mean": "mean_bias_ratio", "median": "median_bias_ratio",
                         "std": "std_bias_ratio", "count": "n"})
        .reset_index()
    )
    cluster_summary.to_csv(out_dir / "gauge_imerg_cluster_summary.csv", index=False)

    station_summary = (
        good.groupby("station_id")["bias_ratio"]
        .agg(["mean", "median", "std", "count"])
        .rename(columns={"mean": "mean_bias_ratio", "median": "median_bias_ratio",
                         "std": "std_bias_ratio", "count": "n"})
        .reset_index()
    )
    station_summary.to_csv(out_dir / "gauge_imerg_station_summary.csv", index=False)
    print(f"✓ Cluster summary → {out_dir / 'gauge_imerg_cluster_summary.csv'}")
    print(f"✓ Station summary → {out_dir / 'gauge_imerg_station_summary.csv'}")


def _plot(df: pd.DataFrame, out_dir: Path) -> None:
    """Three-panel figure: scatter CS vs IMERG, per-station ratio bars, per-event ratio."""
    good = _quality_filter(df)
    if good.empty:
        print("⚠  No valid data for plotting.")
        return

    COLOURS = {
        "overestimation": "#e84040",
        "balanced":       "#2ca02c",
        "underestimation": "#1f77b4",
        "unknown":        "#aaaaaa",
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # ── Panel A: scatter CS vs IMERG ──────────────────────────────────────
    ax = axes[0]
    for cluster, colour in COLOURS.items():
        sub = good[good["cluster"] == cluster]
        if sub.empty:
            continue
        ax.scatter(sub["IMERG_mm"], sub["CS_mm"], c=colour, label=cluster,
                   s=55, alpha=0.75, edgecolors="k", lw=0.4)

    max_val = max(good["IMERG_mm"].max(), good["CS_mm"].max()) * 1.08
    ax.plot([0, max_val], [0, max_val], "k--", lw=1.2, label="1:1 (no bias)")
    ax.set_xlabel("IMERG event total (mm)", fontsize=10)
    ax.set_ylabel("Citizen science gauge (mm)", fontsize=10)
    ax.set_title("Gauge vs IMERG\n(per event × station)", fontsize=11)
    ax.legend(fontsize=8, title="GSSHA cluster")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)

    # ── Panel B: per-station mean ratio ───────────────────────────────────
    ax = axes[1]
    station_means = (
        good.groupby("station_id")["bias_ratio"].mean()
        .reindex(GAG_STATION_ORDER).dropna()
    )
    x = np.arange(len(station_means))
    bar_colours = [
        "#e84040" if v < 0.90 else "#2ca02c" if v <= 1.10 else "#1f77b4"
        for v in station_means.values
    ]
    ax.bar(x, station_means.values, color=bar_colours, alpha=0.85, edgecolor="k", lw=0.5)
    ax.axhline(1.0, color="k", lw=1.2, ls="--")
    ax.axhline(0.9, color="grey", lw=0.8, ls=":", alpha=0.6)
    ax.axhline(1.1, color="grey", lw=0.8, ls=":", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(station_means.index, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mean CS / IMERG ratio", fontsize=10)
    ax.set_title("Per-station IMERG bias\n(ratio < 1 = IMERG overestimates)", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    patches = [
        mpatches.Patch(color="#e84040", label="Overest. (ratio<0.90)"),
        mpatches.Patch(color="#2ca02c", label="Accurate (0.90–1.10)"),
        mpatches.Patch(color="#1f77b4", label="Underest. (ratio>1.10)"),
    ]
    ax.legend(handles=patches, fontsize=7)

    # ── Panel C: per-event watershed-mean ratio ────────────────────────────
    ax = axes[2]
    event_means = (
        good.groupby(["event_id", "cluster"])["bias_ratio"].mean()
        .reset_index()
        .sort_values("bias_ratio")
    )
    colours_ev = [COLOURS.get(c, "#aaaaaa") for c in event_means["cluster"]]
    y = np.arange(len(event_means))
    ax.barh(y, event_means["bias_ratio"], color=colours_ev, alpha=0.85,
            edgecolor="k", lw=0.4)
    ax.axvline(1.0, color="k", lw=1.2, ls="--")
    ax.axvline(0.9, color="grey", lw=0.8, ls=":", alpha=0.6)
    ax.set_yticks(y)
    # Short labels
    labels = [e.replace("event_", "").replace("_flood", "") for e in event_means["event_id"]]
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Mean CS / IMERG ratio (all stations)", fontsize=10)
    ax.set_title("Per-event IMERG bias\n(watershed average)", fontsize=11)
    ax.grid(True, axis="x", alpha=0.3)
    # Cluster legend
    legend_patches = [mpatches.Patch(color=v, label=k) for k, v in COLOURS.items()
                      if k != "unknown"]
    ax.legend(handles=legend_patches, fontsize=7, title="GSSHA cluster")

    fig.suptitle(
        "Citizen Science vs IMERG — Kathmandu Valley\n"
        f"(n={len(good)} quality-filtered station×event pairs, 2018–2025)",
        fontsize=12
    )
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
        help="Max allowed data gap fraction per station/event (default: 0.30)"
    )
    args = parser.parse_args()
    main(run_id=args.run_id, gap_threshold=args.gap_threshold)
