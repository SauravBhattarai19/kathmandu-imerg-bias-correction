#!/usr/bin/env python3
"""
Visualisation helpers for bias correction results.

Provides three main plot types:
    plot_hydrographs         – before/after discharge comparison per event
    plot_alpha_map           – station α factors (bar chart + map stub)
    plot_performance_summary – RMSE / bias reduction summary
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Colour scheme
# ---------------------------------------------------------------------------

COLOUR_OBS  = "#1a1a1a"
COLOUR_BASE = "#4c72b0"
COLOUR_CORR = "#dd4444"
ALPHA_FILL  = 0.15


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_hydrographs(
    validation_df: pd.DataFrame,
    output_path: Optional[str | Path] = None,
    figsize: tuple = (14, 5),
) -> plt.Figure:
    """
    Bar-chart comparison of observed, baseline, and corrected peak discharges.

    Parameters
    ----------
    validation_df : output of ValidationRunner.validate_corrections()
    output_path   : if given, save figure here (PNG, 150 dpi)

    Returns
    -------
    matplotlib Figure
    """
    df = validation_df.copy()
    events = df["Event"].values
    x = np.arange(len(events))
    w = 0.28

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - w, df["Q_Observed"], w, color=COLOUR_OBS,  label="Observed",  alpha=0.9)
    ax.bar(x,     df["Q_Baseline"], w, color=COLOUR_BASE, label="Baseline",  alpha=0.75)
    ax.bar(x + w, df["Q_Corrected"], w, color=COLOUR_CORR, label="Corrected", alpha=0.75)

    ax.set_xticks(x)
    ax.set_xticklabels(events, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Peak Discharge (m³/s)")
    ax.set_title("Flood Event Peak Discharge – Baseline vs Corrected vs Observed")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"✓ Hydrograph comparison saved → {output_path}")

    return fig


def plot_alpha_map(
    alpha_factors: np.ndarray,
    station_names: List[str],
    station_coords: Optional[np.ndarray] = None,
    output_path: Optional[str | Path] = None,
    figsize: tuple = (10, 5),
) -> plt.Figure:
    """
    Visualise per-station α correction factors.

    If *station_coords* (shape n×2, UTM) is supplied, a scatter-map is added
    on the right panel with colours proportional to α.

    Parameters
    ----------
    alpha_factors  : shape (n_stations,)
    station_names  : list of station label strings
    station_coords : optional UTM coordinates, shape (n_stations, 2)
    output_path    : save destination
    """
    n = len(alpha_factors)
    ncols = 2 if station_coords is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=figsize)
    if ncols == 1:
        axes = [axes]

    # Bar chart
    ax = axes[0]
    colours = [COLOUR_CORR if a > 1 else COLOUR_BASE for a in alpha_factors]
    bars = ax.bar(range(n), (alpha_factors - 1) * 100, color=colours, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(n))
    ax.set_xticklabels(station_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Correction  (α − 1) × 100  [%]")
    ax.set_title("Per-Station Rainfall Correction Factors")
    ax.grid(True, axis="y", alpha=0.3)

    for bar, val in zip(bars, alpha_factors):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5 if val >= 1 else bar.get_height() - 2,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=7,
        )

    # Spatial map
    if station_coords is not None:
        ax2 = axes[1]
        sc = ax2.scatter(
            station_coords[:, 0], station_coords[:, 1],
            c=alpha_factors, cmap="RdYlBu_r", s=200,
            vmin=0.7, vmax=1.3, edgecolors="gray", linewidths=0.5,
        )
        plt.colorbar(sc, ax=ax2, label="α factor")
        for i, (name, coord) in enumerate(zip(station_names, station_coords)):
            ax2.annotate(name, coord, textcoords="offset points",
                         xytext=(4, 4), fontsize=7)
        ax2.set_xlabel("Easting (m)")
        ax2.set_ylabel("Northing (m)")
        ax2.set_title("Station Locations – Correction Magnitude")
        ax2.grid(True, alpha=0.3)

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"✓ Alpha map saved → {output_path}")

    return fig


def plot_performance_summary(
    validation_df: pd.DataFrame,
    output_path: Optional[str | Path] = None,
    figsize: tuple = (12, 9),
) -> plt.Figure:
    """
    2×2 diagnostic grid:
        - Peak discharge comparison
        - Absolute errors
        - Observed vs Simulated scatter
        - Relative bias
    """
    df = validation_df.copy()
    events = df["Event"].values
    x = np.arange(len(events))
    w = 0.35

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle("Bias Correction Validation Summary", fontsize=13, fontweight="bold")

    # ---- Peak discharge
    ax = axes[0, 0]
    ax.plot(x, df["Q_Observed"], "o-", color=COLOUR_OBS, lw=2, ms=6, label="Observed")
    ax.plot(x, df["Q_Baseline"], "s--", color=COLOUR_BASE, alpha=0.8, label="Baseline")
    ax.plot(x, df["Q_Corrected"], "^-", color=COLOUR_CORR, alpha=0.8, label="Corrected")
    _xtick(ax, x, events)
    ax.set_ylabel("Peak Discharge (m³/s)")
    ax.set_title("Peak Discharge")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ---- Absolute errors
    ax = axes[0, 1]
    ax.bar(x - w/2, df["Error_Baseline"], w, color=COLOUR_BASE, alpha=0.65, label="Baseline")
    ax.bar(x + w/2, df["Error_Corrected"], w, color=COLOUR_CORR, alpha=0.65, label="Corrected")
    ax.axhline(0, color="k", lw=0.8)
    _xtick(ax, x, events)
    ax.set_ylabel("Error (m³/s)")
    ax.set_title("Absolute Errors")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ---- Scatter
    ax = axes[1, 0]
    lim_max = max(df["Q_Observed"].max(),
                  df[["Q_Baseline", "Q_Corrected"]].max().max()) * 1.1
    ax.scatter(df["Q_Observed"], df["Q_Baseline"], s=60, marker="s",
               color=COLOUR_BASE, alpha=0.75, label="Baseline")
    ax.scatter(df["Q_Observed"], df["Q_Corrected"], s=60, marker="^",
               color=COLOUR_CORR, alpha=0.75, label="Corrected")
    ax.plot([0, lim_max], [0, lim_max], "k--", alpha=0.3, lw=1, label="1:1")
    ax.set_xlim(0, lim_max); ax.set_ylim(0, lim_max)
    ax.set_xlabel("Observed (m³/s)"); ax.set_ylabel("Simulated (m³/s)")
    ax.set_title("Observed vs Simulated")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ---- Relative bias
    ax = axes[1, 1]
    ax.bar(x - w/2, df["Bias_Baseline_%"], w, color=COLOUR_BASE, alpha=0.65, label="Baseline")
    ax.bar(x + w/2, df["Bias_Corrected_%"], w, color=COLOUR_CORR, alpha=0.65, label="Corrected")
    ax.axhline(0, color="k", lw=0.8)
    _xtick(ax, x, events)
    ax.set_ylabel("Relative Bias (%)")
    ax.set_title("Relative Bias")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"✓ Performance summary saved → {output_path}")

    return fig


# ---------------------------------------------------------------------------
# Watershed-mode plots
# ---------------------------------------------------------------------------

def plot_watershed_convergence(
    convergence_df: pd.DataFrame,
    summary: dict,
    output_path: Optional[str | Path] = None,
    figsize: tuple = (10, 4),
) -> plt.Figure:
    """
    Two-panel convergence plot for watershed Newton-Raphson iterations.

    Left  – cumulative alpha per iteration
    Right – MAPE (%) per iteration
    """
    df = convergence_df.copy()
    iters = df["iteration"].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(
        f"Watershed Correction Convergence  "
        f"(converged={summary.get('converged', '?')}, "
        f"α_final={summary.get('final_alpha_cum', 0):.4f})",
        fontsize=12, fontweight="bold",
    )

    # Alpha
    ax1.plot(iters, df["alpha_cum"], "o-", color=COLOUR_CORR, lw=2, ms=7)
    ax1.axhline(1.0, color="gray", lw=0.8, ls="--", label="α = 1 (no change)")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Cumulative α")
    ax1.set_title("Cumulative Correction Factor")
    ax1.set_xticks(iters)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # MAPE
    ax2.plot(iters, df["mape_percent"], "s-", color=COLOUR_BASE, lw=2, ms=7)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("MAPE (%)")
    ax2.set_title("Mean Absolute Percentage Error")
    ax2.set_xticks(iters)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"✓ Convergence plot saved → {output_path}")

    return fig


def plot_watershed_performance(
    perf_iterations: Dict[int, pd.DataFrame],
    output_path: Optional[str | Path] = None,
    figsize: tuple = (13, 9),
) -> plt.Figure:
    """
    2×2 diagnostic grid using per-iteration Performance.csv files.

    Top-left  – peak discharge bar chart (obs vs sim, per event)
    Top-right – absolute % error per event across iterations
    Bottom-left – observed vs simulated scatter (final iteration)
    Bottom-right – improvement in error from iter 1 → final
    """
    if not perf_iterations:
        raise ValueError("perf_iterations dict is empty")

    final_iter = max(perf_iterations.keys())
    df_final   = perf_iterations[final_iter].copy()
    events     = df_final["event_id"].values
    x          = np.arange(len(events))
    w          = 0.35

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(
        f"Watershed Bias Correction – Performance (final iter = {final_iter})",
        fontsize=13, fontweight="bold",
    )

    # ---- Peak discharge bar
    ax = axes[0, 0]
    ax.bar(x - w / 2, df_final["Q_obs_m3s"],  w, color=COLOUR_OBS,  alpha=0.9, label="Observed")
    ax.bar(x + w / 2, df_final["Q_sim_m3s"],  w, color=COLOUR_CORR, alpha=0.75, label="Corrected")
    _xtick(ax, x, events)
    ax.set_ylabel("Peak Discharge (m³/s)")
    ax.set_title("Peak Discharge – Observed vs Corrected")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    # ---- % error per event, one line per iteration
    ax = axes[0, 1]
    colors_iter = plt.cm.Blues(np.linspace(0.4, 0.9, len(perf_iterations)))
    for (it, df_it), col in zip(sorted(perf_iterations.items()), colors_iter):
        lw = 2.5 if it == final_iter else 1.0
        ls = "-" if it == final_iter else "--"
        ax.plot(x, df_it["abs_pct_error"].values, "o" + ls,
                color=col, lw=lw, ms=5, label=f"Iter {it}")
    ax.axhline(20, color="gray", lw=0.8, ls=":", label="20% target")
    _xtick(ax, x, events)
    ax.set_ylabel("Abs. % Error")
    ax.set_title("Per-Event Error by Iteration")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- Scatter obs vs sim (final iteration)
    ax = axes[1, 0]
    lim = max(df_final["Q_obs_m3s"].max(), df_final["Q_sim_m3s"].max()) * 1.15
    ax.scatter(df_final["Q_obs_m3s"], df_final["Q_sim_m3s"],
               s=80, color=COLOUR_CORR, edgecolors="gray", lw=0.5, zorder=3)
    ax.plot([0, lim], [0, lim], "k--", alpha=0.3, lw=1, label="1:1")
    for _, row in df_final.iterrows():
        ax.annotate(row["event_id"], (row["Q_obs_m3s"], row["Q_sim_m3s"]),
                    textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Observed (m³/s)")
    ax.set_ylabel("Simulated (m³/s)")
    ax.set_title("Observed vs Simulated (final)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- Error improvement (iter1 → final)
    ax = axes[1, 1]
    first_iter = min(perf_iterations.keys())
    if first_iter != final_iter:
        df_first = perf_iterations[first_iter]
        err1 = df_first.set_index("event_id")["abs_pct_error"]
        err_f = df_final.set_index("event_id")["abs_pct_error"]
        improvement = (err1 - err_f).reindex(events).values
        colours_imp = [COLOUR_CORR if v >= 0 else COLOUR_BASE for v in improvement]
        ax.bar(x, improvement, color=colours_imp, alpha=0.8)
        ax.axhline(0, color="k", lw=0.8)
        _xtick(ax, x, events)
        ax.set_ylabel("Error Reduction (pp)")
        ax.set_title(f"Error Improvement: Iter {first_iter} → {final_iter}")
        ax.grid(True, axis="y", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Only 1 iteration —\nno improvement to show",
                ha="center", va="center", transform=ax.transAxes, fontsize=11, color="gray")
        ax.set_title("Error Improvement")

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"✓ Watershed performance plot saved → {output_path}")

    return fig


def plot_cluster_analysis(
    perf_df: pd.DataFrame,
    event_alphas: Dict[str, float],
    imerg_totals: Dict[str, float],
    output_path: Optional[str | Path] = None,
    figsize: tuple = (14, 6),
) -> plt.Figure:
    """
    Two-panel cluster diagnostic figure.

    Panel A (left)  — scatter: IMERG event total (mm) vs observed peak discharge
                      coloured by bias cluster (overestimation / balanced / underestimation)
    Panel B (right) — bar: per-event α* coloured by cluster, reference line at α=1

    Parameters
    ----------
    perf_df        : Performance.csv DataFrame (columns: event_id, Q_obs_m3s, abs_pct_error)
    event_alphas   : {event_id: α*} from Alpha_Scalar.json event_alphas field
    imerg_totals   : {event_id: total_mm} average IMERG rainfall across stations
    """
    COLOUR_OVER  = "#e84040"   # overestimation: α < 0.70
    COLOUR_BAL   = "#2ca02c"   # balanced:       0.70 ≤ α ≤ 1.20
    COLOUR_UNDER = "#1f77b4"   # underestimation: α > 1.20

    def _cluster_colour(a: float) -> str:
        if a < 0.70:
            return COLOUR_OVER
        if a > 1.20:
            return COLOUR_UNDER
        return COLOUR_BAL

    def _cluster_label(a: float) -> str:
        if a < 0.70:
            return "Overestimation (α<0.70)"
        if a > 1.20:
            return "Underestimation (α>1.20)"
        return "Balanced (0.70≤α≤1.20)"

    events  = perf_df["event_id"].tolist()
    q_obs   = perf_df["Q_obs_m3s"].tolist()
    alphas  = [event_alphas.get(e, 1.0) for e in events]
    totals  = [imerg_totals.get(e, float("nan")) for e in events]
    colours = [_cluster_colour(a) for a in alphas]
    years   = [e.split("_")[2][:4] if "_" in e else "" for e in events]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # ---- Panel A: scatter IMERG total vs Q_obs ----
    ax = axes[0]
    for label, colour in [
        ("Overestimation (α<0.70)", COLOUR_OVER),
        ("Balanced (0.70≤α≤1.20)", COLOUR_BAL),
        ("Underestimation (α>1.20)", COLOUR_UNDER),
    ]:
        mask = [_cluster_label(a) == label for a in alphas]
        xs = [t for t, m in zip(totals, mask) if m]
        ys = [q for q, m in zip(q_obs,  mask) if m]
        ax.scatter(xs, ys, c=colour, label=label, s=70, zorder=3, edgecolors="k", lw=0.5)

    for i, (x, y, yr) in enumerate(zip(totals, q_obs, years)):
        ax.annotate(yr, (x, y), textcoords="offset points", xytext=(4, 3),
                    fontsize=7, color="#555555")

    ax.set_xlabel("IMERG event total (mm, avg over 7 stations)", fontsize=10)
    ax.set_ylabel("Observed peak discharge (m³/s)", fontsize=10)
    ax.set_title("IMERG Rainfall vs Observed Discharge\n(coloured by bias cluster)", fontsize=11)
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # ---- Panel B: per-event α* bar chart ----
    ax = axes[1]
    x = np.arange(len(events))
    bars = ax.bar(x, alphas, color=colours, alpha=0.85, edgecolor="k", linewidth=0.4)
    ax.axhline(1.0, color="k", lw=1.2, ls="--", label="α = 1 (no correction)")
    ax.axhline(0.70, color=COLOUR_OVER,  lw=0.8, ls=":", alpha=0.7)
    ax.axhline(1.20, color=COLOUR_UNDER, lw=0.8, ls=":", alpha=0.7)

    short = [e.replace("event_", "").replace("_flood", "") for e in events]
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=55, ha="right", fontsize=6)
    ax.set_ylabel("Per-event correction factor α*", fontsize=10)
    ax.set_title("Per-event α* — bimodal distribution\nshows single scalar cannot resolve all events",
                 fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, max(alphas) * 1.15)

    fig.suptitle("IMERG Bias Cluster Analysis — Bagmati at Khokana", fontsize=12, y=1.01)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"✓ Cluster analysis plot saved → {output_path}")

    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _xtick(ax: plt.Axes, x: np.ndarray, labels) -> None:
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
