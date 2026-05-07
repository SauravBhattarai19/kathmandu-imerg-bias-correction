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
# Helpers
# ---------------------------------------------------------------------------

def _xtick(ax: plt.Axes, x: np.ndarray, labels) -> None:
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
