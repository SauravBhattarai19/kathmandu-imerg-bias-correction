#!/usr/bin/env python3
"""
Path/config loader shared by all scripts.

Usage
-----
    from scripts.setup_paths import ROOT, cfg, paths_cfg, sys_cfg
"""

from __future__ import annotations

from pathlib import Path

import yaml

# Project root = parent of this scripts/ directory
ROOT = Path(__file__).resolve().parent.parent


def _load(config_file: str) -> dict:
    p = ROOT / "configs" / config_file
    with open(p) as fh:
        return yaml.safe_load(fh)


paths_cfg  = _load("paths.yaml")
sys_cfg    = _load("system.yaml")

# ---------------------------------------------------------------------------
# Derived absolute paths (resolved against ROOT)
# ---------------------------------------------------------------------------


def abs_path(rel: str) -> Path:
    """Resolve a project-relative path string to an absolute Path."""
    p = Path(rel)
    if p.is_absolute():
        return p
    return ROOT / p


# Frequently used paths, pre-resolved
DATA_RAW        = abs_path(paths_cfg["data"]["raw"]["imerg_raw_dir"])
STAGE_CSV       = abs_path(paths_cfg["data"]["raw"]["stage_csv"])
RATING_CURVE    = abs_path(paths_cfg["data"]["raw"]["rating_curve"])
FLOOD_CSV       = abs_path(paths_cfg["data"]["processed"]["flood_csv"])
BASE_MODEL      = abs_path(paths_cfg["model"]["sbc"])
IMERG_GAG_DIR   = abs_path(paths_cfg["data"]["processed"]["imerg_gag_dir"])
DHM_DIR         = abs_path(paths_cfg["data"]["processed"]["dhm_dir"])
RUNS_ROOT       = abs_path(paths_cfg["outputs"]["runs_root"])
LATEST_LINK     = abs_path(paths_cfg["outputs"]["latest_link"])

# System params
GEE_PROJECT     = sys_cfg["gee"]["project"]
THREADS         = sys_cfg["gssha"]["threads_per_run"]
PARALLEL_RUNS   = sys_cfg["gssha"]["parallel_runs"]
DELTA           = sys_cfg["bias_correction"]["delta"]
LAMBDA_SMOOTH   = sys_cfg["bias_correction"]["lambda_smooth"]
TAU_PRIOR       = sys_cfg["bias_correction"]["tau_prior"]
