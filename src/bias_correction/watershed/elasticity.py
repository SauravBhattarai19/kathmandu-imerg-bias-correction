#!/usr/bin/env python3
"""
Watershed-Scale Elasticity Estimator

Runs 3 GSSHA simulations per flood event (baseline, +δ, -δ) in parallel
and returns the per-event Q0 / Q+ / Q- needed by the Newton-Raphson solver.

GSSHA parallelism on AMD Threadripper PRO 7995WX (192 logical cores)
--------------------------------------------------------------------
GSSHA 8.1 is OpenMP-only. When OMP_NUM_THREADS is NOT set it auto-grabs
ALL available cores. With 48 parallel processes we MUST set it explicitly
to avoid 48 × 192 = 9216 threads on 192 cores (catastrophic over-subscription).

Proven optimal (project HPC guide, validated on this machine):
    48 parallel runs × 4 threads/run = 192 cores fully utilised

Alternative (max speed per run, fewer parallel slots):
    4 parallel runs × 48 threads/run = 192 cores
    → better when n_events is small (<10); worse for 25+ events

Run-directory strategy
----------------------
  baseline (_base)  → permanent dir: iteration_k/sim/{event_id}/
      GSSHA writes Khokana.otl there in real time, timestep by timestep.
      Watch any run:  tail -f iteration_k/sim/{event_id}/Khokana.otl
      Kept after run for validation and debugging.

  plus-delta  (_plus)
  minus-delta (_minus) → ephemeral /tmp dirs, cleaned up automatically.
      Only peak Q is needed; full output discarded.

Progress monitoring
-------------------
A background thread reads the last line of every active Khokana.otl every
30 s and prints the current simulation time and Qout for each event.
This gives real-time visibility without any extra GSSHA runs.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.gssha.gag_editor import GagFile
from src.gssha.runner import GsshaRunner


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class EventResult:
    event_id:         str
    Q0:               float          # baseline  peak Q [m³/s]
    Q_plus:           float          # +δ        peak Q [m³/s]
    Q_minus:          float          # -δ        peak Q [m³/s]
    baseline_run_dir: str = field(default="")   # permanent dir; Khokana.otl lives here


# ---------------------------------------------------------------------------
# Pool worker — top-level for pickling
# ---------------------------------------------------------------------------

def _run_one_job(args: tuple) -> tuple:
    """
    Execute a single GSSHA simulation.

    args = (base_model_dir, gag_path, threads, label, run_dir_or_None)

    run_dir_or_None = None   → temp dir, auto-cleaned (plus/minus runs)
    run_dir_or_None = str    → permanent dir, kept   (baseline runs)

    Returns
    -------
    (label, peak_q, elapsed_s, actual_run_dir, error_or_None)
    """
    base_model_dir, gag_path, threads, label, run_dir = args
    t0 = time.time()
    cleanup = (run_dir is None)
    try:
        runner = GsshaRunner(base_model_dir)
        peak_q, actual_run_dir = runner.run_simulation(
            gag_path,
            run_dir=run_dir,
            cleanup=cleanup,
            threads=threads,
        )
        return label, peak_q, time.time() - t0, actual_run_dir, None
    except Exception as exc:
        return label, float("nan"), time.time() - t0, run_dir or "", str(exc)


# ---------------------------------------------------------------------------
# OTL progress monitor
# ---------------------------------------------------------------------------

def _read_otl_progress(otl_path: Path, log_path: Optional[Path] = None) -> Optional[dict]:
    """
    Read progress from a GSSHA .otl file.

    .otl format (two space-separated columns, no header):
        <elapsed_sim_min>    <Qout_m3s>
    e.g.
        1780.00    1697.33982386

    Optionally reads total simulation duration from the companion run log:
        logs/gssha_run.log  →  "simulation duration 5700 minutes"

    Returns dict with keys: elapsed_min, qout_cms, total_min (or None),
    pct (or None), or None if file unreadable.
    """
    if not otl_path.exists() or otl_path.stat().st_size == 0:
        return None
    try:
        # Read only the last ~256 bytes to get the last line efficiently
        with open(otl_path, "rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - 256), 0)
            tail = fh.read().decode("utf-8", errors="ignore")
        # Last non-empty line
        lines = [l.strip() for l in tail.splitlines() if l.strip()]
        if not lines:
            return None
        parts = lines[-1].split()
        if len(parts) < 2:
            return None
        elapsed_min = float(parts[0])
        qout_cms    = float(parts[1])
    except Exception:
        return None

    # Try to get total duration from run log
    total_min = None
    pct       = None
    if log_path is None and otl_path.parent.exists():
        log_path = otl_path.parent / "logs" / "gssha_run.log"
    if log_path and log_path.exists():
        try:
            with open(log_path, "r", errors="ignore") as fh:
                for line in fh:
                    if "simulation duration" in line:
                        # "simulation duration 5700 minutes"
                        parts2 = line.split()
                        idx = parts2.index("duration") + 1
                        total_min = float(parts2[idx])
                        break
        except Exception:
            pass
    if total_min and total_min > 0:
        pct = elapsed_min / total_min * 100.0

    return {
        "elapsed_min": elapsed_min,
        "qout_cms":    qout_cms,
        "total_min":   total_min,
        "pct":         pct,
    }


def _otl_monitor(
    otl_paths: Dict[str, Path],
    stop_event: threading.Event,
    interval: float = 30.0,
) -> None:
    """
    Background thread: prints .otl progress every `interval` seconds.

    .otl format: two columns — elapsed_sim_min  Qout_m3s
    Total duration read once from logs/gssha_run.log per event.
    """
    total_mins: Dict[str, Optional[float]] = {}   # cached per event

    while not stop_event.wait(interval):
        rows = []
        for event_id, otl_path in sorted(otl_paths.items()):
            info = _read_otl_progress(otl_path)
            if info is None:
                rows.append(f"    {event_id:<33}  initialising …")
                continue

            # Cache total duration once it's available
            if event_id not in total_mins:
                total_mins[event_id] = info.get("total_min")

            elapsed = info["elapsed_min"]
            qout    = info["qout_cms"]
            total   = total_mins.get(event_id) or info.get("total_min")

            if total:
                pct      = elapsed / total * 100.0
                bar_fill = int(pct / 5)          # 20-char bar
                bar      = "█" * bar_fill + "░" * (20 - bar_fill)
                rows.append(
                    f"    {event_id:<33}  [{bar}] {pct:5.1f}%"
                    f"  {elapsed:6.0f}/{total:.0f} min"
                    f"  Q={qout:8.2f} m³/s"
                )
            else:
                rows.append(
                    f"    {event_id:<33}  sim={elapsed:7.1f} min"
                    f"  Q={qout:8.2f} m³/s"
                )

        print(
            f"\n  ┌─ GSSHA progress @ {datetime.now().strftime('%H:%M:%S')} "
            + "─" * 35,
            flush=True,
        )
        for row in rows:
            print(row, flush=True)
        print("  └" + "─" * 70, flush=True)


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------

class WatershedElasticityEstimator:
    """
    Compute per-event Q0, Q+, Q- with parallel GSSHA runs.

    Baseline runs use permanent directories so Khokana.otl is visible
    in real time. A background monitor thread prints progress every 30 s.
    """

    def __init__(
        self,
        base_model_dir: str | Path,
        delta: float = 0.10,
        threads_per_run: int = 4,
        parallel_runs: int = 48,
    ):
        self.base_model_dir = Path(base_model_dir)
        self.delta           = delta
        self.threads         = threads_per_run
        self.parallel_runs   = parallel_runs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        gag_files: List[str],
        event_ids: List[str],
        alpha_cum: float,
        work_dir: Path,
        sim_dir: Optional[Path] = None,
        dry_run: bool = False,
    ) -> List[EventResult]:
        """
        Run baseline / +δ / -δ GSSHA simulations for all events.

        Parameters
        ----------
        gag_files  : original (uncorrected) .gag paths, one per event
        event_ids  : matching event identifiers
        alpha_cum  : current cumulative scalar applied uniformly
        work_dir   : directory for scaled input GAG files (temp, caller-managed)
        sim_dir    : if given, baseline runs land in sim_dir/{event_id}/
                     (permanent — Khokana.otl is watchable here).
        dry_run    : skip GSSHA, return synthetic data

        Returns
        -------
        List[EventResult], one per event; Q values NaN on failure.
        """
        if dry_run:
            return self._synthetic_results(event_ids)

        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        # Build job list — baselines first so all N events start simultaneously.
        # ALL run dirs (including plus/minus temp dirs) are pre-created so the
        # OTL monitor can track every job from the very first pool batch.
        # Order: all _base → all _plus → all _minus
        baseline_run_dir_map: Dict[str, str] = {}
        all_otl_map:          Dict[str, Path] = {}   # ALL 3N jobs monitored
        temp_dirs_to_cleanup: List[str]       = []   # plus/minus ephemeral dirs

        base_jobs:  list = []
        delta_jobs: list = []   # plus and minus

        for event_id, orig_gag in zip(event_ids, gag_files):
            gag = GagFile(orig_gag)
            for scenario, scalar in (
                ("base",  alpha_cum),
                ("plus",  alpha_cum * (1.0 + self.delta)),
                ("minus", alpha_cum * (1.0 - self.delta)),
            ):
                scaled = gag.scale_uniform(scalar)
                out_path = work_dir / f"{event_id}_{scenario}.gag"
                scaled.write(out_path)

                if scenario == "base" and sim_dir is not None:
                    run_dir_for_job = str(sim_dir / event_id)
                    baseline_run_dir_map[event_id] = run_dir_for_job
                else:
                    # Pre-create named temp dir — path known upfront → monitorable
                    tmp = tempfile.mkdtemp(prefix=f"gssha_{event_id}_{scenario}_")
                    run_dir_for_job = tmp
                    temp_dirs_to_cleanup.append(tmp)

                symbol = {"base": "base", "plus": "+δ  ", "minus": "-δ  "}[scenario]
                all_otl_map[f"{event_id} ({symbol})"] = (
                    Path(run_dir_for_job) / "Khokana.otl"
                )

                job = (
                    str(self.base_model_dir),
                    str(out_path),
                    self.threads,
                    f"{event_id}_{scenario}",
                    run_dir_for_job,
                )
                if scenario == "base":
                    base_jobs.append(job)
                else:
                    delta_jobs.append(job)

        # All baselines fill the first pool batch → all events monitored immediately
        jobs = base_jobs + delta_jobs

        total     = len(jobs)
        n_events  = len(event_ids)
        pool_size = min(self.parallel_runs, total)

        print(
            f"\n  ┌{'─'*70}",
            flush=True,
        )
        print(
            f"  │  GSSHA PARALLEL RUNS",
            flush=True,
        )
        print(
            f"  │  {n_events} events × 3 scenarios = {total} jobs",
            flush=True,
        )
        print(
            f"  │  Pool size : {pool_size}  |  threads/run : {self.threads}  "
            f"|  total cores : {pool_size * self.threads}",
            flush=True,
        )
        if sim_dir:
            print(
                f"  │  Baseline dirs (kept) → {sim_dir}/<event_id>/Khokana.otl",
                flush=True,
            )
            print(
                f"  │  Watch live:  tail -f {sim_dir}/<event_id>/Khokana.otl",
                flush=True,
            )
        print(
            f"  │  All {total} jobs monitored (base + ±δ) every 30 s",
            flush=True,
        )
        print(f"  └{'─'*70}\n", flush=True)

        print(
            f"  {'#':>5}  {'LABEL':<35}  {'Q_peak':>9}  {'TIME':>8}  STATUS",
            flush=True,
        )
        print(
            f"  {'─'*5}  {'─'*35}  {'─'*9}  {'─'*8}  {'─'*6}",
            flush=True,
        )

        # Start OTL progress monitor — watches ALL 3N jobs (base + ±δ)
        stop_monitor  = threading.Event()
        monitor_thread = threading.Thread(
            target=_otl_monitor,
            args=(all_otl_map, stop_monitor, 30.0),
            daemon=True,
        )
        monitor_thread.start()

        results_map: Dict[str, float] = {}
        completed = 0
        failed    = 0
        t_wall_start = time.time()

        try:
            with Pool(processes=pool_size) as pool:
                for label, peak_q, elapsed, actual_run_dir, err in \
                        pool.imap_unordered(_run_one_job, jobs):
                    completed += 1
                    if err:
                        failed += 1
                        status = "FAIL"
                        q_str  = "      NaN"
                    else:
                        status = "ok"
                        q_str  = f"{peak_q:9.2f}"

                    print(
                        f"  [{completed:>3}/{total}]  {label:<35}  {q_str} m³/s"
                        f"  {elapsed:7.1f}s  {status}",
                        flush=True,
                    )
                    results_map[label] = peak_q
        finally:
            stop_monitor.set()
            monitor_thread.join(timeout=2.0)
            # Clean up plus/minus temp dirs (baseline dirs kept for inspection)
            for tmp in temp_dirs_to_cleanup:
                shutil.rmtree(tmp, ignore_errors=True)

        wall = time.time() - t_wall_start
        print(
            f"\n  All {total} runs finished in {wall:.1f}s  ({failed} failed)\n",
            flush=True,
        )

        # Per-event summary table
        print(
            f"  {'EVENT':<35}  {'Q0':>8}  {'Q+':>8}  {'Q-':>8}",
            flush=True,
        )
        print(f"  {'─'*35}  {'─'*8}  {'─'*8}  {'─'*8}", flush=True)

        event_results: List[EventResult] = []
        for event_id in event_ids:
            Q0     = results_map.get(f"{event_id}_base",  float("nan"))
            Q_plus = results_map.get(f"{event_id}_plus",  float("nan"))
            Q_minus= results_map.get(f"{event_id}_minus", float("nan"))

            ok  = not (np.isnan(Q0) or np.isnan(Q_plus) or np.isnan(Q_minus))
            sym = "✓" if ok else "✗"
            print(
                f"  {sym} {event_id:<33}  {Q0:8.2f}  {Q_plus:8.2f}  {Q_minus:8.2f}",
                flush=True,
            )

            event_results.append(EventResult(
                event_id=event_id,
                Q0=Q0,
                Q_plus=Q_plus,
                Q_minus=Q_minus,
                baseline_run_dir=baseline_run_dir_map.get(event_id, ""),
            ))

        return event_results

    # ------------------------------------------------------------------
    # Dry-run helper
    # ------------------------------------------------------------------

    @staticmethod
    def _synthetic_results(event_ids: List[str]) -> List[EventResult]:
        """Reproducible synthetic Q values for dry-run/testing."""
        np.random.seed(42)
        print(
            f"\n  [DRY-RUN] Synthetic Q values — no GSSHA.\n",
            flush=True,
        )
        print(f"  {'EVENT':<35}  {'Q0':>8}  {'Q+':>8}  {'Q-':>8}", flush=True)
        print(f"  {'─'*35}  {'─'*8}  {'─'*8}  {'─'*8}", flush=True)
        results = []
        for i, eid in enumerate(event_ids):
            Q0      = 50.0 + i * 15.0
            Q_plus  = Q0 * 1.35
            Q_minus = Q0 * 0.68
            print(
                f"  ~ {eid:<33}  {Q0:8.2f}  {Q_plus:8.2f}  {Q_minus:8.2f}",
                flush=True,
            )
            results.append(EventResult(
                event_id=eid,
                Q0=Q0,
                Q_plus=Q_plus,
                Q_minus=Q_minus,
                baseline_run_dir="",
            ))
        return results
