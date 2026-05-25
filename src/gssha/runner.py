#!/usr/bin/env python3
"""
GSSHA Runner
Sets up a per-simulation scratch directory, executes GSSHA, and returns
the peak discharge from the outlet hydrograph.

Responsibilities
----------------
* Copy/symlink base-model files into a fresh run directory.
* Replace the GAG file with the caller-supplied one.
* Normalise all paths in the .prj / .cmt / .smt files (Windows → Linux).
* Execute gssha81 with the requested OpenMP thread count.
* Parse the .otl output file and return peak discharge [m³/s].

Thread-safety
-------------
Every simulation gets its own temporary directory, so multiple instances
can be run in parallel via ``multiprocessing.Pool``.
"""

from __future__ import annotations

import os
import re
import resource
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple


class GsshaRunner:
    """Execute GSSHA simulations and extract peak discharge."""

    def __init__(
        self,
        base_model_dir: str | Path,
        gssha_executable: str = "gssha81",
    ):
        """
        Parameters
        ----------
        base_model_dir  : directory containing the baseline GSSHA model
        gssha_executable: name or full path of the GSSHA binary
        """
        self.base_model_dir = Path(base_model_dir).resolve()
        self.gssha_executable = gssha_executable
        self.project_file = self._find_project_file()

        if not self.base_model_dir.exists():
            raise FileNotFoundError(f"Base model directory not found: {base_model_dir}")
        if self.project_file is None:
            raise FileNotFoundError(f"No .prj file found in {base_model_dir}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_simulation(
        self,
        gag_file_path: str | Path,
        run_dir: Optional[str | Path] = None,
        cleanup: bool = False,
        threads: int = 48,
    ) -> Tuple[float, str]:
        """
        Run one GSSHA simulation.

        Parameters
        ----------
        gag_file_path : GAG file to use as rainfall input
        run_dir       : directory for model files (temporary if None)
        cleanup       : remove run_dir after extracting results
        threads       : OpenMP thread count

        Returns
        -------
        (peak_discharge_m3s, run_directory_path)
        """
        gag_file_path = Path(gag_file_path)

        if run_dir is None:
            run_dir = Path(tempfile.mkdtemp(prefix="gssha_run_"))
            temp_created = True
        else:
            run_dir = Path(run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            temp_created = False

        try:
            self._setup_run_directory(run_dir, gag_file_path)
            self._execute_gssha(run_dir, threads)
            peak_q = self._extract_peak_discharge(run_dir)
            return peak_q, str(run_dir)

        except Exception:
            print(f"\n⚠️  Run directory preserved for debugging: {run_dir}")
            raise

        finally:
            if cleanup and temp_created:
                shutil.rmtree(run_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_project_file(self) -> Optional[Path]:
        prj_files = list(self.base_model_dir.glob("*.prj"))
        return prj_files[0] if prj_files else None

    def _convert_line_endings(self, file_path: Path) -> None:
        """Convert CRLF → LF in-place."""
        try:
            content = file_path.read_bytes().replace(b"\r\n", b"\n")
            file_path.write_bytes(content)
        except Exception as exc:
            print(f"Warning: line-ending conversion failed for {file_path}: {exc}")

    def _setup_run_directory(self, run_dir: Path, gag_file_path: Path) -> None:
        """Copy model files into run_dir and inject the caller's GAG file."""
        copy_exts = {".prj", ".cif", ".gst", ".cmt", ".smt", ".ihl",
                     ".ohl", ".lsf", ".sto", ".pro", ".idx", ".idx2"}
        link_exts = {".dep", ".wms", ".cdq", ".cdp", ".map", ".ele", ".msk"}
        text_exts = {".prj", ".cif", ".gst", ".cmt", ".smt", ".gag",
                     ".ihl", ".ohl", ".lsf", ".sto", ".pro"}

        for src in self.base_model_dir.iterdir():
            if not src.is_file():
                continue
            dest = run_dir / src.name
            if src.suffix == ".gag":
                continue  # replaced below
            if src.suffix in link_exts:
                if not dest.exists():
                    os.symlink(src, dest)
            else:
                shutil.copy2(src, dest)
                if src.suffix in text_exts:
                    self._convert_line_endings(dest)

        # Inject the specified GAG file
        gag_dest = run_dir / (self.project_file.stem + ".gag")
        shutil.copy2(gag_file_path, gag_dest)
        self._convert_line_endings(gag_dest)

        # Rewrite paths in project and mapping files
        for prj in [run_dir / self.project_file.name]:
            if prj.exists():
                self._update_paths(prj, run_dir)
        for ext in ("*.cmt", "*.smt"):
            for f in run_dir.glob(ext):
                self._update_paths(f, run_dir)

    def _update_paths(self, file_path: Path, run_dir: Path) -> None:
        """Replace old absolute paths in a text config file with run_dir paths."""
        run_dir = run_dir.resolve()
        lines = file_path.read_text(errors="replace").splitlines(keepends=True)

        def replace_path(match: re.Match) -> str:
            old = match.group(1)
            if "\\" in old or "/" in old or "C:" in old:
                if old.endswith("\\") or old.endswith("/"):
                    return f'"{run_dir}/"'
                filename = old.replace("\\", "/").split("/")[-1]
                if filename and ("." in filename or filename in ("maskmap", "satsource_file")):
                    return f'"{run_dir / filename}"'
            return match.group(0)

        updated = [re.sub(r'"([^"]+)"', replace_path, line) for line in lines]
        file_path.write_text("".join(updated))

    def _execute_gssha(self, run_dir: Path, threads: int) -> None:
        """Launch GSSHA, streaming stdout/stderr to log files."""
        prj_file = run_dir / self.project_file.name
        env = os.environ.copy()
        env.update({
            "OMP_NUM_THREADS": str(threads),
            "OMP_STACKSIZE": "512M",
            "OMP_WAIT_POLICY": "PASSIVE",
        })

        def _set_stack():
            try:
                resource.setrlimit(
                    resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
                )
            except Exception:
                pass

        logs_dir = run_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        log_file = logs_dir / "gssha_run.log"
        err_file = logs_dir / "gssha_error.log"

        with open(log_file, "w") as log, open(err_file, "w") as err:
            result = subprocess.run(
                [self.gssha_executable, str(prj_file)],
                cwd=str(run_dir),
                stdout=log,
                stderr=err,
                env=env,
                preexec_fn=_set_stack,
                timeout=7200,    # 2-hour hard cap; typical run ~1400s at 4 threads
            )

        if result.returncode != 0:
            stdout_tail = log_file.read_text()[-500:]
            stderr_tail = err_file.read_text()[-500:]
            raise RuntimeError(
                f"GSSHA exited with code {result.returncode}\n"
                f"STDOUT:\n{stdout_tail}\nSTDERR:\n{stderr_tail}"
            )

    def _extract_peak_discharge(self, run_dir: Path) -> float:
        """Parse .otl / .ihl output and return the maximum discharge value."""
        output_file: Optional[Path] = None
        for pattern in ("*.otl", "*.ihl", "*_out_node.txt"):
            matches = list(run_dir.glob(pattern))
            if matches:
                output_file = matches[0]
                break

        if output_file is None:
            raise FileNotFoundError(f"No hydrograph output file found in {run_dir}")

        peak = 0.0
        with open(output_file, "r") as fh:
            for line in fh:
                if line.startswith(("#", "TIME")) or not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        peak = max(peak, float(parts[1]))
                    except ValueError:
                        pass

        if peak == 0.0:
            raise ValueError(f"Could not read a non-zero peak discharge from {output_file}")

        return peak

    def __repr__(self) -> str:
        return f"GsshaRunner(model={self.base_model_dir.name}, prj={self.project_file.name})"


def run_gssha_simple(
    base_model_dir: str | Path,
    gag_file: str | Path,
    threads: int = 48,
) -> float:
    """One-liner wrapper: run GSSHA and return peak discharge [m³/s]."""
    runner = GsshaRunner(base_model_dir)
    peak_q, _ = runner.run_simulation(gag_file, threads=threads, cleanup=True)
    return peak_q
