# TODO

Tracks outstanding work, open questions, and future improvements.

---

## Critical / Blocking

- [ ] **Verify GAG file availability** — confirm all 25 `event_NNN_*.gag` files are
      in `Stations Based Correction/Data/IMERG/` after running `02_download_imerg.py`.
- [ ] **Rating curve coverage** — validate that `Rating Curve.txt` covers the stage
      range of all 25 flood events (especially the April 2024 event with stage=7.08 m).
- [ ] **GSSHA base model** — confirm `Models/Model-9_17/Khokana.prj` runs cleanly on
      the current system before starting the bias correction loop.

---

## Data Pipeline

- [ ] Run `01_extract_discharge.py` and verify per-event discharge CSVs are correct.
- [ ] Run `02_download_imerg.py` for all 25 events.
- [ ] Check time window consistency: IMERG download window should start 24 h before
      the flood peak and end 24 h after.
- [ ] Investigate events with missing observed discharge (stage data gaps).

---

## Bias Correction

- [ ] Test elasticity estimation on 2–3 events before running all 25.
- [ ] Tune `lambda_smooth` and `tau_prior` in `configs/system.yaml`; plot
      alpha stability vs regularisation strength.
- [ ] Consider adding a minimum-rainfall threshold to suppress noise from
      near-zero IMERG values in the elasticity perturbation.
- [ ] Evaluate whether event_001 (2013-05-23) has enough IMERG coverage
      (pre-IMERG V07 era — some gaps likely).

---

## Validation

- [ ] Compute NSE (Nash–Sutcliffe Efficiency) in addition to RMSE/bias.
- [ ] Plot full discharge hydrographs (not just peak Q) for a subset of events.
- [ ] Compare station-based α with spatial average from iterative correction
      (`Bias_Correction_Results/iteration_02/`).

---

## Code Quality

- [ ] Add unit tests for `GagFile`, `RatingCurve`, `StationOptimizer`.
- [ ] Add `src/` to `PYTHONPATH` via `pyproject.toml` (or `setup.py`).
- [ ] Fix `scripts/00_setup_paths.py` `abs_path` usage in scripts (currently
      each script loads configs independently — harmonise).
- [ ] Remove `__pycache__` directories from version control (add to `.gitignore`).

---

## Documentation

- [ ] Add docstrings to `src/data_acquisition/gag_converter.py` `batch_convert_csvs`.
- [ ] Write `docs/architecture.md` diagram with actual α flow numbers from iteration 2.
- [ ] Write `docs/gssha_model.md` describing the base model setup.

---

## Future Improvements

- [ ] Seasonal α estimation (monsoon vs pre-monsoon events may need different factors).
- [ ] Uncertainty quantification for α factors via bootstrap.
- [ ] Compare IMERG V07 vs IMERG V06 bias characteristics.
- [ ] Extend to Bagmati upstream gauges once stage data is available.
- [ ] Automate the full pipeline with a Makefile or Prefect/Airflow workflow.

---

## Archive / Cleanup

- [ ] Confirm `Stations Based Correction copy/` can be deleted (exact duplicate).
- [ ] Move `Notebooks/Final Plots/*.py` to `src/visualization/` or `scripts/`.
- [ ] Delete `Notebooks/IMERG/__pycache__` and `Archieve/Bias_Correction/__pycache__`.
