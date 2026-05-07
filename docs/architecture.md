# Architecture

## Project Goal

Bias-correct NASA GPM IMERG V07 half-hourly precipitation at 7 citizen-science
tipping-bucket gauge locations in Kathmandu Valley so that the GSSHA distributed
hydrological model accurately reproduces observed flood peak discharges at the
Bagmati Khokana outlet gauge for 25 historical flood events (2013–2024).

---

## Component Map

```
┌─────────────────────────────────────────────────────────────────┐
│  DATA LAYER                                                     │
│                                                                 │
│  DHM Stage CSV          IMERG (GEE)         Citizen-Science GAG │
│  (Khokana gauge,        (30-min, 0.1°)      (7 stations,        │
│  1990–2025)             NASA GPM V07        2013–2024)          │
└───────┬─────────────────────┬───────────────────────────────────┘
        │                     │
        ▼                     ▼
┌───────────────┐   ┌──────────────────┐
│ RatingCurve   │   │ IMERGDownloader  │
│ (src/preprocessing) │ (src/data_acquisition) │
│               │   │                  │
│ Stage → Q     │   │ GEE → CSV → GAG  │
└───────┬───────┘   └────────┬─────────┘
        │                    │
        ▼                    ▼
┌────────────────────────────────────────┐
│  FloodIdentifier                       │
│  (src/preprocessing/flood_identifier)  │
│  Top-25 events from Q time-series      │
└──────────────────┬─────────────────────┘
                   │  25 flood events × per-event:
                   │  • .gag file (IMERG rainfall input)
                   │  • discharge_{event}.csv (observed Q)
                   │
                   ▼
┌────────────────────────────────────────┐
│  ElasticityEstimator                   │
│  (src/bias_correction/elasticity)      │
│                                        │
│  For each event × station:             │
│    Run GSSHA(baseline)  → Q₀           │
│    Run GSSHA(+10% Sₛ)   → Q⁺           │
│    Run GSSHA(−10% Sₛ)   → Q⁻           │
│    ε[e,s] = (lnQ⁺−lnQ⁻) / (ln1.1−ln0.9) │
│                                        │
│  Output: E matrix (n_events×n_stations)│
└──────────────────┬─────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────┐
│  StationOptimizer                      │
│  (src/bias_correction/optimizer)       │
│                                        │
│  Solve:  min ||E·lnα − b||²           │
│              + λ||Llnα||²             │
│              + τ||lnα−m||²            │
│                                        │
│  b = ln(Q_obs) − ln(Q₀)              │
│  L = spatial Laplacian                 │
│  Output: α vector (n_stations,)        │
└──────────────────┬─────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────┐
│  GagFile.scale_all_stations(α)         │
│  (src/gssha/gag_editor)                │
│  → corrected .gag files                │
└──────────────────┬─────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────┐
│  ValidationRunner                      │
│  (src/bias_correction/validator)       │
│                                        │
│  GSSHA(corrected .gag) → Q_corr        │
│  Compare Q_corr vs Q_obs               │
│  Metrics: RMSE, MAE, relative bias     │
│  Plots: Validation_Plots.png           │
└────────────────────────────────────────┘
```

---

## Key Design Decisions

### Why station-wise correction (not event-wide)?

The watershed has strong spatial rainfall gradients.  A single scale factor
per event cannot capture localised orographic enhancement in the northern hills
(TP002, TP003, TP008) vs the valley floor (TP005, TP006, TP007).

### Regularisation

Without regularisation the inverse problem is underdetermined for events where
only a few stations have non-zero rainfall.  The spatial Laplacian penalty
`λ||Llnα||²` enforces smoothly varying corrections across the gauge network.
The prior term `τ||lnα−m||²` prevents extreme corrections where the elasticity
matrix is ill-conditioned.

### Parallel GSSHA execution

Each simulation is fully independent (private run directory + scratch).
`multiprocessing.Pool` scales to the available cores.  On a 192-core HPC node
with 4 threads/run → 48 simultaneous simulations.

### IMERG time convention

IMERG timestamps are in **UTC**.  Nepal Standard Time is UTC+5:45.
All downloads apply `nepal_to_utc()` before querying GEE.  The GAG files
store **Nepal time** (to match the GSSHA model calendar).

---

## Modules and their responsibilities

| Module | Class / Function | Responsibility |
|---|---|---|
| `src/gssha/gag_editor.py` | `GagFile` | Parse / scale / write .gag |
| `src/gssha/runner.py` | `GsshaRunner` | Copy model, exec GSSHA, parse output |
| `src/bias_correction/elasticity.py` | `ElasticityEstimator` | ±δ perturbation runs |
| `src/bias_correction/optimizer.py` | `StationOptimizer` | Regularised least squares |
| `src/bias_correction/validator.py` | `ValidationRunner` | Apply α, re-run, metrics |
| `src/preprocessing/stage_to_discharge.py` | `RatingCurve` | Stage → Q conversion |
| `src/preprocessing/flood_identifier.py` | `FloodIdentifier` | Top-N peak events |
| `src/data_acquisition/imerg_downloader.py` | `IMERGDownloader` | GEE → CSV |
| `src/data_acquisition/gag_converter.py` | `csv_to_gag` | CSV → GSSHA GAG |
| `src/visualization/plots.py` | `plot_*` | Matplotlib figures |

---

## Outputs

| File | Content |
|---|---|
| `Stations Based Correction/Outputs/Elasticity_Matrix.csv` | E[event, station] |
| `Stations Based Correction/Outputs/Alpha_Results.csv` | α per station |
| `Stations Based Correction/Outputs/Optimization_Diagnostics.json` | RMSE before/after |
| `Stations Based Correction/Outputs/Corrected_GAG_Files/*.gag` | Corrected rainfall inputs |
| `Stations Based Correction/Outputs/Corrected_GAG_Files/Validation_Report.csv` | Event-level metrics |
| `Stations Based Correction/Outputs/Corrected_GAG_Files/Validation_Plots.png` | 4-panel diagnostic |
| `outputs/figures/` | Publication figures |
