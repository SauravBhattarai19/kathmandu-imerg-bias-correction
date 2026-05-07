# Kathmandu WMS – IMERG Bias Correction Framework

**Station-wise bias correction of NASA GPM IMERG satellite precipitation
for GSSHA distributed hydrological modelling of Bagmati River at Khokana,
Kathmandu Valley, Nepal.**

---

## Overview

Satellite precipitation products like IMERG systematically under- or
over-estimate rainfall in complex terrain.  This project corrects those
biases using the physical response of the watershed itself:

1. Run the GSSHA hydrodynamic model with unmodified IMERG precipitation.
2. Perturb rainfall at each of the 7 citizen-science gauge locations by ±10 %.
3. Compute the **precipitation elasticity** — how sensitively peak discharge
   responds to each station's rainfall.
4. Solve a regularised inverse problem to find per-station scaling factors α
   that minimise the error between simulated and observed discharge.
5. Apply α to every flood event's `.gag` file and re-run GSSHA to validate.

The method is physically based, event-specific, and accounts for spatial
variability across the 7 gauge locations.

---

## Project Structure

```
19 WMS Kathmandu/
│
├── README.md               ← this file
├── CONTRIBUTING.md
├── TODO.md
├── requirements.txt
│
├── configs/
│   ├── paths.yaml          ← all file/directory paths
│   ├── stations.yaml       ← 7 citizen-science rain gauge metadata
│   ├── system.yaml         ← HPC resources, algorithm parameters
│   └── events/             ← auto-generated per-iteration event configs
│
├── src/                    ← installable Python package
│   ├── gssha/
│   │   ├── runner.py       ← GsshaRunner: setup + execute + extract peak Q
│   │   └── gag_editor.py   ← GagFile: read / scale / write .gag files
│   ├── bias_correction/
│   │   ├── elasticity.py   ← ElasticityEstimator: ±δ perturbation runs
│   │   ├── optimizer.py    ← StationOptimizer: regularised least squares
│   │   └── validator.py    ← ValidationRunner: apply α, re-run, compare
│   ├── preprocessing/
│   │   ├── stage_to_discharge.py  ← RatingCurve + event slicing
│   │   └── flood_identifier.py    ← FloodIdentifier: top-N event selection
│   ├── data_acquisition/
│   │   ├── imerg_downloader.py    ← IMERGDownloader (GEE)
│   │   └── gag_converter.py       ← CSV → GSSHA .gag format
│   └── visualization/
│       └── plots.py               ← hydrograph / alpha-map / summary plots
│
├── scripts/                ← command-line entry points (run in order)
│   ├── 01_extract_discharge.py    ← Stage → discharge conversion
│   ├── 02_download_imerg.py       ← IMERG download + GAG generation
│   ├── 03_run_bias_correction.py  ← full elasticity–optimise–validate loop
│   └── 04_generate_plots.py       ← publication-quality figures
│
├── notebooks/              ← exploratory Jupyter notebooks
│   ├── 01_flood_identification.ipynb
│   ├── 02_imerg_data_exploration.ipynb
│   ├── 03_discharge_analysis.ipynb
│   ├── 04_bias_correction_workflow.ipynb
│   └── 05_citizen_science_analysis.ipynb
│
├── configs/                ← YAML configuration (see above)
├── data/                   ← raw and processed data (not in version control)
│   ├── Discharge-DHM/      ← DHM stage CSV + rating curve
│   ├── IMERG/              ← downloaded IMERG CSVs
│   └── flood_events.csv    ← master 25-event list
│
├── Models/
│   └── Model-9_17/         ← GSSHA base model (Khokana watershed)
│
├── Stations Based Correction/  ← active pipeline (code moved to src/)
│   ├── Data/               ← per-event inputs
│   │   ├── IMERG/          ← .gag files per flood event
│   │   ├── DHM Discharge/  ← per-event discharge CSVs + rating curve
│   │   ├── Gssha_Base_Model/
│   │   └── flood_data.csv  ← event list with precipitation date ranges
│   └── Outputs/            ← α factors, elasticity matrix, validation
│
├── Bias_Correction_Results/    ← GSSHA run outputs (large, not committed)
│   ├── iteration_00_baseline/
│   ├── iteration_01/
│   └── iteration_02/
│
├── outputs/                ← final figures and reports
└── archive/                ← superseded code (kept for reference)
```

---

## Data Flow

```
DHM Stage CSV  ──[RatingCurve]──►  Discharge timeseries
                                        │
                                [FloodIdentifier]
                                        │
                               25 Flood Events list
                                        │
                  ┌─────────────────────┤─────────────────────┐
                  │                     │                     │
         [IMERGDownloader]              │            [extract_event_discharge]
                  │                     │                     │
          IMERG CSVs                    │            per-event discharge CSVs
                  │                     │                     │
          [gag_converter]               │                     │
                  │                     │                     │
          .gag files ──────────────────►│◄────────────────────┘
                                        │
                               [ElasticityEstimator]
                               (GSSHA: baseline + ±10% runs)
                                        │
                            Elasticity Matrix  E[event, station]
                                        │
                               [StationOptimizer]
                               (regularised least squares)
                                        │
                            Per-station α factors
                                        │
                      ┌─────────────────┼─────────────────┐
                      │                 │                 │
              Apply α to .gag      Save outputs      [ValidationRunner]
                      │                                    │
              Corrected .gag files              GSSHA re-runs + metrics
                                                           │
                                               Validation_Report.csv
                                               Validation_Plots.png
```

---

## Quick Start

### Prerequisites

```bash
# Python 3.10+
pip install -r requirements.txt

# GSSHA 8.1 binary in PATH
which gssha81

# Google Earth Engine authentication (for IMERG download)
earthengine authenticate
```

### Run the full pipeline

```bash
# From the project root
cd "19 WMS Kathmandu"

# 1. Convert DHM stage data to discharge
python scripts/01_extract_discharge.py

# 2. Download IMERG and generate .gag files (requires GEE access)
python scripts/02_download_imerg.py

# 3. Run bias correction (requires GSSHA)
python scripts/03_run_bias_correction.py --threads 4 --parallel 8

# 4. Generate plots
python scripts/04_generate_plots.py
```

### Dry-run (no GSSHA required)

```bash
python scripts/03_run_bias_correction.py --dry-run
```

### Configure for your system

Edit `configs/system.yaml` to set parallel GSSHA runs and GEE project:

```yaml
gee:
  project: ee-your-project-id
gssha:
  threads_per_run: 4
  parallel_runs:   8    # set to cpu_count() / threads_per_run
```

---

## Flood Events

25 historical flood events at the Bagmati Khokana gauge (2013–2024):

| # | Event ID | Peak Date | Season |
|---|---|---|---|
| 1 | event_001_20130523_flood | 2013-05-23 | Pre-Monsoon |
| 2 | event_002_20150817_flood | 2015-08-17 | Monsoon |
| … | … | … | … |
| 25 | event_025_20240928_flood | 2024-09-28 | Monsoon |

Full list with observed peak discharges: `Stations Based Correction/Data/flood_data.csv`

---

## Gauge Network

7 citizen-science tipping-bucket gauges across Kathmandu Valley:

| ID | Location | Lat | Lon |
|---|---|---|---|
| TP001 | Nagarjun | 27.738 | 85.289 |
| TP002 | Tokha | 27.770 | 85.329 |
| TP003 | Lapsephedi | 27.747 | 85.481 |
| TP005 | Kusunti | 27.666 | 85.314 |
| TP006 | Bhaktapur (KEC) | 27.671 | 85.439 |
| TP007 | Bhardev | 27.549 | 85.381 |
| TP008 | Okhreni | 27.797 | 85.422 |

Full metadata: `configs/stations.yaml`

---

## Method – Precipitation Elasticity

The correction factors α are solved from:

```
min  ||E · ln(α) - b||²  +  λ ||L ln(α)||²  +  τ ||ln(α) - m||²
```

where:
- **E** = elasticity matrix (n_events × n_stations)
- **b** = ln(Q_observed) − ln(Q_simulated)  per event
- **L** = spatial Laplacian (smoothness constraint)
- **m** = prior (default: ln(1) = 0, i.e. no correction)
- **λ, τ** = regularisation weights

Elasticity `E[e,s]` is estimated by:
```
ε = [ln(Q+) − ln(Q−)] / [ln(1.1) − ln(0.9)]
```
where Q+/Q− are peak discharges from ±10 % rainfall perturbation runs.

---

## Dependencies

| Package | Purpose |
|---|---|
| numpy, scipy | numerical computation |
| pandas | tabular data |
| matplotlib | plotting |
| pyyaml | configuration files |
| pyproj | coordinate transforms |
| earthengine-api | IMERG download (GEE) |
| jupyterlab | interactive notebooks |

See `requirements.txt` for pinned versions.

---

## Archive

`archive/` contains superseded code kept for historical reference:
- `archive/bias_correction_framework/` — old iterative (event-wide) approach
- `archive/old_scripts/` — early experimental scripts

These are **not** used in the current pipeline.

---

## References

- Kling et al. (2012) — Runoff conditions in the upper Danube basin under an
  ensemble of climate change scenarios. *Journal of Hydrology*.
- Teng et al. (2017) — How does bias correction of regional climate model
  precipitation affect modelled runoff? *Hydrology and Earth System Sciences*.
- NASA GPM IMERG V07 — <https://gpm.nasa.gov/data/imerg>
- GSSHA v8.1 — <https://www.gssha.com>

---

*For questions or issues, see `CONTRIBUTING.md` or open a discussion.*
