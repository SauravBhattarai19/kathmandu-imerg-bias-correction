# Notebooks

Interactive Jupyter notebooks for exploration, analysis, and reporting.

| Notebook | Purpose |
|---|---|
| `01_flood_identification.ipynb` | Identify top-25 flood events from Khokana stage data |
| `02_imerg_data_exploration.ipynb` | Explore downloaded IMERG precipitation at station locations |
| `03_discharge_analysis.ipynb` | Compare simulated vs observed discharge time-series |
| `04_bias_correction_workflow.ipynb` | Interactive run of the full bias correction pipeline |
| `05_citizen_science_analysis.ipynb` | Explore citizen science tipping-bucket gauge data |

## Running notebooks

```bash
cd "19 WMS Kathmandu"
jupyter lab notebooks/
```

All notebooks import from `src/` — ensure the project root is on `sys.path`
(this is handled automatically by the first cell of each notebook).
