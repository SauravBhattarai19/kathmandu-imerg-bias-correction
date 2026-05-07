# Contributing Guide

---

## Repository Map (ownership boundaries)

| Directory / File | Owner | Description |
|---|---|---|
| `src/gssha/` | Hydrology lead | GSSHA model interface |
| `src/bias_correction/` | Hydrology lead | Elasticity, optimisation, validation |
| `src/preprocessing/` | Data engineer | Stage→Q, flood ID |
| `src/data_acquisition/` | Data engineer | IMERG download, GAG conversion |
| `src/visualization/` | Any | Plots |
| `scripts/` | Any | Entry-point scripts |
| `configs/` | PI / lead | Paths, system params |
| `notebooks/` | Any | Exploration |
| `docs/` | Documentation lead | Architecture, guides |

---

## Getting Started

```bash
# 1. Clone / copy the project
git clone <repo-url>
cd "19 WMS Kathmandu"

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Authenticate GEE (for IMERG downloads only)
earthengine authenticate

# 5. Verify GSSHA is in PATH
which gssha81
```

---

## Code Conventions

### Python style
- Follow **PEP 8**; max line length 100.
- Use `from __future__ import annotations` in all source files.
- Prefer `pathlib.Path` over `os.path` strings.
- All public functions must have a **NumPy-style docstring**.

### No hardcoded paths
All paths live in `configs/paths.yaml`.  Load them via:

```python
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent   # adjust as needed
with open(ROOT / "configs" / "paths.yaml") as fh:
    paths = yaml.safe_load(fh)
my_dir = (ROOT / paths["some"]["key"]).resolve()
```

### Configuration changes
- Edit `configs/*.yaml` — never put constants in Python source files.
- System-specific settings (cores, GEE project) go in `configs/system.yaml`.

---

## Making a Contribution

1. Create a branch: `git checkout -b feature/your-feature`
2. Make changes in `src/` or `scripts/` (not in `Notebooks/` unless purely exploratory).
3. Test locally with `--dry-run` where applicable.
4. Update `TODO.md` to check off any completed items.
5. Open a pull request with a clear description of the change.

---

## Testing

```bash
# Basic import check
python -c "from src.gssha import GagFile, GsshaRunner; print('OK')"
python -c "from src.bias_correction import ElasticityEstimator, StationOptimizer; print('OK')"

# Dry-run integration test
python scripts/03_run_bias_correction.py --dry-run
```

---

## File naming conventions

| Type | Convention | Example |
|---|---|---|
| Python modules | `snake_case.py` | `gag_editor.py` |
| Config files | `snake_case.yaml` | `paths.yaml` |
| Notebooks | `NN_description.ipynb` | `02_imerg_data_exploration.ipynb` |
| Scripts | `NN_action.py` | `03_run_bias_correction.py` |
| Output CSVs | `descriptive_name.csv` | `Elasticity_Matrix.csv` |

---

## Questions

Open a GitHub issue or contact the PI directly.
