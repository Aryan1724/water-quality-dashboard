# Local (Non‑Colab) Setup

This makes your results reproducible and avoids Colab runtime differences.

## 1) Create a virtual environment
```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

## 2) Install exact versions
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 3) Run the notebook
```bash
jupyter notebook
# or
jupyter lab
```
Open `water_quality_project_starter.ipynb` and run all cells.

## 4) (Optional) Quick validation
Run `python run_checks.py` to verify key numbers match.
