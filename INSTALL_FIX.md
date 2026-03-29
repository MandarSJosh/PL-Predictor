# Installation Fix Guide

## Problem
CatBoost 1.2.2 is failing to install on Python 3.12 due to build dependencies.

## Solution 1: Install Without CatBoost (Recommended)

Install all packages except CatBoost first:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install pandas numpy scikit-learn xgboost lightgbm
python3 -m pip install requests beautifulsoup4 lxml cloudscraper
python3 -m pip install mlflow optuna
python3 -m pip install fastapi uvicorn pydantic
python3 -m pip install streamlit plotly
python3 -m pip install python-dotenv joblib tqdm openpyxl pyyaml
```

Then try CatBoost separately (optional):
```bash
python3 -m pip install catboost
```

## Solution 2: Use Minimal Requirements

```bash
python3 -m pip install -r requirements-minimal.txt
```

## Solution 3: Update pip First

```bash
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r requirements.txt
```

## Solution 4: Install CatBoost from Conda (if using conda)

```bash
conda install -c conda-forge catboost
```

## Note

The system will work fine without CatBoost - it will just use XGBoost and LightGBM. CatBoost is optional and can be added later.

## Verify Installation

```bash
python3 -c "import pandas, xgboost, lightgbm, mlflow; print('Core packages OK!')"
```

If CatBoost is installed:
```bash
python3 -c "import catboost; print('CatBoost OK!')"
```

