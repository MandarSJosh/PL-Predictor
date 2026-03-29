# Manual Installation Instructions

## ⚠️ Important: Directory Name Issue

Your project directory contains a colon (`:`) in the name, which prevents creating a virtual environment. 

## Solution: Install Packages Directly

Run these commands **one by one** in your terminal:

### Step 1: Install Core Packages

```bash
python3 -m pip install --user --upgrade pip setuptools wheel
python3 -m pip install --user pandas numpy scikit-learn xgboost lightgbm
python3 -m pip install --user requests beautifulsoup4 lxml cloudscraper
python3 -m pip install --user mlflow optuna
python3 -m pip install --user fastapi uvicorn pydantic
python3 -m pip install --user streamlit plotly
python3 -m pip install --user python-dotenv joblib tqdm openpyxl pyyaml
```

### Step 2: Verify Installation

```bash
python3 -c "import pandas, numpy, sklearn, xgboost, lightgbm, mlflow; print('✅ All packages installed!')"
```

### Step 3: Run the Pipeline

Once packages are installed, run:

```bash
python3 -m src.pipeline --all
```

## Alternative: Use a Different Directory

If you want to use a virtual environment, rename or copy the project to a directory without colons:

```bash
# Copy to a new directory
cp -r "PremierLeagueChampions:Games predictor2025:26" PremierLeaguePredictor
cd PremierLeaguePredictor

# Then create venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Quick Install Script

Save this as `quick_install.sh` and run it:

```bash
#!/bin/bash
python3 -m pip install --user --upgrade pip setuptools wheel
python3 -m pip install --user pandas numpy scikit-learn xgboost lightgbm requests beautifulsoup4 lxml cloudscraper mlflow optuna fastapi uvicorn pydantic streamlit plotly python-dotenv joblib tqdm openpyxl pyyaml
python3 -c "import pandas, numpy, sklearn, xgboost, lightgbm, mlflow; print('✅ Installation complete!')"
```

Then:
```bash
chmod +x quick_install.sh
./quick_install.sh
```

