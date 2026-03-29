#!/bin/bash

# Quick installation script for packages
# Uses --user flag to avoid permission issues

echo "📦 Installing Premier League Predictor dependencies..."
echo ""

python3 -m pip install --user --upgrade pip setuptools wheel
echo "✅ pip upgraded"
echo ""

python3 -m pip install --user pandas numpy scikit-learn xgboost lightgbm
echo "✅ Core ML packages installed"
echo ""

python3 -m pip install --user requests beautifulsoup4 lxml cloudscraper
echo "✅ Data collection packages installed"
echo ""

python3 -m pip install --user mlflow optuna
echo "✅ MLOps packages installed"
echo ""

python3 -m pip install --user fastapi uvicorn pydantic
echo "✅ API packages installed"
echo ""

python3 -m pip install --user streamlit plotly
echo "✅ Frontend packages installed"
echo ""

python3 -m pip install --user python-dotenv joblib tqdm openpyxl pyyaml
echo "✅ Utility packages installed"
echo ""

echo "🔍 Verifying installation..."
python3 -c "import pandas, numpy, sklearn, xgboost, lightgbm, mlflow; print('✅ All core packages verified!')"

echo ""
echo "================================================"
echo "✅ INSTALLATION COMPLETE!"
echo "================================================"
echo ""
echo "Next: Run the pipeline with:"
echo "  python3 -m src.pipeline --all"
echo ""

