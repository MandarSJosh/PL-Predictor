#!/bin/bash

# Installation script that handles CatBoost issues

set -e

echo "🔧 Installing Premier League Predictor Dependencies"
echo "=================================================="

# Upgrade pip first
echo "📦 Upgrading pip..."
python3 -m pip install --upgrade pip setuptools wheel

# Install core packages first (without CatBoost)
echo ""
echo "📦 Installing core ML packages..."
python3 -m pip install pandas numpy scikit-learn xgboost lightgbm

# Install data collection packages
echo ""
echo "📦 Installing data collection packages..."
python3 -m pip install requests beautifulsoup4 lxml cloudscraper

# Install MLOps packages
echo ""
echo "📦 Installing MLOps packages..."
python3 -m pip install mlflow optuna

# Install API packages
echo ""
echo "📦 Installing API packages..."
python3 -m pip install fastapi uvicorn pydantic

# Install frontend packages
echo ""
echo "📦 Installing frontend packages..."
python3 -m pip install streamlit plotly

# Install utility packages
echo ""
echo "📦 Installing utility packages..."
python3 -m pip install python-dotenv joblib tqdm openpyxl pyyaml

# Try to install CatBoost (optional)
echo ""
echo "📦 Attempting to install CatBoost (optional)..."
if python3 -m pip install catboost 2>/dev/null; then
    echo "✅ CatBoost installed successfully!"
else
    echo "⚠️  CatBoost installation failed - this is OK, the system will work without it"
    echo "   You can try installing it later with: pip install catboost"
fi

echo ""
echo "✅ Installation complete!"
echo ""
echo "Verify installation:"
echo "  python3 -c \"import pandas, xgboost, lightgbm, mlflow; print('Core packages OK!')\""

