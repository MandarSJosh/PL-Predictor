#!/bin/bash

# Step-by-Step Installation and Execution Script
# Run this script to set up and run the Premier League Predictor

set -e

echo "⚽ Premier League Predictor - Step by Step Setup"
echo "================================================"
echo ""

# Step 1: Upgrade pip
echo "📦 STEP 1: Upgrading pip..."
python3 -m pip install --upgrade pip setuptools wheel --quiet
echo "✅ pip upgraded"
echo ""

# Step 2: Install core ML packages
echo "📦 STEP 2: Installing core ML packages..."
python3 -m pip install pandas numpy scikit-learn xgboost lightgbm --quiet
echo "✅ Core ML packages installed"
echo ""

# Step 3: Install data collection packages
echo "📦 STEP 3: Installing data collection packages..."
python3 -m pip install requests beautifulsoup4 lxml cloudscraper --quiet
echo "✅ Data collection packages installed"
echo ""

# Step 4: Install MLOps packages
echo "📦 STEP 4: Installing MLOps packages..."
python3 -m pip install mlflow optuna --quiet
echo "✅ MLOps packages installed"
echo ""

# Step 5: Install API packages
echo "📦 STEP 5: Installing API packages..."
python3 -m pip install fastapi uvicorn pydantic --quiet
echo "✅ API packages installed"
echo ""

# Step 6: Install frontend packages
echo "📦 STEP 6: Installing frontend packages..."
python3 -m pip install streamlit plotly --quiet
echo "✅ Frontend packages installed"
echo ""

# Step 7: Install utility packages
echo "📦 STEP 7: Installing utility packages..."
python3 -m pip install python-dotenv joblib tqdm openpyxl pyyaml --quiet
echo "✅ Utility packages installed"
echo ""

# Step 8: Try CatBoost (optional)
echo "📦 STEP 8: Attempting to install CatBoost (optional)..."
if python3 -m pip install catboost --quiet 2>/dev/null; then
    echo "✅ CatBoost installed"
else
    echo "⚠️  CatBoost skipped (optional - system will work without it)"
fi
echo ""

# Verify installation
echo "🔍 Verifying installation..."
python3 -c "import pandas, numpy, sklearn, xgboost, lightgbm, mlflow; print('✅ All core packages verified!')"
echo ""

echo "================================================"
echo "✅ INSTALLATION COMPLETE!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Collect data: python3 -m src.pipeline --collect"
echo "2. Create features: python3 -m src.pipeline --features"
echo "3. Train model: python3 -m src.pipeline --train"
echo "4. Or run all: python3 -m src.pipeline --all"
echo ""

