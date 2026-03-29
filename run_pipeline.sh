#!/bin/bash

# Premier League Prediction Pipeline Runner
# This script runs the complete pipeline

set -e

echo "⚽ Premier League Prediction Pipeline"
echo "======================================"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Install dependencies if needed
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "Installing dependencies..."
    python3 -m pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Run pipeline
echo ""
echo "Running full pipeline..."
python -m src.pipeline --all

echo ""
echo "✅ Pipeline completed!"
echo ""
echo "Next steps:"
echo "1. Start API: python -m src.api.main"
echo "2. Start Dashboard: streamlit run src/dashboard/app.py"

