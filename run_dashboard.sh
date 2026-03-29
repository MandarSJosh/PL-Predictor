#!/bin/bash

# Start the Streamlit dashboard

set -e

if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found. Run run_pipeline.sh first."
    exit 1
fi

source venv/bin/activate

echo "📊 Starting Premier League Prediction Dashboard..."
echo "Dashboard will be available at http://localhost:8501"
echo ""

streamlit run src/dashboard/app.py

