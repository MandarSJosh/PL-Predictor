#!/bin/bash

# Start the FastAPI server

set -e

if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found. Run run_pipeline.sh first."
    exit 1
fi

source venv/bin/activate

echo "🚀 Starting Premier League Prediction API..."
echo "API will be available at http://localhost:8000"
echo "API docs at http://localhost:8000/docs"
echo ""

python -m src.api.main

