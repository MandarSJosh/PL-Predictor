#!/bin/bash

# Fixed pipeline runner with correct PYTHONPATH

cd "$(dirname "$0")"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

python3 -m src.pipeline "$@"

