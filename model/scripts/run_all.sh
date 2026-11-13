#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
"$PYTHON_BIN" -m src.eda --data-dir data/raw --out-dir reports
"$PYTHON_BIN" -m unittest discover -s tests -v
"$PYTHON_BIN" -m src.tune_lightgbm --data-dir data/raw --config configs/three_tree.json --max-rounds 1000 --patience 100 --trials 16 --output outputs/lightgbm_tuned
