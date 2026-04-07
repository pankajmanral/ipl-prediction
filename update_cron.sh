#!/bin/zsh
PROJECT_DIR="/Users/abhisheklad/Projects/IPL-Winner-Prediction-2026"
cd "$PROJECT_DIR"
source venv/bin/activate
python3 sync_results.py >> sync.log 2>&1
