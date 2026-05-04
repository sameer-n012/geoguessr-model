#!/bin/bash

conda deactivate

set -a
source .venv/bin/activate
source .env
set +a


LOG_FILE="model/train/train.log"

echo "Starting training..."
nohup python model/train.py > "$LOG_FILE" 2>&1 &

PID=$!
echo "Training started in background with PID $PID"
echo "Logs: $LOG_FILE"