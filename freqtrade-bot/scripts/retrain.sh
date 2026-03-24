#!/bin/bash
source "$(dirname "$0")/common.sh"

echo "[$(date)] Starting PhantomBot ML retraining..."
python3 "$BOT_DIR/ml/trainer.py"
echo "[$(date)] Retraining complete."
