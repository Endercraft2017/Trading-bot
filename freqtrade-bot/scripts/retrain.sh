#!/bin/bash
set -euo pipefail

BOT_DIR="/root/.openclaw/workspace/freqtrade-bot"

# Load environment
if [ -f "$BOT_DIR/.env" ]; then
    set -a
    source "$BOT_DIR/.env"
    set +a
fi

echo "[$(date)] Starting PhantomBot ML retraining..."
python3 "$BOT_DIR/ml/trainer.py"
echo "[$(date)] Retraining complete."
