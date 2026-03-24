#!/bin/bash
set -euo pipefail

BOT_DIR="/root/.openclaw/workspace/freqtrade-bot"

# Load environment
if [ -f "$BOT_DIR/.env" ]; then
    set -a
    source "$BOT_DIR/.env"
    set +a
fi

echo "[$(date)] Running LLM Supervisor..."
python3 "$BOT_DIR/ml/llm_supervisor.py"
echo "[$(date)] Done."
