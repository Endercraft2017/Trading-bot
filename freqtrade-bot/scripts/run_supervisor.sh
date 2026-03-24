#!/bin/bash
source "$(dirname "$0")/common.sh"

echo "[$(date)] Running LLM Supervisor..."
python3 "$BOT_DIR/ml/llm_supervisor.py"
echo "[$(date)] Done."
