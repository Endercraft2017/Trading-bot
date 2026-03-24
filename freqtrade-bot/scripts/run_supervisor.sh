#!/bin/bash
echo "[$(date)] Running LLM Supervisor..."
python3 /root/.openclaw/workspace/freqtrade-bot/ml/llm_supervisor.py
echo "[$(date)] Done."
