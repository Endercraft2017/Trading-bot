#!/bin/bash
echo "[$(date)] Starting PhantomBot ML retraining..."
python3 /root/.openclaw/workspace/freqtrade-bot/ml/trainer.py
echo "[$(date)] Retraining complete."
