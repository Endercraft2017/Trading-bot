#!/bin/bash
# Smart retrain: only runs if 10+ new real-feature trades since last training

DB="/root/.openclaw/workspace/freqtrade-bot/user_data/tradesv3.sqlite"
META="/root/.openclaw/workspace/freqtrade-bot/ml/models/signal_filter_meta.json"
LOG="/root/.openclaw/workspace/freqtrade-bot/user_data/logs/retrain.log"

# Count real-feature closed trades in DB
REAL_TRADES=$(sqlite3 "$DB" "SELECT COUNT(*) FROM trades WHERE is_open=0 AND enter_tag LIKE '{%';" 2>/dev/null || echo "0")

# Get trade count from last training run
LAST_TRAINED_COUNT=0
if [ -f "$META" ]; then
    LAST_TRAINED_COUNT=$(python3 -c "import json; m=json.load(open('$META')); print(m.get('real_feature_trades',0))" 2>/dev/null || echo "0")
fi

NEW_SINCE_LAST=$((REAL_TRADES - LAST_TRAINED_COUNT))

echo "[$(date)] Real-feature trades: $REAL_TRADES total, $LAST_TRAINED_COUNT at last train, $NEW_SINCE_LAST new" >> "$LOG"

# Only retrain if 10+ new real-feature trades since last run
if [ "$NEW_SINCE_LAST" -ge 10 ]; then
    echo "[$(date)] Threshold met — retraining..." >> "$LOG"
    python3 /root/.openclaw/workspace/freqtrade-bot/ml/trainer.py >> "$LOG" 2>&1
    echo "[$(date)] Retrain complete." >> "$LOG"
else
    echo "[$(date)] Only $NEW_SINCE_LAST new real trades — skipping retrain (need 10)" >> "$LOG"
fi
