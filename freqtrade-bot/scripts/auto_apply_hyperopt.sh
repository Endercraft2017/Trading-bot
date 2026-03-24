#!/bin/bash
set -euo pipefail

BOT_DIR="/root/.openclaw/workspace/freqtrade-bot"
LOG="$BOT_DIR/user_data/logs/hyperopt_apply.log"

# Load environment
if [ -f "$BOT_DIR/.env" ]; then
    set -a
    source "$BOT_DIR/.env"
    set +a
fi

mkdir -p "$(dirname "$LOG")"

echo "[$(date)] Waiting for hyperopt to finish..." >> "$LOG"

# Wait until hyperopt process completes
while pgrep -f "freqtrade hyperopt" > /dev/null 2>&1; do
    sleep 30
done

echo "[$(date)] Hyperopt finished. Checking results..." >> "$LOG"

cd "$BOT_DIR"

# Get best epoch results as JSON
BEST_JSON=$(freqtrade hyperopt-show --best --print-json --userdir user_data 2>/dev/null || true)
if [ -z "$BEST_JSON" ]; then
    echo "[$(date)] ERROR: Could not retrieve best hyperopt results" >> "$LOG"
    exit 1
fi

echo "[$(date)] Best parameters: $BEST_JSON" >> "$LOG"

# Write params to JSON file (Freqtrade auto-loads)
PARAM_FILE="$BOT_DIR/user_data/strategies/PhantomStrategy.json"
echo "$BEST_JSON" > "$PARAM_FILE"
echo "[$(date)] Parameters written to $PARAM_FILE" >> "$LOG"

# Validation backtest
echo "[$(date)] Running validation backtest..." >> "$LOG"
RESULT=$(freqtrade backtesting     --config user_data/config_backtest.json     --strategy PhantomStrategy     --userdir user_data     --timerange 20260101-20260323     2>&1 | grep -E "Total profit|Win  Draw|Profit factor|Sharpe" | head -10 || true)

echo "[$(date)] Validation result:" >> "$LOG"
echo "$RESULT" >> "$LOG"

# Restart bot with optimized parameters
echo "[$(date)] Restarting bot with optimized parameters..." >> "$LOG"
bash "$BOT_DIR/scripts/stop_bot.sh"
sleep 3
bash "$BOT_DIR/scripts/start_bot.sh"

echo "[$(date)] Done. Bot restarted with hyperopt params." >> "$LOG"
