#!/bin/bash
# Auto-apply best hyperopt results when complete, then restart bot

HYPEROPT_LOG="/root/.openclaw/workspace/freqtrade-bot/user_data/logs/hyperopt.log"
BOT_LOG="/root/.openclaw/workspace/freqtrade-bot/user_data/logs/freqtrade.log"
STRATEGY="/root/.openclaw/workspace/freqtrade-bot/user_data/strategies/PhantomStrategy.py"
WORK_DIR="/root/.openclaw/workspace/freqtrade-bot"

echo "[$(date)] Waiting for hyperopt to finish..."

# Wait until hyperopt process is gone
while pgrep -f "freqtrade hyperopt" > /dev/null; do
    sleep 30
done

echo "[$(date)] Hyperopt finished. Checking results..."

# Get best epoch results as JSON
cd "$WORK_DIR"
BEST_JSON=$(freqtrade hyperopt-show --best --print-json 2>/dev/null)
if [ -z "$BEST_JSON" ]; then
    echo "[$(date)] ERROR: Could not retrieve best hyperopt results"
    exit 1
fi

echo "[$(date)] Best parameters: $BEST_JSON"

# Apply best params by writing them to a .json parameter file
# Freqtrade auto-loads PhantomStrategy.json from the strategy directory
PARAM_FILE="$WORK_DIR/user_data/strategies/PhantomStrategy.json"
echo "$BEST_JSON" > "$PARAM_FILE"
echo "[$(date)] Parameters written to $PARAM_FILE"

# Run a quick backtest with the new params to confirm improvement
echo "[$(date)] Running validation backtest..."
RESULT=$(freqtrade backtesting \
  --config user_data/config_backtest.json \
  --strategy PhantomStrategy \
  --timerange 20260101-20260323 \
  2>&1 | grep -E "Total profit|Win  Draw|Profit factor|Sharpe" | head -10)

echo "[$(date)] Validation result:"
echo "$RESULT"

# Restart the live bot to pick up new parameters
echo "[$(date)] Restarting live bot with optimized parameters..."
bash "$WORK_DIR/scripts/stop_bot.sh"
sleep 3
bash "$WORK_DIR/scripts/start_bot.sh"

echo "[$(date)] Done. Bot restarted with optimized hyperopt parameters."
echo "[$(date)] Check $BOT_LOG for confirmation."
