#!/bin/bash
BOT_DIR="/root/.openclaw/workspace/freqtrade-bot"
PID_FILE="/tmp/freqtrade_phantom.pid"
LOG_FILE="$BOT_DIR/user_data/logs/freqtrade.log"
if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
    echo "PhantomBot already running (PID $(cat $PID_FILE))"
    exit 1
fi
mkdir -p "$BOT_DIR/user_data/logs"
nohup freqtrade trade --strategy PhantomStrategy --userdir "$BOT_DIR/user_data" --config "$BOT_DIR/user_data/config.json" --logfile "$LOG_FILE" >> "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "PhantomBot started (PID $(cat $PID_FILE))"
