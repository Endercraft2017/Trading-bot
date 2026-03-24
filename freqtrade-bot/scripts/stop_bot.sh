#!/bin/bash
PID_FILE="/tmp/freqtrade_phantom.pid"

# Kill PID file process if present
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        kill -SIGTERM "$PID" 2>/dev/null
        sleep 2
        kill -0 "$PID" 2>/dev/null && kill -9 "$PID" 2>/dev/null
    fi
    rm -f "$PID_FILE"
fi

# Kill any remaining freqtrade processes (catches manually-started instances)
REMAINING=$(pgrep -f "freqtrade trade" 2>/dev/null)
if [ -n "$REMAINING" ]; then
    echo "Killing stray freqtrade PIDs: $REMAINING"
    kill -9 $REMAINING 2>/dev/null
fi

echo "PhantomBot stopped"
