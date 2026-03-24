#!/bin/bash
set -uo pipefail

PID_FILE="/tmp/freqtrade_phantom.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "Stopping PhantomBot (PID $PID)..."
        kill -SIGTERM "$PID" 2>/dev/null
        # Wait up to 10 seconds for graceful shutdown
        for i in $(seq 1 10); do
            kill -0 "$PID" 2>/dev/null || break
            sleep 1
        done
        # Force kill if still running
        if kill -0 "$PID" 2>/dev/null; then
            echo "Force killing PID $PID"
            kill -9 "$PID" 2>/dev/null
        fi
    fi
    rm -f "$PID_FILE"
fi

# Kill any remaining freqtrade trade processes
REMAINING=$(pgrep -f "freqtrade trade" 2>/dev/null || true)
if [ -n "$REMAINING" ]; then
    echo "Killing stray freqtrade PIDs: $REMAINING"
    kill -SIGTERM $REMAINING 2>/dev/null || true
    sleep 2
    kill -9 $REMAINING 2>/dev/null || true
fi

echo "PhantomBot stopped"
