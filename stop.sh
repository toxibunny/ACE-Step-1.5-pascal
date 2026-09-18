#!/usr/bin/env bash
# Stop ACE-Step server
set -euo pipefail

PID_FILE="/tmp/acestep.pid"
PORT="${ACESTEP_PORT:-7860}"

if [[ ! -f "$PID_FILE" ]]; then
    # Try to find by port
    PID=$(ss -tlnp 2>/dev/null | grep ":${PORT}" | grep -oP 'pid=\K[0-9]+' | head -1)
    if [[ -n "$PID" ]]; then
        echo "Found ACE-Step on port ${PORT} (PID $PID)"
        kill "$PID"
        echo "✅ Stopped"
        exit 0
    fi
    echo "Not running (no PID file, nothing on port ${PORT})."
    exit 0
fi

PID=$(cat "$PID_FILE")
if kill -0 "$PID" 2>/dev/null; then
    kill "$PID"
    echo "Stopped ACE-Step (PID $PID)"
else
    echo "Process $PID not running (already stopped?)"
fi
rm -f "$PID_FILE"
echo "✅ Done"
