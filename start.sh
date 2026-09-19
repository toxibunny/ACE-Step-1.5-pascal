#!/usr/bin/env bash
# Start ACE-Step server (Gradio UI + REST API) on port 7860
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="/tmp/acestep.pid"
LOG_FILE="/tmp/acestep_gradio.log"
PORT="${ACESTEP_PORT:-7860}"
HOST="${ACESTEP_HOST:-0.0.0.0}"
PYTHON="${ACESTEP_PYTHON:-/home/toxi/miniconda3/envs/comfyui/bin/python}"

# Check if already running
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "Already running (PID $(cat "$PID_FILE")). Use stop.sh first."
    exit 1
fi

echo "Starting ACE-Step on ${HOST}:${PORT}..."
cd "$SCRIPT_DIR"

nohup "$PYTHON" -m acestep.acestep_v15_pipeline \
    --port "$PORT" \
    --server-name "$HOST" \
    --config_path acestep-v15-turbo \
    --device cuda \
    --backend pt \
    --lm_model_path acestep-5Hz-lm-1.7B \
    --init_llm true \
    --enable-api \
    --quantization int8_weight_only \
    > "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"
echo "PID: $(cat "$PID_FILE")"
echo "Log: $LOG_FILE"

# Wait for server to be ready (model loading takes ~60s)
echo "Waiting for server to be ready..."
for i in $(seq 1 60); do
    if curl -s -m 2 "http://localhost:${PORT}/health" 2>/dev/null | grep -q "ok"; then
        echo "✅ ACE-Step is ready!"
        echo "   Web UI:  http://${HOST}:${PORT}/"
        echo "   API:     POST http://${HOST}:${PORT}/release_task"
        echo "   Health:  GET  http://${HOST}:${PORT}/health"
        exit 0
    fi
    sleep 2
done

echo "⚠️  Server started but not responding to /health after 120s."
echo "    Check log: tail -f $LOG_FILE"
exit 1
