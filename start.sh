#!/bin/bash
# Fusion-RAG — Start/Stop script
# Usage: ./start.sh [start|stop|restart|status]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Config
PORT=${FUSION_RAG_PORT:-11436}
HOST=${FUSION_RAG_HOST:-"127.0.0.1"}
MLX_URL=${FUSION_MLX_URL:-"http://127.0.0.1:11432/v1"}
EMBED_MODEL=${FUSION_RAG_EMBED:-"BGE-M3"}
PID_FILE="$SCRIPT_DIR/.fusion-rag.pid"
LOG_DIR="$SCRIPT_DIR/logs"
STDOUT_LOG="$LOG_DIR/stdout.log"

ensure_log_dir() {
    mkdir -p "$LOG_DIR"
}

get_pid() {
    if [ -f "$PID_FILE" ]; then
        cat "$PID_FILE"
    else
        echo ""
    fi
}

is_running() {
    local pid=$(get_pid)
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        return 0
    fi
    return 1
}

start() {
    if is_running; then
        echo "Fusion-RAG is already running (PID $(get_pid))"
        exit 1
    fi

    ensure_log_dir

    echo "Starting Fusion-RAG on $HOST:$PORT ..."
    echo "  MLX URL: $MLX_URL"
    echo "  Embedding model: $EMBED_MODEL"
    echo ""

    # O-P1-2: nohup output is the BOOTSTRAP sink — it captures lines emitted
    # before configure_logging runs (import errors, startup-probe stderr) and
    # uvicorn's own lifecycle messages. The app's own logs go to
    # logs/fusion-rag.log with a RotatingFileHandler (10MB x 5). This bootstrap
    # file is NOT rotated by the app; keep it small by rotating externally
    # (logrotate) or rely on the app's primary sink. Stderr merged into stdout
    # so a single tail shows the full bootstrap trail.
    FUSION_RAG_PORT="$PORT" FUSION_RAG_HOST="$HOST" \
    FUSION_MLX_URL="$MLX_URL" FUSION_RAG_EMBED="$EMBED_MODEL" \
    nohup python3 -m fusion_rag.api.server \
        >> "$STDOUT_LOG" 2>&1 &

    PID=$!
    echo $PID > "$PID_FILE"
    echo "Started (PID $PID)"

    # Wait for startup
    sleep 1
    if is_running; then
        echo "Fusion-RAG is running on http://$HOST:$PORT"
    else
        echo "Failed to start. Check logs: $STDOUT_LOG"
        cat "$STDOUT_LOG" | tail -10
        exit 1
    fi
}

stop() {
    local pid=$(get_pid)
    if [ -z "$pid" ]; then
        echo "Fusion-RAG is not running"
        return
    fi

    echo "Stopping Fusion-RAG (PID $pid) ..."
    # O-P1-5: SIGTERM lets uvicorn drain in-flight requests for up to 30s
    # (timeout_graceful_shutdown=30 in server.run_server). Poll up to 35s so a
    # graceful stop finishes pending work (an in-flight rerank/RAG call) BEFORE
    # the force kill. Prior 5s poll cut off the drain and kill -9'd mid-response.
    kill -TERM "$pid" 2>/dev/null || true
    rm -f "$PID_FILE"

    # Wait for process to exit (graceful drain window)
    for i in {1..35}; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "Stopped (graceful)"
            return
        fi
        sleep 1
    done

    # Force kill if still running
    if kill -0 "$pid" 2>/dev/null; then
        echo "Force stopping ..."
        kill -9 "$pid" 2>/dev/null || true
        echo "Stopped (forced)"
    fi
}

restart() {
    stop
    sleep 1
    start
}

status() {
    if is_running; then
        local pid=$(get_pid)
        echo "Fusion-RAG is running (PID $pid)"
        echo "  URL: http://$HOST:$PORT"

        # Check if responding
        if command -v curl &>/dev/null; then
            if curl -s "http://$HOST:$PORT/health" >/dev/null 2>&1; then
                echo "  Health: OK"
            else
                echo "  Health: Not responding"
            fi
        fi
    else
        echo "Fusion-RAG is not running"
    fi
}

case "${1:-status}" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        restart
        ;;
    status)
        status
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        echo ""
        echo "Environment variables:"
        echo "  FUSION_RAG_PORT     Port (default: 11436)"
        echo "  FUSION_RAG_HOST     Host (default: 127.0.0.1)"
        echo "  FUSION_MLX_URL     fusion-mlx URL (default: http://127.0.0.1:11432/v1)"
        echo "  FUSION_RAG_EMBED    Embedding model (default: BGE-M3)"
        exit 1
        ;;
esac