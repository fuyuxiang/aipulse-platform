#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
PID_DIR="$PROJECT_ROOT/.pids"
LOG_DIR="$PROJECT_ROOT/data/logs"

mkdir -p "$PID_DIR" "$LOG_DIR"

BACKEND_PID_FILE="$PID_DIR/backend.pid"
FRONTEND_PID_FILE="$PID_DIR/frontend.pid"
BACKEND_LOG="$LOG_DIR/backend.log"
FRONTEND_LOG="$LOG_DIR/frontend.log"

is_running() {
    local pid_file="$1"
    if [[ -f "$pid_file" ]]; then
        local pid
        pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
        rm -f "$pid_file"
    fi
    return 1
}

start_backend() {
    if is_running "$BACKEND_PID_FILE"; then
        echo "[backend] Already running (PID $(cat "$BACKEND_PID_FILE"))"
        return 0
    fi

    echo "[backend] Starting on port 8000..."
    export PYTHONPATH="$PROJECT_ROOT/backend:$PROJECT_ROOT:$PROJECT_ROOT/echo-agent:${PYTHONPATH:-}"

    cd "$PROJECT_ROOT/backend"
    nohup python -m uvicorn app.main:app \
        --host 127.0.0.1 \
        --port 8000 \
        --reload \
        > "$BACKEND_LOG" 2>&1 &

    local pid=$!
    echo "$pid" > "$BACKEND_PID_FILE"
    cd "$PROJECT_ROOT"

    sleep 1
    if kill -0 "$pid" 2>/dev/null; then
        echo "[backend] Started (PID $pid), log: $BACKEND_LOG"
    else
        echo "[backend] Failed to start, check log: $BACKEND_LOG"
        rm -f "$BACKEND_PID_FILE"
        return 1
    fi
}

start_frontend() {
    if is_running "$FRONTEND_PID_FILE"; then
        echo "[frontend] Already running (PID $(cat "$FRONTEND_PID_FILE"))"
        return 0
    fi

    echo "[frontend] Starting on port 3000..."
    cd "$PROJECT_ROOT/frontend"
    nohup npm run dev > "$FRONTEND_LOG" 2>&1 &

    local pid=$!
    echo "$pid" > "$FRONTEND_PID_FILE"
    cd "$PROJECT_ROOT"

    sleep 2
    if kill -0 "$pid" 2>/dev/null; then
        echo "[frontend] Started (PID $pid), log: $FRONTEND_LOG"
    else
        echo "[frontend] Failed to start, check log: $FRONTEND_LOG"
        rm -f "$FRONTEND_PID_FILE"
        return 1
    fi
}

usage() {
    echo "Usage: $0 [backend|frontend|all]"
    echo "  backend   - Start backend service only"
    echo "  frontend  - Start frontend service only"
    echo "  all       - Start all services (default)"
}

case "${1:-all}" in
    backend)
        start_backend
        ;;
    frontend)
        start_frontend
        ;;
    all)
        start_backend
        start_frontend
        echo ""
        echo "All services started."
        echo "  Backend:  http://127.0.0.1:8000"
        echo "  Frontend: http://127.0.0.1:3000"
        echo "  Health:   http://127.0.0.1:8000/health"
        ;;
    -h|--help)
        usage
        ;;
    *)
        echo "Unknown option: $1"
        usage
        exit 1
        ;;
esac
