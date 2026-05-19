#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
PID_DIR="$PROJECT_ROOT/.pids"

BACKEND_PID_FILE="$PID_DIR/backend.pid"
FRONTEND_PID_FILE="$PID_DIR/frontend.pid"

stop_process() {
    local name="$1"
    local pid_file="$2"

    if [[ ! -f "$pid_file" ]]; then
        echo "[$name] Not running (no PID file)"
        return 0
    fi

    local pid
    pid=$(cat "$pid_file")

    if ! kill -0 "$pid" 2>/dev/null; then
        echo "[$name] Process $pid already exited, cleaning up PID file"
        rm -f "$pid_file"
        return 0
    fi

    echo "[$name] Stopping (PID $pid)..."
    kill "$pid" 2>/dev/null || true

    local wait_count=0
    while kill -0 "$pid" 2>/dev/null && [[ $wait_count -lt 10 ]]; do
        sleep 1
        wait_count=$((wait_count + 1))
    done

    if kill -0 "$pid" 2>/dev/null; then
        echo "[$name] Graceful stop timed out, force killing..."
        kill -9 "$pid" 2>/dev/null || true
        sleep 1
    fi

    rm -f "$pid_file"
    echo "[$name] Stopped"
}

stop_backend() {
    stop_process "backend" "$BACKEND_PID_FILE"
    # Clean up any orphaned uvicorn processes for this project
    pkill -f "uvicorn app.main:app.*--port 8000" 2>/dev/null || true
}

stop_frontend() {
    stop_process "frontend" "$FRONTEND_PID_FILE"
    # Clean up any orphaned webpack-dev-server processes for this project
    pkill -f "webpack.*serve.*--port 3000" 2>/dev/null || true
}

usage() {
    echo "Usage: $0 [backend|frontend|all]"
    echo "  backend   - Stop backend service only"
    echo "  frontend  - Stop frontend service only"
    echo "  all       - Stop all services (default)"
}

case "${1:-all}" in
    backend)
        stop_backend
        ;;
    frontend)
        stop_frontend
        ;;
    all)
        stop_backend
        stop_frontend
        echo ""
        echo "All services stopped."
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
