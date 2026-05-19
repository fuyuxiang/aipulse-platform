#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

TARGET="${1:-all}"

echo "=== Stopping services ==="
"$PROJECT_ROOT/stop.sh" "$TARGET"

echo ""
echo "=== Starting services ==="
"$PROJECT_ROOT/start.sh" "$TARGET"
