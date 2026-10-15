#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Auto-detect python environment with fastapi installed
if [ -x "/home/rayhan/miniconda3/envs/quickmachine/bin/python" ]; then
    PYTHON_BIN="/home/rayhan/miniconda3/envs/quickmachine/bin/python"
elif command -v python3 >/dev/null 2>&1 && python3 -c "import fastapi" >/dev/null 2>&1; then
    PYTHON_BIN="python3"
elif [ -x "$DIR/venv/bin/python" ]; then
    PYTHON_BIN="$DIR/venv/bin/python"
elif [ -x "$DIR/.venv/bin/python" ]; then
    PYTHON_BIN="$DIR/.venv/bin/python"
else
    PYTHON_BIN="$(which python3 || echo 'python3')"
fi

WORKERS="${WORKERS:-1}"
if [ "$WORKERS" -gt 1 ]; then
    echo "Starting QuickMachine FastAPI backend on http://0.0.0.0:8000 ($WORKERS workers) using $PYTHON_BIN..."
    exec "$PYTHON_BIN" -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --workers "$WORKERS"
else
    echo "Starting QuickMachine FastAPI backend on http://0.0.0.0:8000 using $PYTHON_BIN..."
    exec "$PYTHON_BIN" -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
fi
