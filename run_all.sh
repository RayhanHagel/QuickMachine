#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "=========================================================="
echo " Starting QuickMachine (FastAPI Backend + Next.js Frontend)"
echo "=========================================================="

cleanup() {
    echo "Stopping QuickMachine services..."
    kill 0
}
trap cleanup EXIT INT TERM

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

# Start Backend
"$PYTHON_BIN" -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Wait for backend to be healthy
echo "Waiting for FastAPI backend to initialize..."
for i in {1..30}; do
    if curl -s http://127.0.0.1:8000/api/health >/dev/null 2>&1; then
        echo "✅ FastAPI backend is healthy at http://localhost:8000"
        break
    fi
    sleep 0.5
done

# Start Frontend
cd "$DIR/frontend"
echo "Starting Next.js frontend..."
npm run dev &
FRONTEND_PID=$!

wait
