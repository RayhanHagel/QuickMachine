#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR/frontend"

echo "Starting QuickMachine Next.js frontend on http://localhost:3000..."
npm run dev
