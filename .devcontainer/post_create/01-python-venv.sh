#!/usr/bin/env bash
set -euo pipefail

echo "==> Setting up Python virtual environment..."

until command -v python3 >/dev/null 2>&1; do
    echo "Waiting for python3..."
    sleep 2
done

sudo chown vscode:vscode .venv
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
