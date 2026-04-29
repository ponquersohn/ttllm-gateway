#!/usr/bin/env bash
set -euo pipefail

if command -v claude >/dev/null 2>&1; then
    echo "==> Claude Code already installed: $(claude --version)"
    exit 0
fi

echo "==> Installing Claude Code..."
curl -fsSL https://claude.ai/install.sh | bash
