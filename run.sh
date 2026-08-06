#!/usr/bin/env bash
# Launch smart-router-proxy with OPENROUTER_API_KEY pulled from ~/.hermes/.env
set -euo pipefail
cd "$(dirname "$0")"

KEY="$(grep -E '^OPENROUTER_API_KEY=' "$HOME/.hermes/.env" | head -1 | cut -d= -f2- | tr -d '"')"
if [ -z "${KEY:-}" ]; then
  echo "FATAL: OPENROUTER_API_KEY not found in ~/.hermes/.env" >&2
  exit 1
fi

export OPENROUTER_API_KEY="$KEY"
exec .venv/bin/smart-router-proxy --config config.yaml
