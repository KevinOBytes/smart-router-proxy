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

# exec -a sets a clean process name so the proxy shows as "smart-router-proxy"
# (not "bash" / "python3.13") in Activity Monitor / the macOS Background
# activity register. Exec the python binary directly (not the shebang script)
# so the argv[0] override actually sticks. $@ is passthrough for extra flags.
exec -a "smart-router-proxy" .venv/bin/python3 -m smart_router_proxy.server --config config.yaml "$@"
