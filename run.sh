#!/usr/bin/env bash
# Launch smart-router-proxy with its configured upstream key from ~/.hermes/.env.
set -euo pipefail
cd "$(dirname "$0")"

# Parse api_key_env from config.yaml so changing the destination endpoint does
# not require editing this launcher. Extract the value without sourcing the
# dotenv file (it may contain unquoted multi-word values).
KEY_ENV="$(.venv/bin/python3 - <<'PY'
from smart_router_proxy.config import load_config
print(load_config("config.yaml").upstream.api_key_env)
PY
)"
KEY="$(grep -E "^${KEY_ENV}=" "$HOME/.hermes/.env" | head -1 | cut -d= -f2- | tr -d '"')"
if [ -z "${KEY:-}" ]; then
  echo "FATAL: ${KEY_ENV} not found in ~/.hermes/.env" >&2
  exit 1
fi

export "${KEY_ENV}=${KEY}"

# exec -a sets a clean process name so the proxy shows as "smart-router-proxy"
# (not "bash" / "python3.13") in Activity Monitor / the macOS Background
# activity register. Exec the python binary directly (not the shebang script)
# so the argv[0] override actually sticks. $@ is passthrough for extra flags.
exec -a "smart-router-proxy" .venv/bin/python3 -m smart_router_proxy.server --config config.yaml "$@"
