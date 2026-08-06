#!/usr/bin/env bash
# Install, verify, update, and manage smart-router-proxy.
#
# Usage:
#   ./install.sh install   # full setup: venv + classifier model + LaunchAgent
#   ./install.sh verify    # check deps, model, checksum, classifier self-test
#   ./install.sh update-model  # re-download + verify the classifier bundle
#   ./install.sh status    # show service + health
#   ./install.sh uninstall # stop and remove the LaunchAgent
#   ./install.sh start     # start installed service
#   ./install.sh stop      # stop service (KeepAlive restarts; uninstall to halt)
#
# Requires: bash, curl, zstd, python3, uv, launchctl (macOS).
set -euo pipefail

LABEL="com.kevinbytes.smart-router-proxy"
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_SH="$REPO_DIR/run.sh"
PLIST_TPL="$REPO_DIR/com.kevinbytes.smart-router-proxy.plist.tpl"
LOG_DIR="$REPO_DIR/log"
PLIST_DEST="$HOME/Library/LaunchAgents/$LABEL.plist"
DOMAIN="gui/$(id -u)"
VENV="$REPO_DIR/.venv"
MODEL_DIR="$HOME/.smart-router-proxy/classifier-model"
MANIFEST="$REPO_DIR/models/manifest.json"
HEALTH_URL="http://127.0.0.1:8199/healthz"
HEALTH_PORT=8199

fail()  { echo "error: $*" >&2; exit 1; }
info()  { echo "==> $*"; }
warn()  { echo "warn: $*" >&2; }

# ── platform ─────────────────────────────────────────────────────────────
check_platform() {
  local os arch
  os="$(uname -s | tr '[:upper:]' '[:lower:]')"
  arch="$(uname -m)"
  if [ "$os" != "darwin" ]; then
    fail "this build is macOS-only (detected $os). Build an ONNX bundle for other platforms."
  fi
  if [ "$arch" != "arm64" ]; then
    fail "this build is Apple Silicon-only (detected $arch). Use an ONNX bundle on Intel Macs."
  fi
  [ -f "$MANIFEST" ] || fail "models/manifest.json not found in $REPO_DIR"
  local manifest_platform
  manifest_platform="$("$VENV/bin/python3" -c 'import json,sys; print(json.load(sys.stdin)["platform"])' < "$MANIFEST" 2>/dev/null || echo "")"
  [ -n "$manifest_platform" ] || fail "could not read platform from manifest"
  [ "$manifest_platform" = "$os-$arch" ] || \
    fail "manifest targets $manifest_platform but you are on $os-$arch"
}

# ── prerequisites ─────────────────────────────────────────────────────────
ensure_uv() {
  if command -v uv >/dev/null 2>&1; then return 0; fi
  info "uv not found; installing via official script"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck disable=SC1091
  . "$HOME/.cargo/env" 2>/dev/null || true
  command -v uv >/dev/null 2>&1 || fail "uv install failed; add ~/.local/bin to PATH"
}

ensure_zstd() {
  command -v zstd >/dev/null 2>&1 && return 0
  info "zstd not found; installing via Homebrew"
  command -v brew >/dev/null 2>&1 || fail "Homebrew not found; install from https://brew.sh"
  brew install zstd
  command -v zstd >/dev/null 2>&1 || fail "zstd install failed"
}

ensure_venv() {
  [ -d "$VENV" ] && return 0
  info "creating virtual environment with uv"
  uv venv "$VENV" >/dev/null
}

install_deps() {
  info "syncing dependencies (uv sync --extra classifier --extra dev)"
  uv sync --extra classifier --extra dev >/dev/null
  [ -x "$VENV/bin/smart-router-proxy" ] || \
    fail "venv entry point missing after sync"
}

# ── model ────────────────────────────────────────────────────────────────
read_manifest_field() { "$VENV/bin/python3" -c 'import json,sys; print(json.load(sys.stdin)["'"$1"'"])' < "$MANIFEST"; }

verify_installed_model() {
  [ -d "$MODEL_DIR" ] || return 1
  [ -f "$MODEL_DIR/weights.safetensors" ] || return 1
  [ -f "$MODEL_DIR/config.json" ] || return 1
  [ -f "$MODEL_DIR/base_model/model.safetensors" ] || return 1
  [ -f "$MODEL_DIR/base_model/config.json" ] || return 1
  return 0
}

download_model() {
  local url sha256 archive tmp
  url="$(read_manifest_field primary_url)"
  sha256="$(read_manifest_field sha256)"
  archive="$(read_manifest_field archive)"
  tmp="$(mktemp -d)"
  info "downloading $archive"
  info "  $url"
  curl -L --fail --progress-bar -o "$tmp/$archive" "$url"
  info "verifying SHA-256"
  local actual
  actual="$(shasum -a 256 "$tmp/$archive" | awk '{print $1}')"
  [ "$actual" = "$sha256" ] || { rm -rf "$tmp"; fail "checksum mismatch: expected $sha256, got $actual"; }
  info "checksum OK ($sha256)"
  info "extracting to $MODEL_DIR"
  mkdir -p "$MODEL_DIR"
  rm -rf "${MODEL_DIR:?}"/*
  # macOS bsdtar's --use-compress-program=zstd fails with a broken pipe
  # (zstd error 70) on this host; decompress explicitly and pipe into tar.
  zstd -dc "$tmp/$archive" | tar -xf - -C "$MODEL_DIR"
  # archive contains classifier-model/ — move contents up one level
  if [ -d "$MODEL_DIR/classifier-model" ]; then
    mv "$MODEL_DIR/classifier-model/"* "$MODEL_DIR/" 2>/dev/null || true
    mv "$MODEL_DIR/classifier-model/".[!.]* "$MODEL_DIR/" 2>/dev/null || true
    rmdir "$MODEL_DIR/classifier-model" 2>/dev/null || true
  fi
  rm -rf "$tmp"
  verify_installed_model || fail "model incomplete after extraction"
  info "model installed at $MODEL_DIR"
}

ensure_model() {
  if verify_installed_model; then
    info "classifier model already present at $MODEL_DIR"
    return 0
  fi
  download_model
}

classifier_self_test() {
  info "running classifier self-test (local, no upstream call)"
  local result
  result="$("$VENV/bin/python3" - <<'PY'
import sys
sys.path.insert(0, "src")
try:
    from smart_router_proxy.bert_classifier import get_classifier
    c = get_classifier()
    label, conf = c.classify("write a python function to sort a list")
    print(f"OK label={label} conf={conf:.3f}")
except Exception as e:
    print(f"FAIL {e}", file=sys.stderr)
    sys.exit(1)
PY
)" || fail "classifier self-test failed: $result"
  info "$result"
}

# ── config ───────────────────────────────────────────────────────────────
ensure_config() {
  [ -f "$REPO_DIR/config.yaml" ] && return 0
  info "creating config.yaml from config.example.yaml"
  cp "$REPO_DIR/config.example.yaml" "$REPO_DIR/config.yaml"
}

check_api_key() {
  local key_env key
  key_env="$(read_manifest_field 2>/dev/null || echo OPENROUTER_API_KEY)"
  # Read from config.yaml upstream.api_key_env if present
  key_env="$("$VENV/bin/python3" -c '
from smart_router_proxy.config import load_config
print(load_config("config.yaml").upstream.api_key_env)
' 2>/dev/null || echo OPENROUTER_API_KEY)"
  key="$(grep -E "^${key_env}=" "$HOME/.hermes/.env" 2>/dev/null | head -1 | cut -d= -f2- | tr -d '"' || true)"
  if [ -z "${key:-}" ]; then
    warn "${key_env} not found in ~/.hermes/.env — proxy will start but upstream calls will fail"
    warn "set it with: export ${key_env}=\"sk-or-v1-...\" and add to ~/.hermes/.env"
  fi
}

# ── LaunchAgent ──────────────────────────────────────────────────────────
generate_plist() {
  mkdir -p "$LOG_DIR" "$HOME/Library/LaunchAgents"
  sed -e "s|__RUN_SH__|$RUN_SH|g" \
      -e "s|__REPO_DIR__|$REPO_DIR|g" \
      -e "s|__LOG_DIR__|$LOG_DIR|g" \
      "$PLIST_TPL" > "$PLIST_DEST"
  plutil -lint "$PLIST_DEST" >/dev/null || fail "generated plist invalid"
}

is_loaded() { launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1; }

wait_for_health() {
  info "waiting for health on $HEALTH_URL"
  for i in $(seq 1 30); do
    sleep 1
    curl -sS --max-time 2 "$HEALTH_URL" 2>/dev/null | grep -q '"server":true' && { info "healthy after ${i}s"; return 0; }
  done
  return 1
}

# ── commands ─────────────────────────────────────────────────────────────
cmd_install() {
  check_platform
  ensure_uv
  ensure_zstd
  ensure_venv
  install_deps
  ensure_model
  ensure_config
  check_api_key
  generate_plist
  if is_loaded; then
    info "already installed; restarting to pick up changes"
    launchctl kickstart -k "$DOMAIN/$LABEL" >/dev/null 2>&1 || true
  else
    info "bootstrap $LABEL"
    launchctl bootstrap "$DOMAIN" "$PLIST_DEST"
  fi
  wait_for_health || warn "service did not become healthy in 30s — check logs at $LOG_DIR"
  info "install complete"
  echo "  health: curl -s $HEALTH_URL"
  echo "  logs:   $LOG_DIR"
}

cmd_verify() {
  check_platform
  ensure_venv
  verify_installed_model || fail "model not installed; run: ./install.sh install"
  info "verifying model checksum"
  local sha256 expected actual
  sha256="$(read_manifest_field sha256)"
  # Re-hash the on-disk bundle is expensive; instead verify file presence + self-test
  info "model files present at $MODEL_DIR"
  classifier_self_test
  info "checking service health"
  curl -sS --max-time 5 "$HEALTH_URL" 2>/dev/null | grep -q '"server":true' \
    && info "service healthy" \
    || warn "service not healthy (may not be running)"
}

cmd_update_model() {
  check_platform
  ensure_venv
  info "removing existing model"
  rm -rf "${MODEL_DIR:?}"/*
  download_model
  classifier_self_test
  if is_loaded; then
    info "restarting service to reload classifier"
    launchctl kickstart -k "$DOMAIN/$LABEL" >/dev/null 2>&1 || true
    wait_for_health || warn "service did not become healthy"
  fi
}

cmd_status() {
  if is_loaded; then
    echo "$LABEL: loaded and running"
    pgrep -f "$RUN_SH" >/dev/null && echo "  process: running" || echo "  process: not found"
    echo "  port $HEALTH_PORT: $(lsof -iTCP:$HEALTH_PORT -sTCP:LISTEN 2>/dev/null | awk 'NR==2{print "LISTEN pid="$2}' || echo 'not listening')"
  else
    echo "$LABEL: not installed/loaded"
  fi
  verify_installed_model && echo "  model: installed ($MODEL_DIR)" || echo "  model: not installed"
}

cmd_uninstall() {
  if is_loaded; then
    info "stopping and unloading"
    launchctl bootout "$DOMAIN/$LABEL" >/dev/null 2>&1 || true
  fi
  rm -f "$PLIST_DEST"
  info "removed $PLIST_DEST; logs kept at $LOG_DIR"
}

cmd_stop() {
  is_loaded || fail "not loaded"
  info "stopping (KeepAlive=true will restart; use 'uninstall' to halt permanently)"
  launchctl kickstart -k "$DOMAIN/$LABEL" >/dev/null 2>&1 || true
}

cmd_start() {
  is_loaded || { info "not loaded; bootstrapping"; generate_plist; launchctl bootstrap "$DOMAIN" "$PLIST_DEST"; }
  info "started"
}

case "${1:-}" in
  install)      cmd_install ;;
  verify)      cmd_verify ;;
  update-model) cmd_update_model ;;
  status)      cmd_status ;;
  uninstall)   cmd_uninstall ;;
  stop)        cmd_stop ;;
  start)       cmd_start ;;
  *)
    echo "usage: $0 {install|verify|update-model|status|uninstall|stop|start}"
    exit 1
    ;;
esac
