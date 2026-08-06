#!/usr/bin/env bash
# Install, start, stop, and uninstall smart-router-proxy as a macOS LaunchAgent.
#
# Usage:
#   ./install.sh install      # install as a login service and start it
#   ./install.sh uninstall    # stop and remove the service
#   ./install.sh status       # show whether it's loaded / running
#   ./install.sh stop         # stop but keep installed
#   ./install.sh start        # start installed service
#
# Requires: bash, launchctl, plutil. macOS only.
set -euo pipefail

LABEL="com.kevinbytes.smart-router-proxy"
PLIST_TPL="$(cd "$(dirname "$0")" && pwd)/com.kevinbytes.smart-router-proxy.plist.tpl"
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_SH="$REPO_DIR/run.sh"
LOG_DIR="$REPO_DIR/log"
PLIST_DEST="$HOME/Library/LaunchAgents/$LABEL.plist"
DOMAIN="gui/$(id -u)"

fail() { echo "error: $*" >&2; exit 1; }
info() { echo "==> $*"; }

ensure_repo_files() {
    [ -f "$RUN_SH" ] || fail "run.sh not found in $REPO_DIR (wrong dir?)"
    [ -f "$PLIST_TPL" ] || fail "plist template not found"
    [ -x "$RUN_SH" ] || fail "run.sh not executable; run: chmod +x run.sh"
    [ -f "$REPO_DIR/.venv/bin/smart-router-proxy" ] || \
        fail "venv entry point missing; create venv first (see README)"
}

generate_plist() {
    mkdir -p "$LOG_DIR" "$HOME/Library/LaunchAgents"
    sed -e "s|__RUN_SH__|$RUN_SH|g" \
        -e "s|__REPO_DIR__|$REPO_DIR|g" \
        -e "s|__LOG_DIR__|$LOG_DIR|g" \
        "$PLIST_TPL" > "$PLIST_DEST"
    plutil -lint "$PLIST_DEST" >/dev/null || fail "generated plist invalid"
}

is_loaded() {
    launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1
}

cmd_install() {
    ensure_repo_files
    generate_plist
    if is_loaded; then
        info "already installed; restarting to pick up any changes"
        launchctl kickstart -k "$DOMAIN/$LABEL" >/dev/null 2>&1 || true
    else
        info "bootstrap $LABEL"
        launchctl bootstrap "$DOMAIN" "$PLIST_DEST"
    fi
    info "installed and started. Logs: $LOG_DIR"
    echo "  check:  launchctl print $DOMAIN/$LABEL"
    echo "  health: curl -s http://127.0.0.1:8199/healthz"
}

cmd_uninstall() {
    if is_loaded; then
        info "stopping and unloading"
        launchctl bootout "$DOMAIN/$LABEL" >/dev/null 2>&1 || true
    fi
    rm -f "$PLIST_DEST"
    info "removed $PLIST_DEST; logs kept at $LOG_DIR"
}

cmd_status() {
    if is_loaded; then
        echo "$LABEL: loaded and running"
        if pgrep -f "$RUN_SH" >/dev/null; then echo "  process: running ($(pgrep -f "$RUN_SH" | tr '\n' ' '))"; fi
        echo "  port 8199: $(lsof -iTCP:8199 -sTCP:LISTEN 2>/dev/null | awk 'NR==2{print "LISTEN pid="$2}' || echo 'not listening')"
    else
        echo "$LABEL: not installed/loaded"
    fi
}

cmd_stop() {
    is_loaded || fail "not loaded"
    info "stopping (installed service stays; relaunches at next login unless disabled)"
    launchctl kickstart -k "$DOMAIN/$LABEL" >/dev/null 2>&1 || true
    # KeepAlive=true restarts it immediately; to truly stop, uninstall instead.
    info "note: KeepAlive=true means stop restarts it. Use 'uninstall' to fully stop."
}

cmd_start() {
    is_loaded || { info "not loaded; bootstrapping"; generate_plist; launchctl bootstrap "$DOMAIN" "$PLIST_DEST"; }
    info "started"
}

case "${1:-}" in
    install)   cmd_install ;;
    uninstall) cmd_uninstall ;;
    status)    cmd_status ;;
    stop)      cmd_stop ;;
    start)     cmd_start ;;
    *)
        echo "usage: $0 {install|uninstall|status|stop|start}"
        exit 1
        ;;
esac
