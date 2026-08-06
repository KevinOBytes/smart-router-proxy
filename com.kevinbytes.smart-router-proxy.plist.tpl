<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<!--
  LaunchAgent for smart-router-proxy.

  Managed by install.sh — do not edit by hand (paths are generated at
  install time). Run:
      ./install.sh install    # install as a login service
      ./install.sh status
      ./install.sh uninstall
-->
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.kevinbytes.smart-router-proxy</string>

    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>__RUN_SH__</string>
    </array>

    <key>WorkingDirectory</key>
    <string>__REPO_DIR__</string>

    <!-- Start at login -->
    <key>RunAtLoad</key>
    <true/>

    <!-- Restart on crash -->
    <key>KeepAlive</key>
    <true/>

    <key>ProcessType</key>
    <string>Background</string>

    <key>StandardOutPath</key>
    <string>__LOG_DIR__/proxy.log</string>
    <key>StandardErrorPath</key>
    <string>__LOG_DIR__/proxy-error.log</string>
</dict>
</plist>
