#!/usr/bin/env bash
source "$(dirname -- "$0")/common.sh"
# Run on the host during setup; Pi itself runs offline in its namespace.
for tool in node npm bwrap rg; do command -v "$tool" >/dev/null; done
FD=$(command -v fd || command -v fdfind)
[[ $(node -p 'process.versions.node.split(".")[0]') -ge 22 ]] || { echo 'Node 22 or newer required' >&2; exit 2; }
mkdir -p "$STATE/pi-runtime/runtime"
cp -L "$(command -v node)" "$STATE/pi-runtime/runtime/node"
cp -L "$(command -v rg)" "$STATE/pi-runtime/runtime/rg"
cp -L "$FD" "$STATE/pi-runtime/runtime/fd"
npm install --prefix "$STATE/pi-runtime/pi" --save-exact @earendil-works/pi-coding-agent@1.6.0 @aliou/pi-guardrails@0.24.0 pi-token-stats@0.1.1
