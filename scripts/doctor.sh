#!/usr/bin/env bash
source "$(dirname -- "$0")/common.sh"
for tool in git curl cmake ninja g++-13 python3 bwrap; do
    command -v "$tool" >/dev/null || { echo "Missing prerequisite: $tool" >&2; exit 1; }
done
[[ $(uname -m) == x86_64 ]]
[[ -e /dev/dxg && -d /usr/lib/wsl/lib ]]
"$CUDA_HOME/bin/nvcc" --version | grep 'release 13.0'
[[ $(g++-13 -dumpversion) == 13 ]]
python3 -c 'import sys; assert sys.version_info[:2] == (3,12), sys.version; import venv'
echo 'Prerequisite tools and WSL GPU paths found. This does not allocate GPU memory or validate model fit.'
