#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
STATE=${INKLING_STATE:-"$REPO/.state"}
mkdir -p "$STATE"
STATE=$(realpath "$STATE")
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0}
export REPO STATE CUDA_HOME
