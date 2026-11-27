#!/usr/bin/env bash
source "$(dirname -- "$0")/common.sh"
export INKLING_STATE="$STATE"
profile=${1:-both}
[[ $profile == gsq || $profile == mx8 || $profile == both ]] || { echo 'Expected gsq, mx8 or both' >&2; exit 2; }
python3 "$REPO/scripts/download.py" "$profile"
[[ $(git -C "$STATE/basalt" rev-parse 'HEAD^{tree}') == 0026804d2191e2c55ff7cb4d9eddfa122f4ac511 ]]
[[ -z $(git -C "$STATE/basalt" status --porcelain --untracked-files=no) ]]
mkdir -p "$STATE/models"
if [[ $profile != mx8 && ! -f $STATE/models/gsq-base.basalt ]]; then
    "$STATE/venv/bin/python" "$STATE/basalt/tools/convert/convert.py" \
        --gguf "$STATE/inputs/gsq/Inkling-2-Mini-GSQ-RCO-IQ3_S.gguf" \
        --sections-from "$STATE/inputs/mx8/Inkling-2-Mini-NVFP4-MX8.basalt" \
        --out "$STATE/models/gsq-base.basalt" --workers 1
fi
"$STATE/venv/bin/python" "$REPO/scripts/pack.py" build "$profile"
