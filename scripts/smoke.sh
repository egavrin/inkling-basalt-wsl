#!/usr/bin/env bash
source "$(dirname -- "$0")/common.sh"
context=${1:-270336}
socket="$STATE/run/inference.sock"
out="$STATE/smoke-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir "$out"
git -C "$REPO" rev-parse HEAD > "$out/revision.txt"
printf 'bash scripts/smoke.sh %s\n' "$context" > "$out/command.txt"
curl --fail --silent --show-error --max-time 30 --unix-socket "$socket" http://localhost/health > "$out/health.json"
python3 - "$out/health.json" "$context" <<'PY'
import json,sys
h=json.load(open(sys.argv[1]))
assert h['loaded'] is True and h['loading'] is False, h
assert h['model']=='inkling-2-mini' and h['max_context']==int(sys.argv[2]), h
PY
printf '%s\n' '{"model":"inkling-2-mini","messages":[{"role":"user","content":"Calculate 17 * 23. End your answer with the number."}],"temperature":0,"max_tokens":256,"stream":false}' > "$out/request.json"
curl --fail --silent --show-error --max-time 300 --unix-socket "$socket" \
    -H 'Content-Type: application/json' --data-binary "@$out/request.json" \
    http://localhost/v1/chat/completions > "$out/response.json"
python3 - "$out" <<'PY'
import json,pathlib,sys
p=pathlib.Path(sys.argv[1]); r=json.loads((p/'response.json').read_text())
answer=r['choices'][0]['message'].get('content') or ''
ok='391' in answer and r['choices'][0]['finish_reason']=='stop'
(p/'result.json').write_text(json.dumps({'pass':ok,'expected':'final answer contains 391; finish_reason stop','observed':answer,'finish_reason':r['choices'][0]['finish_reason']},indent=2)+'\n')
assert ok,r
print('Health + inference PASS; saved evidence:',p)
PY
