# Implementation checks

CPU-only implementation validation on 2026-11-27 passed:

- Parsed every Python script with `ast.parse` and every shell script with `bash -n`.
- Parsed every JSON manifest.
- Generated all four launch contracts (`gsq` and `mx8`, each at 270336 and 131072 context) with `serve.py --dry-run`; checked the declared context, FP8 KV, spec8, suffix0, parallel1, vision off, ten workers, 512/4096 MiB reserves, loopback host, token fitting, read-only model bind, clear environment and network namespace flags.
- Invalid context caps 0 and 1048577 both returned argument error status 2.
- `git diff --check` passed for the tracked working tree at check time.

These checks neither read the full models nor allocate GPU memory. They do not establish that a fresh download, conversion, build, sandbox launch or GPU inference succeeds. Those require the separately recorded acceptance run. No full reconstruction or new runtime result is claimed by this file.

Repeat the cheap syntax and launch inspection from WSL:

```bash
python3 - <<'PY'
import ast, json, pathlib, subprocess
for p in pathlib.Path('scripts').glob('*.py'):
    ast.parse(p.read_text(), filename=str(p))
for p in pathlib.Path('scripts').glob('*.sh'):
    subprocess.run(['bash', '-n', str(p)], check=True)
for p in pathlib.Path('manifests').glob('*.json'):
    json.loads(p.read_text())
print('Syntax and JSON PASS')
PY
python3 scripts/serve.py --profile gsq --dry-run
python3 scripts/serve.py --profile mx8 --context 131072 --dry-run
```

Full runtime smoke is `bash scripts/smoke.sh <context>`. It saves revision, command, inputs, expected and observed results in the local state directory. Never publish local request/response logs without reviewing them.

## WSL2 GPU acceptance on 2026-11-27

Built the pinned source tree with CUDA 13.0.2 and GCC 13.3 on the RTX 5080 reference machine. The new binary SHA256 was `975bfc98aaacdf86aeb479c4ac298bdbb1d4b5541067f67c8db33e9757bff330`. The new virtual environment passed `pip check`.

Assembled both packs using `bash scripts/prepare-packs.sh both` in a separate WSL state directory. The reconstruction reused local copies of the three large pinned input files and verified their SHA256 before conversion. Each derived pack passed the full section, tensor, geometry and tokenizer checks.

| Reconstructed pack | Bytes | SHA256 |
| --- | ---: | --- |
| GSQ IQ3_S + DFlash 2 | 13451914629 | `4bd14a49d986de6624857590e3c16462f7a73d2b4c44ba79d4f0c335bc49cd93` |
| MX8 + DFlash 2 | 19598669487 | `c356ff04a8bd42caa927b197fc90f6a2cb77c3a4fa111d15fd772519e1bab4c5` |

Launched both from a clean GitHub checkout at `3ddaea7ff86743e2807f09533aadd1bd7616f8f5` using the rebuilt Basalt and reconstructed packs, one profile at a time. Both passed the sandbox visibility check, reported loaded at 270336 tokens, returned 391 from the documented smoke request, completed a synthetic tool call and replayed `reasoning_content` on continuation. The larger-context experiment and six-turn Pi measurements are described in the README.

## Explicit context tuning: focused repair check

CPU-only dry-run checks on 2026-11-27 passed after adding `--expert-cache` and `--vram-reserve-mib`:

- The complete default command remains byte-for-byte equal as an argument list to the original default; automatic expert-cache sizing is preserved by omitting that option.
- Both profiles accept `--context 393216 --expert-cache 450 --vram-reserve-mib 768` and pass exactly those values to Basalt.
- Expert-cache bounds 1 and 3072 pass. The upper bound is the model's 24 physical layers times 128 routed experts; it is an argument bound, not a fit guarantee.
- All three VRAM reserves (512, 768, 1024) pass; default remains 512.
- Cache values -1, 0, 3073 and non-integer `x`; reserve values 0, 513, 2048 and `x`; and context values 0 and 1048577 all return argument error status 2.
- Python syntax parsing passes. No model read, conversion or GPU allocation occurred.

Exact dry-run command:

```bash
python3 scripts/serve.py --profile gsq --context 393216 --expert-cache 450 --vram-reserve-mib 768 --dry-run
```

Observed Basalt command:

```text
/basalt/bin/basalt serve /model/Inkling-2-Mini-GSQ-RCO-IQ3_S-DFlash2.basalt --alias inkling-2-mini --host 127.0.0.1 --port 8191 --parallel 1 --max-context 393216 --kv fp8 --spec 8 --suffix-draft 0 --vision off --pool-workers 10 --vram-reserve-mib 768 --ram-reserve-mib 4096 --fit-max-tokens --request-lines --expert-cache 450
```

This check proves the public launcher can express the independently tested settings. Runtime long-context results belong to the coordinator's separate acceptance evidence.
