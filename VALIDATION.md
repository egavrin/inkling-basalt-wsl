# Implementation checks

## Selected GSQ profile publication (2026-11-27)

This publication tested stock GSQ at 270336 total tokens, automatic experts, a 768 MiB VRAM reserve and 8 CPU workers. The explicit 393216 configuration uses 450 experts, 768 MiB and 10 workers. MX8 and other-context defaults use automatic experts, 512 MiB and 10 workers. `--pool-workers` can override the count. The API model alias is `inkling-2-mini` and the Pi example sets `contextWindow` to 270336.

The [sanitized benchmark aggregate](benchmarks/gsq-270336.json) records independently collected local results using Basalt tree `0026804d2191e2c55ff7cb4d9eddfa122f4ac511`, matching this recipe's pinned tree, and the reference GSQ pack. No source, dependency or weights pin changed. The benchmark used a local harness and a different local API alias. Publication checks validate the public launcher on the CPU.

The matched input was 265477 cached plus 1973 fresh tokens, giving 267450 total, or 98.93% of 270336. Each selected repeat measured six warm Pi turns after an excluded warmup. Median decode was 228.15 and 229.25 tokens/s. Median full turn time was 5.15597 and 5.11517 s. Minimum sampled Windows available system RAM was 8466 and 8740 MiB. The matched 384K controls measured 67.30 and 67.80 tokens/s, 15.00869 and 14.61557 s, and 4680 and 3994 MiB. These are sampled host RAM minima during request intervals. Continuous minima may be lower between samples. Historical results below and in the README are separate runs.

The selected profile passed a 131092-token marker/tool qualification and a settled 262184-token marker/continuation qualification. The latter reused 262221 cached tokens and handled a requested 32768-token output cap with 8084 tokens left through token fitting. The settled minimum Windows available RAM was 9434 MiB. A rapid-restart attempt returned correct markers and continuation but failed the 8192 MiB memory criterion at 6915 MiB. Memory qualification therefore depends on host state, including restart settling.

Publication validation was CPU-only. WSL dry-run assertions passed for GSQ default (270336/auto/768/8), GSQ 393216 (393216/450/768/10), MX8 default (270336/auto/512/10), GSQ 131072 (131072/auto/512/10), and explicit expert/reserve/worker overrides. Worker counts 17 and 2147483647 passed through unchanged. Counts 0, -1, 2147483648 and non-integer `x` returned argument error status 2. The API alias remained `inkling-2-mini` in all checked commands, and `git diff --check` passed. Use `python3 scripts/serve.py --profile gsq --dry-run` to inspect the selected command, and add `--context 393216` for the retained larger-window defaults. These commands inspect arguments on the CPU. The aggregate contains method, binary/model identity and reviewed measurements. Workload replay requires the private fixtures.

## Historical implementation checks

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

The subsequent dependency update pins Pillow 13.0.0, the patched version reported by GitHub. The updated virtual environment passed `pip check`, a Pillow image API check and Basalt's server help startup.

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


## Pi-preview default profile

The speed and context measurements above apply to stock GSQ. In a separate
local Pi-preview comparison, the first LoRA generation needed a follow-up repair.
The public recipe preserves the tested 270336 / FP8 / spec 8 / suffix 0 /
automatic experts / 8 workers / 768 MiB VRAM reserve / 4096 MiB RAM reserve /
parallel 1 / no vision / fit-output settings and the pinned Basalt tree.

The local accepted pack was 13,451,914,797 bytes with SHA256
`8e5cd88cc5c7ae75e08f9a690e48fe86f6bdf30d91af265d2feebf06c14385cb`.
Clean reconstruction can have a different whole-file hash because packaging
metadata differs. `pi-pack.py` first applies the stock equivalence checks and
then verifies 96 expected adapted tensor hashes, all 641 untouched tensors
(including 81 DFlash tensors), unchanged metadata and all section checksums.
It records the actual output hash used by the launcher.

Publication acceptance on the WSL2 RTX 5080 reused the previously reconstructed,
verified stock GSQ pack and rebuilt the adapted pack with `bash scripts/prepare-packs.sh`.
The downloader fetched and verified the pinned adapter from Hugging Face. The
result was 13,451,914,629 bytes with SHA256
`c6b9e38a6bcd00fd76fba8acd3683873e544461e0cd08f57cabb51b241fb90a8`.
All 96 merged tensors matched the reference hashes. The verification also
confirmed the 641 unchanged tensors, including all 81 DFlash tensors.

`python3 scripts/serve.py --check-sandbox` passed, and `python3 scripts/serve.py`
loaded the rebuilt default pack at 270336 tokens over its private Unix socket.
After a fresh `bash scripts/install-pi.sh`, the public Pi launcher returned the
official [Three.js OrbitControls documentation](https://threejs.org/docs/pages/OrbitControls.html)
through a live Exa query nested in Code Mode. CPU checks covered Pi startup with
and without search, missing-credential handling and workspace isolation. The
explicit stock GSQ (270336 and 393216) and MX8 launch contracts matched their
previous definitions.
