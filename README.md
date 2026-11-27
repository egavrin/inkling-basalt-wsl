# Inkling 2 Mini + Basalt on WSL2

Reproduce and run **GSQ-RCO IQ3_S + IncoAI DFlash 2** or **MX8 + the same DFlash 2 checkpoint** in a private WSL2 GPU sandbox. The source, model inputs, conversion settings, tensor hashes and default launch arguments are pinned. Everything lives in this checkout or the explicit `INKLING_STATE` directory; no pre-existing personal project is required.

This is a community setup, not an official release from Thinking Machines, ISTA-DASLab, IncoAI or Basalt. Models retain their upstream licenses. See [MODEL-NOTICE.txt](MODEL-NOTICE.txt) and [MODEL-LICENSE.txt](MODEL-LICENSE.txt).

## Requirements

- Windows with WSL2, Ubuntu 24.04 x86-64, and an NVIDIA Windows driver supporting CUDA 13.0. `/dev/dxg` and `/usr/lib/wsl/lib` must be available. Install the GPU driver on Windows; do not install a Linux display driver in WSL.
- CUDA Toolkit **13.0** (the reference build used **13.0.2**), GCC/G++ **13**, Python **3.12**, CMake, Ninja, Git, curl, bubblewrap, and Python venv support.
- Working unprivileged user namespaces for bubblewrap. The launcher fails closed if the sandbox cannot start. It does not fall back to an unrestricted process.
- Sufficient system RAM, GPU memory and SSD space. Plan for at least **100 GB free** for both upstream reconstructions, intermediates, source and build. Full SHA verification reads tens of GB. Avoid concurrent conversion, hashing and GPU benchmarks. Both profiles default to 270336 total tokens (264K). GSQ also offers an opt-in 393216-token (384K) configuration. Available context depends on the GPU and host memory.

Ubuntu system packages (CUDA is a separate prerequisite):

```bash
sudo apt-get update
sudo apt-get install -y git curl cmake ninja-build g++-13 python3 python3-venv bubblewrap
```

Obtain CUDA 13.0.2 from the [NVIDIA CUDA Toolkit archive](https://developer.nvidia.com/cuda-toolkit-archive), selecting WSL-Ubuntu. Set `CUDA_HOME` to its installation root. The build checks for the 13.0 compiler family and GCC 13. It does not change WSL resource limits, restart WSL or modify other services.

## Install from pinned inputs

Run inside WSL. A Linux filesystem checkout and state directory are recommended for build speed and Unix socket permissions.

```bash
git clone https://github.com/egavrin/inkling-basalt-wsl.git
cd inkling-basalt-wsl
export INKLING_STATE="$HOME/.local/share/inkling-basalt-wsl"
export CUDA_HOME=/usr/local/cuda-13.0
bash scripts/doctor.sh
bash scripts/build.sh
bash scripts/prepare-packs.sh
```

Keep these two environment variables set in each terminal. Without `INKLING_STATE`, scripts use `.state/` under this checkout. Without `CUDA_HOME`, they use `/usr/local/cuda-13.0`. No files in either location should be committed.

`bash scripts/prepare-packs.sh gsq|mx8|both` selects outputs (default `both`). MX8-only skips the 11.4 GB GSQ input and intermediate; GSQ still needs the official MX8 input for shared sections. `pack.py verify gsq|mx8|both` similarly scopes verification. Input downloads can be run separately with `python3 scripts/download.py gsq|mx8|both`.

`build.sh` creates a clean Basalt checkout, applies the four pinned cherry-picks, checks the exact Git tree, and builds with two jobs (`BUILD_JOBS` overrides this). It installs pinned Python conversion/runtime dependencies in a private venv and writes `binary.sha256`. It never expects different machines to produce byte-identical native binaries.

`prepare-packs.sh` downloads pinned upstream files with resumable curl transfers and SHA256 checks. It converts the GSQ target using the pinned Basalt converter, recreates all 81 DFlash tensors using Basalt's upstream `Dflash2` implementation, and attaches those exact tensors to both targets. DSpark tensors/settings are removed. Existing completed packs are verified rather than overwritten. A checksum failure requires explicit inspection/removal of the reported corrupt file before retrying.

Each completed pack must match the reference manifest for every target tensor, every DFlash tensor, tensor type/shape/size, model and drafter geometry, tokenizer settings, and non-tensor sections (including expert data). Only then is a `.partial` pack promoted to its final filename. Verification writes `$INKLING_STATE/pack-verification.json`.

To repeat full verification later:

```bash
"$INKLING_STATE/venv/bin/python" scripts/pack.py verify
```

The two installable profiles are assembled from public Hugging Face inputs. The resulting `.basalt` files live under `$INKLING_STATE/models/` and can be reused on another compatible installation after `pack.py verify gsq|mx8|both`.

## Run one profile

```bash
python3 scripts/serve.py --check-sandbox
python3 scripts/serve.py --profile gsq
# Or, after stopping the first server with Ctrl-C:
python3 scripts/serve.py --profile mx8
```

The foreground launcher holds a lock in the state directory and hashes the binary and selected pack before starting. This coordinates launches that share a state directory; it does not discover or stop unrelated GPU workloads. Stop other GPU workloads yourself before measuring performance.

The profiles use these defaults:

| Setting | Value |
| --- | --- |
| Context cap | 270336 total tokens for both profiles |
| Expert cache | 450 for GSQ at 393216, automatic otherwise |
| KV cache | FP8 |
| DFlash speculative window | 8 |
| Suffix/prompt-lookup draft | 0 |
| Parallel slots | 1 |
| Vision | off |
| CPU workers | 8 for GSQ at 270336, 10 otherwise |
| VRAM reserve | 768 MiB for GSQ at 270336 or 393216, 512 MiB otherwise |
| RAM reserve | 4096 MiB |
| Token fitting | `--fit-max-tokens` |
| API model alias | `inkling-2-mini` for both profiles |

Inspect the complete launch contract without reading model data or using the GPU:

```bash
python3 scripts/serve.py --profile gsq --dry-run
python3 scripts/serve.py --profile mx8 --context 131072 --dry-run
```

For the larger GSQ configuration, run `python3 scripts/serve.py --profile gsq --context 393216`. This uses 450 GPU experts, a 768 MiB VRAM reserve and 10 workers. The `--context` option accepts caps from 1024 to the model's 1048576-token limit. The cap includes prompt and generated tokens. `--fit-max-tokens` caps output at the remaining window. GSQ at 270336 uses automatic expert placement, a 768 MiB reserve and 8 workers. Other contexts use automatic sizing, a 512 MiB reserve and 10 workers. Set `--expert-cache`, `--vram-reserve-mib` or `--pool-workers` to override those defaults. `--pool-workers` accepts positive 32-bit integers. Check full prompt fit at your chosen context.

## RTX 5080 measurements

The reference machine used a 16 GiB RTX 5080, 32 GiB of system RAM and Ubuntu under WSL2. The selected GSQ profile was measured on 2026-11-27 with **265477 cached + 1973 fresh = 267450 input tokens**, occupying **98.93% of the 270336-token window** before generation. Each repeat reports medians of **six measured warm Pi turns**, following a cold request and an excluded warmup. Decode throughput and full Pi turn time are separate measurements.

| Configuration | Median decode (tokens/s) | Median full warm Pi turn (s) | Minimum sampled Windows available RAM (MiB) |
| --- | ---: | ---: | ---: |
| Selected 270336 / auto experts / 768 MiB / 8 workers, repeat 1 | 228.15 | 5.16 | 8466 |
| Selected same settings, repeat 2 | 229.25 | 5.12 | 8740 |
| Matched 384K control A1 393216 / 450 experts / 768 MiB / 10 workers | 67.30 | 15.01 | 4680 |
| Matched 384K control A2 same settings | 67.80 | 14.62 | 3994 |
| Initial candidate 270336 / auto / 512 MiB / 10 workers | 143.75 | 7.31 | 7958 |
| Settled 10-worker arm c2, 270336 / auto / 768 MiB | 220.50 | 5.31 | 8107 |
| Settled 10-worker arm workers-control-2, same settings | 229.35 | 5.13 | 8543 |
| Worker sweep 270336 / auto / 768 MiB / 12 workers | 193.20 | 5.89 | 8327 |

Controls used the same requests, cached-token counts, model pack and native binary. The 384K controls served the same 267450-token input within their larger window in this comparison. Selected cold prompt-read times were 79.914 and 79.445 s. Memory minima cover samples within cold, warmup and measured request intervals. Continuous minima may be lower between samples. These values measure available **Windows system RAM**.

Reducing the maximum window from 393216 to 270336 frees **1.875 GiB** of full-attention FP8 KV budget at about 16 KiB per token. Automatic expert placement can use the released GPU budget. Sampled final request logs reported roughly 35% GPU expert-cache hits in the 450-expert controls versus 98.6% in the selected arms. This is a sampled mechanism observation. Multiple parameters and host state contribute to the measured result. The initial 512 MiB/10-worker candidate missed the 8192 MiB Windows-memory floor. At 768 MiB, the two 10-worker arms reached similar decode rates of 220.50 and 229.35 tokens/s, with RAM minima of 8107 and 8543 MiB. Both 8-worker repeats passed the memory floor and outperformed 12 workers at the same reserve. The selection reflects the observed speed and memory tradeoff across repeats. Placement and memory headroom depend on the host. The VRAM reserve controls the planner budget, and sampled free VRAM can fall below it.

Qualification recovered all three synthetic markers at 131092 input tokens and passed a tool call. A settled run also recovered all three at 262184 input tokens, replayed a continuation with 262221 cached tokens, and completed a request for a 32768-token output cap through token fitting (8084 tokens remained). Its minimum sampled Windows available RAM was 9434 MiB. An earlier rapid-restart run produced correct answers and continuation but failed the 8192 MiB memory criterion at 6915 MiB. Allow host memory to settle and check available RAM before a large cold request. Memory qualification requires measurements after restart.

These results came from a separate local benchmark harness using the pinned source tree and reference GSQ pack. The public launcher now expresses the selected settings with the stable API alias `inkling-2-mini`. Publication validation uses CPU-only launcher checks. See [VALIDATION.md](VALIDATION.md) and [the sanitized measurement summary](benchmarks/gsq-270336.json) for provenance, exact values and method. The aggregate publishes reviewed measurements and method. Workload replay requires the private fixtures.

### Historical measurements

An earlier 270336-token benchmark used the same Basalt build and DFlash 2 with 265477 cached tokens and 1973 new tokens. Each number below is the median of six continuations from that separate historical run.

| Profile | Decode | Full warm Pi turn |
| --- | ---: | ---: |
| GSQ IQ3_S + DFlash 2 | 97.55 tokens/s | 10.24 s |
| MX8 + DFlash 2 | 64.60 tokens/s | 16.64 s |

The larger GSQ setting below completed a separate synthetic check with 385003 actual input tokens. It found all three markers in 165.27 s and reused 385157 cached tokens on a follow-up at 55.0 decode tokens/s. The longer-context check used a different prompt and one follow-up.

```bash
python3 scripts/serve.py --profile gsq --context 393216 \
  --expert-cache 450 --vram-reserve-mib 768 --pool-workers 10
```

A later cold restart recovered all three markers from the same 385003-token prompt in 145.16 s. A separate Pi run started cold with 385344 input tokens and passed five warm continuations with at least 385343 cached tokens. Median decode was **66.2 tokens/s** and median full warm turn was **16.62 s**. Each continuation passed the Python code check and replayed reasoning. A separate continuation requested a 32768-token output limit and completed with token fitting. The minimum available **Windows system RAM** in the Pi run was 3046 MiB. The 768 MiB setting is a separate **GPU VRAM** reserve.

The automatic planner reached 589824 tokens at startup in one run. Full runs above 393216 on the reference machine were constrained by Windows RAM or GPU headroom. At 655360 and 1048576, the 512 MiB-reserve planner reported VRAM deficits of 686 and 6830 MiB with every expert in RAM. FP8 full-attention KV uses about 16 KiB per token on the GPU. At 1048576 tokens the full-attention KV alone is about 16 GiB, before the fixed model, drafter and workspace allocations.

## API and health smoke

Basalt listens only on `127.0.0.1:8191` **inside a private network namespace**. It has no outbound network access. A narrowly scoped Unix socket exposes its HTTP API:

```bash
curl --unix-socket "$INKLING_STATE/run/inference.sock" http://localhost/health
bash scripts/smoke.sh
# Or require an exact cap:
bash scripts/smoke.sh 270336
```

With no argument, `smoke.sh` reads and validates the served context from health, so it works with either profile. To require an exact cap, pass that number to `smoke.sh`. The smoke checks loaded state, model alias and context, then asks `17 * 23` and requires a final answer containing `391` with finish reason `stop`. It saves Git revision, exact command, request, health, response and expected/observed result under `$INKLING_STATE/smoke-<UTC timestamp>/`. This is a short inference check, not a long-context benchmark.

For tools that require TCP, opt into a host-side **loopback-only** bridge in a second terminal:

```bash
python3 scripts/relay.py --socket "$INKLING_STATE/run/inference.sock" --port 8191
# OpenAI-compatible base URL: http://127.0.0.1:8191/v1
# Model: inkling-2-mini
```

The bridge has no API authentication. Keep it bound to loopback; it intentionally exposes no `--host` option. Other local processes can access it. Ctrl-C stops the bridge; stop the model separately in its foreground terminal.

For Pi, merge the provider from [examples/pi-models.json](examples/pi-models.json) into your Pi `models.json`. Start the loopback bridge first, then use provider `inkling`, model `inkling-2-mini`. The example uses the shared default of 270336. Set `contextWindow` to the limit you actually serve, including 393216 for the larger GSQ configuration or your explicit context override. Preserve the assistant's `reasoning_content` in later OpenAI-compatible requests, including after tool calls. Pi's native OpenAI-compatible path replayed that field in the benchmark. Keep any agent tool sandbox on the Pi side of the bridge.

The sandbox exposes only read-only system libraries, CUDA, Basalt source/binary, venv, launcher scripts and the selected model, plus the WSL GPU device. It has a temporary HOME and temporary filesystem. The sole host-writable mount is its private `run/` bridge directory; personal homes and Windows drives are absent. GPU drivers and the host kernel remain shared, so this is process/filesystem/network isolation rather than a virtual machine security boundary.

## Reproducibility contract

| Component | Pin |
| --- | --- |
| Basalt merge base | `0f578db2e5f5601980f00dde09e0ba9e13c43718` |
| Cherry-picks (#750, #799) | `2ca45f57312922936e4f6f22e341636e8cfac407`, `19e0410ff6e40555546fdfa52c601ddb11870ba7`, `7a66425cff98d64324c14eebc4744a5a75487145`, `acaa96a6cb6a35b79f7e264bf2a480496814d2b9` |
| Final Basalt tree | `0026804d2191e2c55ff7cb4d9eddfa122f4ac511` |
| GSQ input revision | `3e3884f326a07e2df6d01122508c507cdf6f6dd1` |
| Official MX8 input revision | `c2afac3adb772e14d77b2ea2ac7068ad4460e932` |
| IncoAI DFlash 2 revision | `4f9088ea66c21f2b99ed657d1941c306b03e84d2` |

All download URLs, file sizes and full SHA256 values are in [manifests/inputs.json](manifests/inputs.json). [manifests/equivalence.json](manifests/equivalence.json) records benchmark tensor hashes, geometry and tokenizer hashes, and non-tensor section hashes. Geometry values are hashed as UTF-8 JSON with `separators=(',', ':')` and `ensure_ascii=True`. [manifests/target-config.json](manifests/target-config.json) preserves the source model config used to validate the DFlash converter's dimensions and layer visits; the final geometry is independently checked against the benchmark packs.

Reference exact packaged hashes:

| Pack | Bytes | SHA256 |
| --- | ---: | --- |
| GSQ IQ3_S + DFlash 2 | 13451914797 | `1fc2e87ab9fa12446319c1429d9edcda38c541ed0db064ec437b60247e6c680e` |
| MX8 + DFlash 2 | 19591671136 | `5c1f52101cd598c1578832f6c17c20a9c72a0a50ede0839b31d971369301907a` |

Reconstruction writes new packaging metadata and offsets. `pack.py` verifies all recorded tensor bytes, types, shapes, geometry, tokenizer settings and other sections and saves the generated file hashes in `pack-verification.json`. The reference native binary SHA256 was `c9240671880aed9f5c8074278285169e578b7c83a7fb93ca6a46f3f4158486cc`. Local binary hashes depend on the toolchain and build environment.

## Attribution and licenses

- [Thinking Machines Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini): base model, Apache-2.0.
- [ISTA-DASLab GSQ-RCO](https://huggingface.co/ISTA-DASLab/Inkling-2-Mini-GSQ-RCO-GGUF/tree/3e3884f326a07e2df6d01122508c507cdf6f6dd1): IQ3_S quantization, Apache-2.0 metadata.
- [Jesús Gulfo's Basalt](https://github.com/jesdga95/basalt): engine and converters, MIT; [official MX8 pack](https://huggingface.co/jesdga/Inkling-2-Mini-Basalt/tree/c2afac3adb772e14d77b2ea2ac7068ad4460e932), Apache-2.0.
- [IncoAI DFlash 2](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2/tree/4f9088ea66c21f2b99ed657d1941c306b03e84d2): speculative drafter, Apache-2.0 metadata.
- Basalt contributors to PR #750 and #799 supplied the pinned DFlash/mixed-IQ support used here.

Model-card license metadata was checked from the pinned Hugging Face revisions on 2026-11-27 and saved in `inputs.json`. Downloaded README and license files remain beside their source weights. Preserve these attributions and licenses when redistributing weights. This repository's original setup scripts are under [LICENSE](LICENSE); that license does not replace upstream model, CUDA or dependency licenses.

## Agent/operator rules

Use the documented scripts and explicit state path. Do not follow upstream `main`, silently change the pinned tree, reduce the integrity assertions, or replace DFlash with DSpark. Do not stop unrelated services, change WSL limits or start GPU tests without coordinating exclusive use. Do not commit `.state`, downloaded weights, credentials, personal paths, prompts or service logs. Keep runtime verification artifacts local unless reviewed for publication. Read the launch contract before requesting a context cap beyond the documented default.
