#!/usr/bin/env bash
source "$(dirname -- "$0")/common.sh"
src="$STATE/basalt"
if [[ ! -e $src ]]; then
    git clone --no-checkout https://github.com/jesdga95/basalt.git "$src"
    git -C "$src" fetch https://github.com/ahostbr/basalt.git acaa96a6cb6a35b79f7e264bf2a480496814d2b9
    git -C "$src" checkout --detach 0f578db2e5f5601980f00dde09e0ba9e13c43718
    git -C "$src" -c user.name=Reproducible-build -c user.email=build@localhost cherry-pick \
        2ca45f57312922936e4f6f22e341636e8cfac407 19e0410ff6e40555546fdfa52c601ddb11870ba7 \
        7a66425cff98d64324c14eebc4744a5a75487145 acaa96a6cb6a35b79f7e264bf2a480496814d2b9
fi
[[ $(git -C "$src" rev-parse 'HEAD^{tree}') == 0026804d2191e2c55ff7cb4d9eddfa122f4ac511 ]]
[[ -z $(git -C "$src" status --porcelain --untracked-files=no) ]]
"$CUDA_HOME/bin/nvcc" --version | grep -q 'release 13.0'
[[ $(g++-13 -dumpversion) == 13 ]]
export PATH="$CUDA_HOME/bin:$PATH"
cmake -S "$src" -B "$STATE/build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DBASALT_MODELS=inkling2 -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
    -DCUDAToolkit_ROOT="$CUDA_HOME" -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13
cmake --build "$STATE/build" --target basalt -j "${BUILD_JOBS:-2}"
(cd "$src" && sha256sum bin/basalt) > "$STATE/binary.sha256"
python3 -m venv "$STATE/venv"
"$STATE/venv/bin/pip" install -r "$REPO/requirements.txt"
