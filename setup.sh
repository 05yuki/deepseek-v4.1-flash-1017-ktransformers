#!/usr/bin/env bash
# Build everything serve.sh needs, inside this directory:
#   third_party/sglang         sgl-project/sglang dsv4.1 @ 1aa0e962 + patches/sglang-dsv41-tree.patch
#   third_party/ktransformers  kvcache-ai/ktransformers @ 95009ea + patches/kt-kernel-tree.patch
#                              (+ patches/kt-kernel-llama-cpp.patch in its llama.cpp submodule)
#   .venv                      Python 3.12, requirements.txt, kt-kernel built for this CPU and GPU
#   engram-adapter/librow_store.so, build/libkv_vmm.so
#   state/engram-<model>/engram-manifest.json   when MODEL=<checkpoint dir> is given
#
# usage: [MODEL=<checkpoint dir>] ./setup.sh
#
# Needs: git, a C/C++ compiler, the CUDA toolkit (CUDA_HOME, default /usr/local/cuda;
# built and run here with 13.1), libnuma and hwloc headers (Debian/Ubuntu: libnuma-dev
# libhwloc-dev), and either uv or python3.12. Re-running skips the finished steps.
# Knobs: CUDA_HOME; KT_CUDA_ARCHS (default: the visible GPUs' compute capability);
# CPUINFER_CPU_INSTRUCT (kt-kernel's CPU target, default NATIVE: this machine's ISA);
# CMAKE_ARGS (extra CMake flags for kt-kernel, space separated); headers and libraries
# outside the system paths go in through CPLUS_INCLUDE_PATH / LIBRARY_PATH.
set -Eeuo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"
SGLANG_COMMIT=1aa0e962b206102b7c439a4a0c4981cfec6e87bc
SGLANG_TAG=dsv41-1aa0e962  # the same commit, tagged in 05yuki/sglang-upstream
KT_COMMIT=95009ea6856c0799e517e93cb12be5e8494bc7ce
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
mkdir -p third_party build state
step() { printf '\n== %s\n' "$*"; }
# a patch is checked against the pinned tree first; nothing is applied when it does not fit
apply_checked() {
  local tree="$1" patch="$2"
  if ! git -C "$tree" apply --check --whitespace=nowarn "$patch"; then
    echo "$patch does not apply to $tree; the tree was left as checked out" >&2
    exit 4
  fi
  git -C "$tree" apply --whitespace=nowarn "$patch"
}

step "SGLang @ ${SGLANG_COMMIT:0:8}"
if [[ ! -d third_party/sglang/.git ]]; then
  git clone --filter=blob:none https://github.com/sgl-project/sglang.git third_party/sglang
fi
if [[ ! -f third_party/sglang/.dsv41-1017-patched ]]; then
  # the dsv4.1 branch this commit came from is gone from sgl-project/sglang; the commit is
  # still fetchable there, and tagged in a fork as a fallback
  git -C third_party/sglang fetch --quiet origin "$SGLANG_COMMIT" 2>/dev/null \
    || git -C third_party/sglang fetch --quiet https://github.com/05yuki/sglang-upstream.git "refs/tags/$SGLANG_TAG"
  git -C third_party/sglang checkout --quiet --force "$SGLANG_COMMIT"
  git -C third_party/sglang clean -fdq
  apply_checked third_party/sglang "$here/patches/sglang-dsv41-tree.patch"
  touch third_party/sglang/.dsv41-1017-patched
fi

step "KTransformers @ ${KT_COMMIT:0:8}"
if [[ ! -d third_party/ktransformers/.git ]]; then
  git clone --filter=blob:none https://github.com/kvcache-ai/ktransformers.git third_party/ktransformers
fi
if [[ ! -f third_party/ktransformers/.dsv41-1017-patched ]]; then
  git -C third_party/ktransformers checkout --quiet --force "$KT_COMMIT"
  git -C third_party/ktransformers clean -fdq
  git -C third_party/ktransformers submodule update --init --force --quiet third_party/llama.cpp third_party/pybind11
  git -C third_party/ktransformers/third_party/llama.cpp clean -fdq
  apply_checked third_party/ktransformers "$here/patches/kt-kernel-tree.patch"
  apply_checked third_party/ktransformers/third_party/llama.cpp "$here/patches/kt-kernel-llama-cpp.patch"
  touch third_party/ktransformers/.dsv41-1017-patched
fi

step "Python environment (.venv)"
if [[ ! -x .venv/bin/python ]]; then
  if command -v uv >/dev/null; then
    uv venv --python 3.12 --seed .venv
  else
    python3.12 -m venv .venv
  fi
fi
py="$here/.venv/bin/python"
pip_install() {
  if command -v uv >/dev/null; then
    uv pip install --python "$py" "$@"
  else
    "$py" -m pip install "$@"
  fi
}
if [[ ! -f .venv/.requirements-done ]] || [[ requirements.txt -nt .venv/.requirements-done ]]; then
  # torch and its CUDA 13 wheels come from PyTorch's index, the rest from PyPI
  if command -v uv >/dev/null; then
    pip_install --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu130 -r requirements.txt
  else
    pip_install --extra-index-url https://download.pytorch.org/whl/cu130 -r requirements.txt
  fi
  touch .venv/.requirements-done
fi
# the patched SGLang tree (serve.sh also puts it first on PYTHONPATH)
if ! "$py" -c "import sglang, sys; sys.exit(0 if sglang.__file__.startswith('$here/third_party/sglang') else 1)" 2>/dev/null; then
  # the Rust extensions (router, gRPC) are not used here and would need cargo
  SGLANG_BUILD_RUST_EXTS=none pip_install --no-deps -e third_party/sglang/python
fi

step "kt-kernel (CPU experts)"
site="$("$py" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
if [[ ! -f build/.kt-kernel-done ]]; then
  if [[ -z "${KT_CUDA_ARCHS:-}" ]] && command -v nvidia-smi >/dev/null; then
    KT_CUDA_ARCHS="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d . | sort -u | paste -sd';')"
  fi
  export CPUINFER_CUDA_ARCHS="${KT_CUDA_ARCHS:-80;86;89;90;120}"
  export CPUINFER_CPU_INSTRUCT="${CPUINFER_CPU_INSTRUCT:-NATIVE}"
  # the CUDA side (stream memops, the streamed prefill's helpers)
  export CPUINFER_USE_CUDA="${CPUINFER_USE_CUDA:-1}"
  export PATH="$CUDA_HOME/bin:$here/.venv/bin:$PATH"
  echo "CUDA archs $CPUINFER_CUDA_ARCHS, CPU target $CPUINFER_CPU_INSTRUCT"
  pip_install --no-deps --no-build-isolation third_party/ktransformers/kt-kernel
  # each expert layer is read into the page cache in parallel before kt-kernel copies it
  "$py" patches/kt-kernel-load-prefetch.py "$site"
  touch build/.kt-kernel-done
fi

step "engram row store (engram-adapter/librow_store.so)"
if [[ ! -f engram-adapter/librow_store.so || engram-adapter/row_store.cpp -nt engram-adapter/librow_store.so ]]; then
  g++ -O2 -std=c++17 -shared -fPIC -pthread engram-adapter/row_store.cpp -o engram-adapter/librow_store.so
fi

step "kv-vmm (build/libkv_vmm.so)"
if [[ ! -f build/libkv_vmm.so || kv-vmm/kv_vmm.c -nt build/libkv_vmm.so ]]; then
  gcc -O2 -Wall -shared -fPIC kv-vmm/kv_vmm.c -I"$CUDA_HOME/include" -L"$CUDA_HOME/lib64/stubs" -lcuda -o build/libkv_vmm.so
fi
# the KV pool module looks for the library beside itself first
cp build/libkv_vmm.so third_party/sglang/python/sglang/srt/layers/moe/libkv_vmm.so

if [[ -n "${MODEL:-}" ]]; then
  step "engram manifest for $MODEL"
  "$py" tools/make-engram-manifest.py "$MODEL" "state/engram-$(basename "$MODEL")"
fi

step "check: the attention-output FP8 kernel on this GPU"
"$py" tools/dsv4-o-fp8-qdq-test.py ${MODEL:+"$MODEL"} 2>&1 | grep -v -i warn | tail -3

printf '\nDone. Start the server with: MODEL=<checkpoint dir> ./serve.sh\n'
