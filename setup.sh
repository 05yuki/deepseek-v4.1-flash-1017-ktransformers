#!/usr/bin/env bash
# Build everything serve.sh needs, inside this directory:
#   third_party/sglang         sgl-project/sglang main @ 3edea93c + patches/sglang-tier*
#   third_party/ktransformers  kvcache-ai/ktransformers @ 95009ea (+ patches/kt-kernel-tree.patch
#                              and patches/kt-kernel-llama-cpp.patch from tier 1)
#   .venv                      Python 3.12, requirements.txt, kt-kernel built for this CPU and GPU
#   engram-adapter/librow_store.so, build/libkv_vmm.so
#   state/engram-<model>/engram-manifest.json   when MODEL=<checkpoint dir> is given
#
# usage: [MODEL=<checkpoint dir>] [TIER=0|1|2] [SGLANG_FEATURES=a,b,...] ./setup.sh
#
# TIER picks how much of this repository's SGLang changes go in, each tier on top of the ones
# below it (docs/patches.md):
#   0  run DeepSeek-V4.1-Flash-1017 correctly on the hybrid MoE, on any GPU; KTransformers as pinned
#   1  + speed and memory: streamed prefill, lends, hot experts, the CPU kernels
#   2  + all-reduce through host memory, for two GPUs without peer access (default)
# Changing TIER re-applies the patches and rebuilds kt-kernel.
# Tier 1 is one patch per feature (patches/sglang-tier1/, docs/tier1.md). SGLANG_FEATURES picks
# some of them, comma-separated (e.g. fp8-decode,engram-row-prefetch), or none; default all. A feature's
# prerequisites (patches/sglang-tier1/requires.txt) must be picked too.
# Needs: git, a C/C++ compiler, the CUDA toolkit (CUDA_HOME, default /usr/local/cuda;
# built and run here with 13.1; without a usable one, NVIDIA's 13.2 toolkit wheels are installed under build/
# and serve as CUDA_HOME through lib/cuda-home.sh, enough for the run-time JITs and the small
# builds but not for kt-kernel's CMake build), libnuma and hwloc headers (Debian/Ubuntu: libnuma-dev libhwloc-dev), and either
# uv or python3.12. Re-running skips the finished steps.
# PREBUILT=<archive or directory from tools/prebuilt-pack.sh>: installs its kt-kernel wheel (every CPU
# variant), MMQ extension (one file a GPU arch), kv-vmm, engram row store and hwloc instead of building
# them; no compiler or CUDA toolkit is needed then (the FlashInfer CUTLASS MoE of the MMQ-off path is
# not built either). Every file is checked against the archive's manifest.json.
# PREPARE_ONLY=1 stops after the trees, the venv and the toolkit choice (for tools/prebuilt-pack.sh).
# Knobs: CUDA_HOME; KT_CUDA_ARCHS (default: the visible GPUs' compute capability);
# CPUINFER_CPU_INSTRUCT (kt-kernel's CPU target, default NATIVE: this machine's ISA);
# CMAKE_ARGS (extra CMake flags for kt-kernel, space separated); headers and libraries
# outside the system paths go in through CPLUS_INCLUDE_PATH / LIBRARY_PATH.
set -Eeuo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"
SGLANG_COMMIT=3edea93cbc0d857026253df0535a4d89791032a3
KT_COMMIT=95009ea6856c0799e517e93cb12be5e8494bc7ce
TIER="${TIER:-2}"
[[ "$TIER" =~ ^[012]$ ]] || { echo "TIER must be 0, 1 or 2 (tiers 2 and 3 of before 10-21 are 1 and 2)" >&2; exit 2; }
SGLANG_FEATURES="${SGLANG_FEATURES:-all}"
# the SGLang patches up to TIER, in order (paths under patches/)
sglang_patches=()
for f in "$here"/patches/sglang-tier0/[0-9][0-9]-*.patch; do sglang_patches+=("sglang-tier0/$(basename "$f")"); done
features=()
if (( TIER >= 1 )); then
  for f in "$here"/patches/sglang-tier1/[0-9][0-9]-*.patch; do
    slug="$(basename "$f" .patch)"; slug="${slug#[0-9][0-9]-}"
    if [[ "$SGLANG_FEATURES" == all || ",$SGLANG_FEATURES," == *",$slug,"* ]]; then
      features+=("$slug")
      sglang_patches+=("sglang-tier1/$(basename "$f")")
    fi
  done
  for want in ${SGLANG_FEATURES//,/ }; do
    [[ "$want" == all || "$want" == none || " ${features[*]} " == *" $want "* ]] || { echo "no tier-1 feature named $want (patches/sglang-tier1/)" >&2; exit 2; }
  done
  while IFS=: read -r feat needs; do
    [[ -z "$feat" || "$feat" == \#* ]] && continue
    [[ " ${features[*]} " == *" $feat "* ]] || continue
    for n in $needs; do
      [[ " ${features[*]} " == *" $n "* ]] || { echo "tier-1 feature $feat needs $n (add it to SGLANG_FEATURES)" >&2; exit 2; }
    done
  done < "$here/patches/sglang-tier1/requires.txt"
fi
(( TIER >= 2 )) && sglang_patches+=(sglang-tier2-no-p2p.patch)
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
mkdir -p third_party build state
step() { printf '\n== %s\n' "$*"; }
# a step's marker file holds what it built: the tier or variant and a hash of the patches, so a
# changed patch is applied (and kt-kernel rebuilt) again
built_as() { [[ -f "$1" && "$(cat "$1")" == "$2" ]]; }
patch_hash() { (cd "$here/patches" && cat "$@") | sha256sum | cut -c1-12; }
sglang_state="$TIER $(patch_hash "${sglang_patches[@]}" llama-mmq.patch llama-mmq.sha256)"
kt_variant=pinned
kt_patches=(kt-kernel-tree.patch kt-kernel-llama-cpp.patch kt-kernel-load-prefetch.py)
if [[ " ${features[*]} " == *" cpu-gpu-overlap "* ]]; then
  kt_patches+=(kt-kernel-cpu-handoff.patch)
fi
if [[ " ${features[*]} " == *" expert-cache "* ]]; then
  kt_patches+=(kt-kernel-expert-cache.patch)
fi
(( TIER >= 1 )) && kt_variant="patched $(patch_hash "${kt_patches[@]}")"
# a patch is checked against the pinned tree first; nothing is applied when it does not fit
apply_checked() {
  local tree="$1" patch="$2"
  if ! git -C "$tree" apply --check --whitespace=nowarn "$patch"; then
    echo "$patch does not apply to $tree; the tree was left as checked out" >&2
    exit 4
  fi
  git -C "$tree" apply --whitespace=nowarn "$patch"
}

step "SGLang @ ${SGLANG_COMMIT:0:8}, tier $TIER${features[*]:+ (tier 1: ${features[*]})}"
if [[ ! -d third_party/sglang/.git ]]; then
  git clone --filter=blob:none https://github.com/sgl-project/sglang.git third_party/sglang
fi
if ! built_as third_party/sglang/.dsv41-1017-patched "$sglang_state"; then
  # a commit on sgl-project/sglang's main branch
  git -C third_party/sglang fetch --quiet origin "$SGLANG_COMMIT"
  git -C third_party/sglang checkout --quiet --force "$SGLANG_COMMIT"
  git -C third_party/sglang clean -fdq
  for p in "${sglang_patches[@]}"; do
    apply_checked third_party/sglang "$here/patches/$p"
  done
  # stream-mmq's llama.cpp sources: fetched, patched and checked (lib/llama-mmq.sh)
  if [[ " ${features[*]} " == *" stream-mmq "* ]]; then
    bash "$here/lib/llama-mmq.sh" third_party/sglang
  fi
  echo "$sglang_state" > third_party/sglang/.dsv41-1017-patched
  printf '%s\n' "${features[@]}" > third_party/sglang/.dsv41-1017-features
fi

step "KTransformers @ ${KT_COMMIT:0:8}, $kt_variant"
if [[ ! -d third_party/ktransformers/.git ]]; then
  git clone --filter=blob:none https://github.com/kvcache-ai/ktransformers.git third_party/ktransformers
fi
if ! built_as third_party/ktransformers/.dsv41-1017-patched "$kt_variant"; then
  git -C third_party/ktransformers checkout --quiet --force "$KT_COMMIT"
  git -C third_party/ktransformers clean -fdq
  git -C third_party/ktransformers submodule update --init --force --quiet third_party/llama.cpp third_party/pybind11
  git -C third_party/ktransformers/third_party/llama.cpp clean -fdq
  if [[ "$kt_variant" == patched* ]]; then
    apply_checked third_party/ktransformers "$here/patches/kt-kernel-tree.patch"
    if [[ " ${features[*]} " == *" cpu-gpu-overlap "* ]]; then
      apply_checked third_party/ktransformers "$here/patches/kt-kernel-cpu-handoff.patch"
    fi
    if [[ " ${features[*]} " == *" expert-cache "* ]]; then
      apply_checked third_party/ktransformers "$here/patches/kt-kernel-expert-cache.patch"
    fi
    apply_checked third_party/ktransformers/third_party/llama.cpp "$here/patches/kt-kernel-llama-cpp.patch"
  fi
  echo "$kt_variant" > third_party/ktransformers/.dsv41-1017-patched
fi

step "Python environment (.venv)"
if [[ ! -x .venv/bin/python ]]; then
  if command -v uv >/dev/null; then
    uv venv --python 3.12 --seed .venv
  elif command -v python3.12 >/dev/null; then
    python3.12 -m venv .venv
  else
    echo "neither uv nor python3.12 is on PATH: install uv (https://docs.astral.sh/uv/, it fetches Python 3.12) or Python 3.12" >&2
    exit 2
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
# The toolkit for the run-time JITs and the small builds. A system toolkit is used when it is CUDA 13
# (torch's cu130 build refuses another major) and its headers match this glibc (lib/cuda-home.sh,
# cuda_home_unusable). Otherwise NVIDIA's toolkit wheels (lib/cuda-wheels.txt, CUDA 13.2, 1.3 GB) go
# under build/ and serve as CUDA_HOME: nothing outside this directory changes. CUDA_WHEELS=0 forbids
# that download. kt-kernel's CMake build still wants a system toolkit (use PREBUILT= without one).
source "$here/lib/cuda-home.sh"
why="$(cuda_home_unusable "$CUDA_HOME")"
if [[ -n "$why" ]]; then
  [[ -x "$CUDA_HOME/bin/nvcc" ]] && echo "the system toolkit is not used: $why"
  if [[ "${CUDA_WHEELS:-1}" == 0 ]]; then
    echo "no usable CUDA 13 toolkit under CUDA_HOME ($CUDA_HOME: $why) and CUDA_WHEELS=0: install one (13.2 or later) or allow the wheels" >&2
    exit 2
  fi
  pip_cuda="$(cuda_home_from_wheels "$py" "$here/build")"
  if [[ -n "$pip_cuda" ]]; then
    export CUDA_HOME="$pip_cuda"
    echo "CUDA toolkit: NVIDIA's wheels, $("$pip_cuda/bin/nvcc" --version | tail -1 | sed 's/.*cuda_//; s/,.*//'), under build/cuda-home (lib/cuda-wheels.txt)"
  else
    echo "no usable CUDA 13 toolkit under CUDA_HOME and the toolkit wheels could not be installed (lib/cuda-wheels.txt)" >&2
    exit 2
  fi
fi
# the patched SGLang tree (serve.sh also puts it first on PYTHONPATH)
if ! "$py" -c "import sglang, sys; sys.exit(0 if sglang.__file__.startswith('$here/third_party/sglang') else 1)" 2>/dev/null; then
  # the Rust extensions (router, gRPC) are not used here and would need cargo
  SGLANG_BUILD_RUST_EXTS=none pip_install --no-deps -e third_party/sglang/python
fi

# PREPARE_ONLY=1: the trees, the venv and the toolkit choice, no build (tools/prebuilt-pack.sh builds
# from here; .github/workflows/prebuilt.yml)
if [[ "${PREPARE_ONLY:-0}" == 1 ]]; then
  echo; echo "PREPARE_ONLY=1: trees and venv ready, nothing built"; exit 0
fi

prebuilt=""
if [[ -n "${PREBUILT:-}" ]]; then
  step "prebuilt pieces from $PREBUILT"
  if [[ -d "$PREBUILT" ]]; then
    [[ "$(cd "$PREBUILT" && pwd)" == "$here/prebuilt" ]] || { rm -rf prebuilt; cp -a "$PREBUILT" prebuilt; }
  else
    rm -rf prebuilt build/prebuilt-unpack; mkdir -p build/prebuilt-unpack
    tar -xzf "$PREBUILT" -C build/prebuilt-unpack
    mv "$(ls -d build/prebuilt-unpack/*/ | head -1)" prebuilt; rmdir build/prebuilt-unpack
  fi
  "$py" - "$here/prebuilt" <<'PY'
import hashlib, json, os, platform, sys
d = sys.argv[1]; m = json.load(open(os.path.join(d, "manifest.json")))
bad = [f for f, h in m["files"].items() if hashlib.sha256(open(os.path.join(d, f), "rb").read()).hexdigest() != h]
if bad:
    sys.exit("prebuilt files do not match manifest.json: " + ", ".join(bad))
want = [l.split("==")[1].strip() for l in open("requirements.txt") if l.startswith("torch==")][0]
if m["torch"] != want:
    sys.exit(f"prebuilt is for torch {m['torch']}, requirements.txt pins {want}")
if platform.python_version().rsplit(".", 1)[0] != m["python"].rsplit(".", 1)[0]:
    sys.exit(f"prebuilt is for Python {m['python']}, this is {platform.python_version()}")
print(f"recipe {m['recipe']}, torch {m['torch']}, CUDA {m['cuda']}, GPU archs {' '.join(m['gpu_archs'])}, "
      f"{len(m['files'])} files verified")
PY
  prebuilt="$here/prebuilt"
fi

step "kt-kernel (CPU experts)"
site="$("$py" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
if [[ -n "$prebuilt" ]]; then
  kt_variant="prebuilt $(sha256sum "$prebuilt"/kt_kernel-*.whl | cut -c1-12)"
fi
if [[ -n "$prebuilt" ]] && ! built_as build/.kt-kernel-done "$kt_variant"; then
  echo "the archive's wheel: $(basename "$prebuilt"/kt_kernel-*.whl) (every CPU variant, picked at import)"
  pip_install --no-deps "$prebuilt"/kt_kernel-*.whl
  "$py" patches/kt-kernel-load-prefetch.py "$site"
  echo "$kt_variant" > build/.kt-kernel-done
elif ! built_as build/.kt-kernel-done "$kt_variant"; then
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
  [[ "$kt_variant" == patched* ]] && "$py" patches/kt-kernel-load-prefetch.py "$site"
  echo "$kt_variant" > build/.kt-kernel-done
fi

# The streamed prefill's CUTLASS MoE is a FlashInfer JIT module. Built inside the server it would
# compile on the first long prompt, while the expert arenas fill the NUMA nodes, and a node-local OOM
# kills the scheduler before the build finishes; so it is built here, once per FlashInfer version and
# GPU arch, into the cache serve.sh uses.
if (( TIER >= 1 )) && [[ " ${features[*]} " == *" stream-prefill "* ]] && [[ -z "$prebuilt" ]] && command -v nvidia-smi >/dev/null; then
  cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)"
  fi_backend="${cap/./}"
  case "$cap" in 9.0|10.0|10.3|12.0|12.1) fi_arch="${cap}a" ;; *) fi_arch="$cap" ;; esac
  step "FlashInfer CUTLASS MoE for sm $cap (streamed prefill)"
  FLASHINFER_WORKSPACE_BASE="${SGLANG_CACHE_DIR:-$here/.cache/sglang}" \
  FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-$fi_arch}" MAX_JOBS="${MAX_JOBS:-16}" \
  PATH="$CUDA_HOME/bin:$here/.venv/bin:$PATH" \
    "$py" -c "from flashinfer.fused_moe.core import get_cutlass_fused_moe_module as g; g('$fi_backend'); print('built (or cached)')"
fi

# The streamed prefill's MMQ (tier-1 stream-mmq): llama.cpp's kernels as a torch extension, built here once for
# the GPU's arch into the cache serve.sh uses, not inside the server on the first long prompt
mmq_arch=""
if [[ -n "$prebuilt" ]] && command -v nvidia-smi >/dev/null; then
  cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)"
  case "$cap" in 9.0|10.0|10.3|12.0|12.1) mmq_arch="${cap/./}a" ;; *) mmq_arch="${cap/./}" ;; esac
fi
if (( TIER >= 1 )) && [[ " ${features[*]} " == *" stream-mmq "* ]] && [[ -n "$mmq_arch" && -f "$prebuilt/mmq/kt_gguf_moe_mmq_$mmq_arch.so" ]]; then
  step "MMQ kernels for the streamed prefill: prebuilt/mmq/kt_gguf_moe_mmq_$mmq_arch.so"
elif (( TIER >= 1 )) && [[ " ${features[*]} " == *" stream-mmq "* ]] && command -v nvidia-smi >/dev/null; then
  [[ -n "$prebuilt" ]] && echo "the archive has no MMQ build for sm $cap ($mmq_arch); building it needs the CUDA toolkit"
  step "MMQ kernels for the streamed prefill (kt_gguf_moe_mmq)"
  TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$here/.cache/torch_extensions}" MAX_JOBS="${MAX_JOBS:-16}" \
  PATH="$CUDA_HOME/bin:$here/.venv/bin:$PATH" PYTHONPATH="$here/third_party/sglang/python" \
    "$py" -c "from sglang.srt.layers.moe import kt_gguf_moe_mmq as m; m._load(); print('built (or cached)')"
fi

step "engram row store (engram-adapter/librow_store.so)"
if [[ -n "$prebuilt" ]]; then
  cp "$prebuilt/lib/librow_store.so" engram-adapter/librow_store.so
elif [[ ! -f engram-adapter/librow_store.so || engram-adapter/row_store.cpp -nt engram-adapter/librow_store.so ]]; then
  g++ -O2 -std=c++17 -shared -fPIC -pthread engram-adapter/row_store.cpp -o engram-adapter/librow_store.so
fi

step "kv-vmm (build/libkv_vmm.so)"
if [[ -n "$prebuilt" ]]; then
  cp "$prebuilt/lib/libkv_vmm.so" build/libkv_vmm.so
elif [[ ! -f build/libkv_vmm.so || kv-vmm/kv_vmm.c -nt build/libkv_vmm.so ]]; then
  # the driver API: the toolkit's stub, or the driver's own library where there is no toolkit
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
