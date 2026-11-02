#!/usr/bin/env bash
# Build the pieces setup.sh otherwise compiles on the user's machine, for every supported CPU and GPU,
# into one archive that `PREBUILT=<archive> ./setup.sh` installs without a compiler or a CUDA toolkit:
#   kt_kernel-*.whl                kt-kernel with its six CPU variants (avx2, avx512 base / vnni / vbmi /
#                                  bf16, amx; picked at import) and CUDA archs 89;90;100;120
#   mmq/kt_gguf_moe_mmq_<arch>.so  the streamed prefill's MMQ torch extension, one file a GPU arch
#   lib/libkv_vmm.so, lib/librow_store.so
#   manifest.json                  what it was built from and against, and every file's SHA-256
# Run on a machine set up by setup.sh (the toolkit, ninja, the pinned torch). Takes ~20 minutes.
# usage: tools/prebuilt-pack.sh [out dir]   (default: build/prebuilt-<git describe>)
set -Eeuo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
py="$here/.venv/bin/python"
[[ -x "$py" ]] || { echo "run ./setup.sh first (.venv)" >&2; exit 2; }
tag="$(git -C "$here" describe --tags --always 2>/dev/null || echo untagged)"
out="${1:-$here/build/prebuilt-$tag}"
[[ -e "$out" ]] && { echo "$out exists; remove it or give another directory" >&2; exit 2; }
mkdir -p "$out/mmq" "$out/lib"
archs="${PREBUILT_GPU_ARCHS:-89 90a 100a 120a}"
export PATH="$CUDA_HOME/bin:$here/.venv/bin:$PATH"

echo "== kt-kernel, all CPU variants"
if [[ -n "${PREBUILT_KT_WHEEL:-}" ]]; then
  # a wheel built earlier with CPUINFER_BUILD_ALL_VARIANTS=1 (the build takes ~16 minutes)
  cp "$PREBUILT_KT_WHEEL" "$out/"
else
  env CPUINFER_BUILD_ALL_VARIANTS=1 CPUINFER_CUDA_ARCHS="${PREBUILT_KT_CUDA_ARCHS:-89;90;100;120}" CPUINFER_USE_CUDA=1 \
    "$py" -m pip wheel --no-deps --no-build-isolation -w "$out" third_party/ktransformers/kt-kernel
fi
whl="$(ls "$out"/kt_kernel-*.whl)"

echo "== MMQ extension, one build an arch: $archs"
for a in $archs; do
  TORCH_EXTENSIONS_DIR="$out/.ext" MAX_JOBS="${MAX_JOBS:-16}" KT_GGUF_MOE_MMQ_ARCH="$a" \
  PYTHONPATH="$here/third_party/sglang/python" \
    "$py" third_party/sglang/python/sglang/srt/layers/moe/kt_gguf_moe_mmq.py
  cp "$out/.ext/kt_gguf_moe_mmq_$a/kt_gguf_moe_mmq_$a.so" "$out/mmq/"
done
rm -rf "$out/.ext"

echo "== hwloc (kt-kernel links libhwloc.so.15; BSD-3, shipped beside the wheel)"
# PREBUILT_HWLOC=<path> names it; else the one the venv's kt-kernel loads (LD_LIBRARY_PATH / the system)
hw="${PREBUILT_HWLOC:-}"
if [[ -z "$hw" ]]; then
  for ext in $(find "$here/.venv" -path "*site-packages/kt_kernel/*" -name "*.so" 2>/dev/null); do
    hw="$(ldd "$ext" 2>/dev/null | awk '/libhwloc.so.15 => \// {print $3; exit}')"
    [[ -n "$hw" ]] && break
  done
fi
[[ -n "$hw" ]] || hw="$(ldconfig -p 2>/dev/null | awk '/libhwloc.so.15 / {print $NF; exit}')"
[[ -f "$hw" ]] || { echo "libhwloc.so.15 not found (PREBUILT_HWLOC=<path>)" >&2; exit 2; }
cp "$hw" "$out/lib/libhwloc.so.15"

echo "== kv-vmm, engram row store"
gcc -O2 -Wall -shared -fPIC kv-vmm/kv_vmm.c -I"$CUDA_HOME/include" -L"$CUDA_HOME/lib64/stubs" -lcuda -o "$out/lib/libkv_vmm.so"
g++ -O2 -std=c++17 -shared -fPIC -pthread engram-adapter/row_store.cpp -o "$out/lib/librow_store.so"

echo "== manifest"
"$py" - "$out" "$tag" "$(basename "$whl")" <<'EOF'
import hashlib, json, os, platform, subprocess, sys, torch
out, tag, whl = sys.argv[1:4]
files = {}
for root, _, names in os.walk(out):
    for n in sorted(names):
        p = os.path.join(root, n)
        files[os.path.relpath(p, out)] = hashlib.sha256(open(p, "rb").read()).hexdigest()
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "__file__" in dir() else os.getcwd()
pin = {}
for k, v in (("sglang", "third_party/sglang"), ("ktransformers", "third_party/ktransformers")):
    try:
        pin[k] = subprocess.check_output(["git", "-C", v, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        pin[k] = None
json.dump({
    "recipe": tag, "pins": pin, "wheel": whl,
    "torch": torch.__version__, "cuda": torch.version.cuda, "python": platform.python_version(),
    "glibc": platform.libc_ver()[1], "cpu_variants": ["avx2", "avx512", "avx512_vnni", "avx512_vbmi", "avx512_bf16", "amx"],
    "gpu_archs": sorted(f[len("mmq/kt_gguf_moe_mmq_"):-3] for f in files if f.startswith("mmq/")),
    "runtime_packages": ["libnuma1", "libgomp1"],
    "files": files,
}, open(os.path.join(out, "manifest.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in json.load(open(os.path.join(out, "manifest.json"))).items() if k != "files"}, indent=1))
EOF
tarball="$out.tar.gz"
tar -C "$(dirname "$out")" -czf "$tarball" "$(basename "$out")"
sha256sum "$tarball" | tee "$tarball.sha256"
echo "packed $tarball"
