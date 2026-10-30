#!/usr/bin/env bash
# llama.cpp's MMQ sources for the tier-1 stream-mmq extension (kt_gguf_moe_mmq): 33 files of llama.cpp at
# LLAMA_COMMIT, fetched from GitHub into a clone of their own, patches/llama-mmq.patch on them (15 changed files and
# the IQ1_M instance), and every file checked against patches/llama-mmq.sha256, the bytes the extension was built
# and measured with. The patched SGLang tree carries only this repository's own MMQ code.
#
#   lib/llama-mmq.sh <patched SGLang tree> [<llama.cpp clone, default third_party/llama.cpp-mmq>]
set -Eeuo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tree="$(cd "$1" && pwd)"
clone="${2:-$here/third_party/llama.cpp-mmq}"
LLAMA_COMMIT=23b0202a189c44a54625aadcb37a946dd1d6278d
dest="$tree/python/sglang/srt/layers/moe/kt_gguf_moe_mmq/llama_mmq"
files=(
  ggml/include/ggml-alloc.h ggml/include/ggml-backend.h ggml/include/ggml-cuda.h ggml/include/ggml.h
  ggml/include/gguf.h ggml/src/ggml-common.h ggml/src/ggml-impl.h
  ggml/src/ggml-cuda/common.cuh ggml/src/ggml-cuda/mma.cuh ggml/src/ggml-cuda/mmid.cu ggml/src/ggml-cuda/mmid.cuh
  ggml/src/ggml-cuda/mmq.cuh ggml/src/ggml-cuda/mmq-load-tiles.cuh ggml/src/ggml-cuda/mmq-vec-dot.cuh
  ggml/src/ggml-cuda/mmq-config-ampere.cuh ggml/src/ggml-cuda/mmq-config-blackwell.cuh
  ggml/src/ggml-cuda/mmq-config-cdna.cuh ggml/src/ggml-cuda/mmq-config-gcn.cuh
  ggml/src/ggml-cuda/mmq-config-pascal-dp4a.cuh ggml/src/ggml-cuda/mmq-config-pascal-older.cuh
  ggml/src/ggml-cuda/mmq-config-rdna2.cuh ggml/src/ggml-cuda/mmq-config-rdna3-5.cuh
  ggml/src/ggml-cuda/mmq-config-rdna3.cuh ggml/src/ggml-cuda/mmq-config-rdna4.cuh
  ggml/src/ggml-cuda/vecdotq.cuh ggml/src/ggml-cuda/vendors/cuda.h
  ggml/src/ggml-cuda/template-instances/mmq-instance-iq1_s.cu
  ggml/src/ggml-cuda/template-instances/mmq-instance-iq2_xs.cu
  ggml/src/ggml-cuda/template-instances/mmq-instance-iq2_xxs.cu
  ggml/src/ggml-cuda/template-instances/mmq-instance-mxfp4.cu
  ggml/src/ggml-cuda/template-instances/mmq-instance-q2_0.cu
  ggml/src/ggml-cuda/template-instances/mmq-instance-q5_k.cu
  ggml/src/ggml-cuda/template-instances/mmq-instance-q8_0.cu
)
[[ -d "$dest" ]] || { echo "$dest missing: apply the stream-mmq patch first" >&2; exit 4; }

# only the commit, and of it only the blobs these files need
if [[ ! -d "$clone/.git" ]]; then
  git init -q "$clone"
  git -C "$clone" remote add origin https://github.com/ggml-org/llama.cpp.git
fi
if ! git -C "$clone" cat-file -e "$LLAMA_COMMIT^{commit}" 2>/dev/null; then
  git -C "$clone" fetch -q --depth 1 --filter=blob:none origin "$LLAMA_COMMIT"
fi
git -C "$clone" checkout -q --force "$LLAMA_COMMIT" -- LICENSE "${files[@]}"

rm -rf "$dest/ggml"
for f in "${files[@]}"; do
  mkdir -p "$dest/$(dirname "$f")"
  cp "$clone/$f" "$dest/$f"
done
cp "$clone/LICENSE" "$dest/ggml/LICENSE"
echo "$LLAMA_COMMIT" > "$dest/ggml/LLAMA_CPP_COMMIT"
git -C "$tree" apply --whitespace=nowarn "$here/patches/llama-mmq.patch"
(cd "$dest" && sha256sum --quiet -c "$here/patches/llama-mmq.sha256") || {
  echo "llama.cpp's MMQ sources differ from patches/llama-mmq.sha256" >&2
  exit 4
}
