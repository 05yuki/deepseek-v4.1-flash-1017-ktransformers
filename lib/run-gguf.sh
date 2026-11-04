#!/usr/bin/env bash
# From nothing to a running server in one command: build what serve.sh needs (setup.sh), download a GGUF build
# of the pack and the Engram file when they are not there, and start the server (serve.sh).
# A later run skips what is done and only starts the server.
#
#   BUILD=IQ2_XXS ./run.sh      a build of the pack (IQ1_M, IQ2_XXS or IQ2_XS) into ./models/<build>, then serve
#   BUILD=... ./run.sh --background   the same, the server in the background (its log under logs/)
#   GGUF=<dir> ./run.sh         a build already downloaded (either layout of the pack)
#
# run.sh runs this when BUILD or GGUF is set; without either it serves the MXFP4 checkpoint.
# The GGUF directory and the build are remembered in state/run-gguf.env.
# Downloads use plain HTTP (HF_HUB_DISABLE_XET=1): Xet stalled near the end of 40 GB shards in testing.
# HF_HUB_DISABLE_XET=0 uses Xet. Everything setup.sh and serve.sh read (CUDA_HOME, KT_*, SGLANG_*) passes through.
set -Eeuo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"

mode="${1:---foreground}"
if [[ "$mode" != "--foreground" && "$mode" != "--background" ]]; then
  echo "usage: [BUILD=IQ1_M|IQ2_XXS|IQ2_XS] [GGUF=<dir>] ./run.sh [--foreground|--background]" >&2
  exit 2
fi

REPO=ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF
REVISION=a419d74cd28d5b9f8c5e2b9077d11a609c17bc32  # the README's (llama.cpp's layout, 2026-10-27)
ENGRAM=DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf
step() { printf '\n== %s\n' "$*"; }

# what the last run used, unless the caller names another
remembered_gguf="" remembered_build=""
if [[ -f state/run-gguf.env ]]; then
  remembered_gguf="$(sed -n 's/^GGUF=//p' state/run-gguf.env)"
  remembered_build="$(sed -n 's/^BUILD=//p' state/run-gguf.env)"
fi
BUILD="${BUILD:-${remembered_build:-IQ2_XXS}}"
case "$BUILD" in
  IQ1_M) need_gb=121 ;;
  IQ2_XXS) need_gb=138 ;;
  IQ2_XS) need_gb=155 ;;
  *) echo "BUILD must be IQ1_M, IQ2_XXS or IQ2_XS (got $BUILD)" >&2; exit 2 ;;
esac
if [[ -z "${GGUF:-}" && -n "$remembered_gguf" && "$BUILD" == "$remembered_build" ]]; then
  GGUF="$remembered_gguf"
fi
GGUF="${GGUF:-$here/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO-$BUILD}"

command -v nvidia-smi >/dev/null || { echo "no nvidia-smi: this recipe needs an NVIDIA GPU and its driver" >&2; exit 3; }

step "the stack (setup.sh; the finished steps are skipped)"
env -u GGUF ./setup.sh

# a build is there when its first shard (either layout) and the Engram file are; an interrupted run.sh download
# leaves its .cache, and hf download picks it up where it stopped
have_build() {
  compgen -G "$GGUF/*-$BUILD-00001-of-0000[46].gguf" >/dev/null && [[ -f "$GGUF/$ENGRAM" ]] \
    && [[ ! -d "$GGUF/.cache/huggingface/download" || -z "$(find "$GGUF/.cache/huggingface/download" -name '*.incomplete' -print -quit)" ]]
}
if ! have_build; then
  step "download: $REPO, the $BUILD build and the Engram file, into $GGUF"
  mkdir -p "$GGUF"
  [[ -f "$GGUF/$ENGRAM" ]] || need_gb=$((need_gb + 111))
  free_gb=$(( $(df --output=avail -B1 "$GGUF" | tail -1) / 1000000000 ))
  if (( free_gb < need_gb + 5 )); then
    echo "$GGUF has $free_gb GB free; the download needs about $need_gb GB" >&2
    exit 3
  fi
  HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" .venv/bin/hf download "$REPO" --revision "$REVISION" \
    --include "*-$BUILD-0*.gguf" --include "$ENGRAM" --local-dir "$GGUF"
  have_build || { echo "the download did not leave the $BUILD shards and $ENGRAM in $GGUF" >&2; exit 3; }
fi

step "the build's state (setup.sh with GGUF=$GGUF)"
GGUF="$GGUF" ./setup.sh

mkdir -p state
printf 'GGUF=%s\nBUILD=%s\n' "$GGUF" "$BUILD" > state/run-gguf.env

# the host settings the README asks for, as root; the server runs without them, more slowly or less steadily
if [[ "$(cat /proc/sys/vm/min_free_kbytes 2>/dev/null || echo 0)" -lt 8388608 ]]; then
  echo "note: vm.min_free_kbytes is below 8388608 (sudo sysctl -w vm.min_free_kbytes=8388608)" >&2
fi
if [[ -n "$(swapon --noheadings 2>/dev/null)" ]]; then
  echo "note: swap is on (sudo swapoff -a)" >&2
fi

step "serve.sh $mode (http://${SGLANG_HOST:-127.0.0.1}:${SGLANG_PORT:-8080}, OpenAI-compatible)"
export GGUF
exec ./serve.sh "$mode"
