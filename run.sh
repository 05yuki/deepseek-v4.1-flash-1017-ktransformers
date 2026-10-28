#!/usr/bin/env bash
# From nothing to a running server in one command: build what serve.sh needs (setup.sh), download the
# DeepSeek-V4.1-Flash-1017 checkpoint when it is not there, and start the server (serve.sh).
# A later run skips what is done and only starts the server.
#
#   ./run.sh                    the checkpoint into ./models/DeepSeek-V4.1-Flash-1017, then serve (Ctrl-C stops it)
#   ./run.sh --background       the same, the server in the background (its log under logs/)
#   MODEL=<dir> ./run.sh        a checkpoint already downloaded
#
# The checkpoint directory is remembered in state/run.env: a plain ./run.sh starts the same one again.
# Downloads use plain HTTP (HF_HUB_DISABLE_XET=1): Xet stalled near the end of large files in testing.
# HF_HUB_DISABLE_XET=0 uses Xet. Everything setup.sh and serve.sh read (CUDA_HOME, TIER, KT_*, SGLANG_*) passes through.
set -Eeuo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"

mode="${1:---foreground}"
if [[ "$mode" != "--foreground" && "$mode" != "--background" ]]; then
  echo "usage: [MODEL=<dir>] $0 [--foreground|--background]" >&2
  exit 2
fi

REPO=deepseek-ai/DeepSeek-V4.1-Flash-1017
NEED_GB=510  # the checkpoint
step() { printf '\n== %s\n' "$*"; }

if [[ -z "${MODEL:-}" && -f state/run.env ]]; then
  MODEL="$(sed -n 's/^MODEL=//p' state/run.env)"
fi
MODEL="${MODEL:-$here/models/DeepSeek-V4.1-Flash-1017}"

command -v nvidia-smi >/dev/null || { echo "no nvidia-smi: this recipe needs an NVIDIA GPU and its driver" >&2; exit 3; }
# the server holds about 420 GiB (README, What it needs); it does not start with much less
ram_gib=$(( $(sed -n 's/^MemTotal: *\([0-9]*\) kB/\1/p' /proc/meminfo) / 1048576 ))
if (( ram_gib < 430 )); then
  echo "note: $ram_gib GiB of RAM; the server holds about 420 GiB and needs a 512 GB class machine" >&2
  echo "      (the GGUF recipe runs this model in 64 GB: github.com/05yuki/deepseek-v4.1-flash-1017-gguf-ktransformers)" >&2
fi

step "the stack (setup.sh; the finished steps are skipped)"
env -u MODEL ./setup.sh

# the checkpoint is there when its config and every shard its index names are; an interrupted run.sh download
# leaves its .cache, and hf download picks it up where it stopped
have_model() {
  [[ -f "$MODEL/config.json" && -f "$MODEL/model.safetensors.index.json" ]] || return 1
  [[ ! -d "$MODEL/.cache/huggingface/download" || -z "$(find "$MODEL/.cache/huggingface/download" -name '*.incomplete' -print -quit)" ]] || return 1
  .venv/bin/python - "$MODEL" <<'PY'
import json, os, sys
d = sys.argv[1]
files = set(json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"].values())
sys.exit(0 if all(os.path.isfile(os.path.join(d, f)) for f in files) else 1)
PY
}
if ! have_model; then
  step "download: $REPO into $MODEL"
  mkdir -p "$MODEL"
  have_gb=$(( $(du -sb "$MODEL" | cut -f1) / 1000000000 ))
  free_gb=$(( $(df --output=avail -B1 "$MODEL" | tail -1) / 1000000000 ))
  if (( free_gb + have_gb < NEED_GB + 5 )); then
    echo "$MODEL has $free_gb GB free (and $have_gb GB already there); the checkpoint is about $NEED_GB GB" >&2
    exit 3
  fi
  HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" .venv/bin/hf download "$REPO" --local-dir "$MODEL"
  have_model || { echo "the download did not leave the whole checkpoint in $MODEL" >&2; exit 3; }
fi

step "the checkpoint's state (setup.sh with MODEL=$MODEL)"
MODEL="$MODEL" ./setup.sh

mkdir -p state
printf 'MODEL=%s\n' "$MODEL" > state/run.env

# the host settings the README asks for, as root; the server runs without them, more slowly or less steadily
if [[ "$(cat /proc/sys/vm/min_free_kbytes 2>/dev/null || echo 0)" -lt 8388608 ]]; then
  echo "note: vm.min_free_kbytes is below 8388608 (sudo sysctl -w vm.min_free_kbytes=8388608)" >&2
fi
if [[ -n "$(swapon --noheadings 2>/dev/null)" ]]; then
  echo "note: swap is on (sudo swapoff -a)" >&2
fi

step "serve.sh $mode (http://${SGLANG_HOST:-127.0.0.1}:${SGLANG_PORT:-8080}, OpenAI-compatible)"
export MODEL
exec ./serve.sh "$mode"
