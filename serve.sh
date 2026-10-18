#!/usr/bin/env bash
# Serve DeepSeek-V4.1-Flash(-1017) with SGLang + KTransformers from what setup.sh built.
#
#   MODEL=<checkpoint dir> ./serve.sh [--foreground|--background]
#
# The routed experts run on the CPU (kt-kernel, MXFP4); attention, the hottest experts and
# the prefill (experts streamed over PCIe) run on the GPUs. The engram tables are read
# from the checkpoint shards on disk through engram-adapter/ (a small RAM cache).
#
# The machine is read at start: one tensor-parallel rank per visible GPU
# (CUDA_VISIBLE_DEVICES picks them), one CPU expert pool per NUMA node, all physical
# cores, and the GPUs' compute capability. Every default below can be overridden by
# the variable named next to it. Logs go to logs/<timestamp>/.
set -Eeuo pipefail
export LC_ALL=C PYTHONUNBUFFERED=1
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$here"  # read by lib/*.sh
mode="${1:---foreground}"
if [[ "$mode" != "--foreground" && "$mode" != "--background" ]]; then
  echo "Usage: MODEL=<checkpoint dir> $0 [--foreground|--background]" >&2
  exit 2
fi
model="${MODEL:-${DSV41_MODEL:-}}"
[[ -n "$model" && -f "$model/config.json" ]] || { echo "set MODEL to the checkpoint directory" >&2; exit 4; }
model="$(cd "$model" && pwd)"
venv="$here/.venv"
python="$venv/bin/python"
source_dir="$here/third_party/sglang/python"
adapter_dir="$here/engram-adapter"
engram_dir="${DSV41_ENGRAM_DIR:-$here/state/engram-$(basename "$model")}"
export KT_KV_VMM_LIB="${KT_KV_VMM_LIB:-$here/build/libkv_vmm.so}"
export KT_LEND_PY="$source_dir/sglang/srt/layers/moe/kt_lend.py"
port="${SGLANG_PORT:-8080}"
log_root="$here/logs"
alias_name="${SGLANG_SERVED_NAME:-$(basename "$model")}"
[[ -x "$python" ]] || { echo "run ./setup.sh first (.venv missing)" >&2; exit 3; }
[[ -f "$source_dir/sglang/srt/layers/engram.py" ]] || { echo "run ./setup.sh first (patched SGLang missing)" >&2; exit 3; }
[[ -f "$adapter_dir/librow_store.so" ]] || { echo "run ./setup.sh first (engram-adapter/librow_store.so missing)" >&2; exit 3; }
if [[ ! -f "$engram_dir/engram-manifest.json" ]]; then
  "$python" "$here/tools/make-engram-manifest.py" "$model" "$engram_dir" >&2
fi

# --- the machine ------------------------------------------------------------
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found" >&2; exit 3; }
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -ra gpus <<<"$CUDA_VISIBLE_DEVICES"
else
  mapfile -t gpus < <(nvidia-smi --query-gpu=index --format=csv,noheader)
fi
n_gpus="${#gpus[@]}"
tp="${KT_TP:-$n_gpus}"
cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i "${gpus[0]}" | head -1)"
n_nodes="$(ls -d /sys/devices/system/node/node[0-9]* 2>/dev/null | wc -l)"
(( n_nodes > 0 )) || n_nodes=1
phys_cores="$(lscpu -p=CORE,SOCKET 2>/dev/null | grep -v '^#' | sort -u | wc -l)"
(( phys_cores > 0 )) || phys_cores="$(nproc)"
# rank r runs on node r * nodes / tp (two GPUs on two sockets: 0 1; one GPU: 0)
numa_list=""
for ((r = 0; r < tp; r++)); do numa_list+="$(( r * n_nodes / tp )) "; done

# --- model and memory -------------------------------------------------------
context_length="${SGLANG_CONTEXT_LENGTH:-1048576}"
max_total_tokens="${SGLANG_MAX_TOTAL_TOKENS:-$context_length}"
# experts kept on each GPU per layer, the hottest by a routing recording
# (placements/); 5 leaves a 16 GB card room for the 1M-token KV pool
gpu_experts="${KT_GPU_EXPERTS:-5}"
cpu_threads="${KT_CPU_THREADS:-$phys_cores}"
threadpools="${KT_THREADPOOL_COUNT:-$n_nodes}"
deferred_experts="${KT_DEFERRED_EXPERTS:-0}"  # >0 drops expert work for speed; it hurts the text
mem_fraction="${SGLANG_MEM_FRACTION:-0.81}"
kv_cache_dtype="${SGLANG_KV_CACHE_DTYPE:-fp8_e4m3}"
chunked_prefill="${SGLANG_CHUNKED_PREFILL_SIZE:-2048}"
swa_prefix_tails="${SGLANG_SWA_PREFIX_TAILS:-6}"
extra_args=(${SGLANG_EXTRA_ARGS:-})
radix_args=(--disable-radix-cache)
[[ "${DSV41_RADIX:-0}" == "1" ]] && radix_args=()

# DeepSeek-V4.1-Flash-1017 rounds the attention output to FP8 before wo_a; the block
# comes from the checkpoint's own inference code (off for the earlier release)
source "$here/tools/o-fp8-qdq-block.sh"
# wo_a stays in the checkpoint's FP8 (32x32 E8M0 blocks) instead of a bf16 copy (0: bf16)
export SGLANG_DSV4_WO_A_FP8_NATIVE="${SGLANG_DSV4_WO_A_FP8_NATIVE:-1}"
# the vision tower waits in pinned host memory and is on the GPU only while a batch's
# images are encoded; the prefill plan gives a request with images that much less room
# (0: resident)
export SGLANG_DSV4_VISION_ON_DEMAND="${SGLANG_DSV4_VISION_ON_DEMAND:-1}"
# the Marlin MoE steps around its two GEMMs (SwiGLU, routing weights, top-k sum) in fp32
# with one rounding each (0: the bf16 steps)
export SGLANG_MARLIN_MOE_FP32_GLUE="${SGLANG_MARLIN_MOE_FP32_GLUE:-1}"
# the GPU experts (resident and hot) without Marlin: the checkpoint's MXFP4 layout through
# Triton kernels that round once, at the weighted top-k sum (0: Marlin, with the glue above)
export SGLANG_MXFP4_MOE_FP32="${SGLANG_MXFP4_MOE_FP32:-1}"

# hot experts: the resident ones, and with the hot lend 16 more a layer in the KV
# pool's free pages outside prefill (KT_KV_HOT_LEND=0 turns it off)
source "$here/lib/kt-kv-hot-default.sh"
kt_kv_hot_default "$here/placements/dsv41-expert-placement-hot64.json" 5 "$gpu_experts" 16 "$source_dir" swa
# in decode the hot lend's slots also take the driver's idle memory above a 1 GiB floor
# (-1: only the KV pool's free pages)
export KT_KV_HOT_SPARE_FLOOR_MB="${KT_KV_HOT_SPARE_FLOOR_MB:-1024}"
placement="${KT_EXPERT_PLACEMENT:-${KT_KV_HOT_PLACEMENT:-$here/placements/dsv41-expert-placement-${gpu_experts}.json}}"
if [[ "$placement" != "none" && -f "$placement" ]]; then
  extra_args+=(--init-expert-location "$placement")
fi
# the resident experts through Marlin (W4A16, matches the checkpoint)
if [[ "$gpu_experts" != "0" && "${KT_HOT_BACKEND:-marlin}" == "marlin" ]]; then
  extra_args+=(--moe-runner-backend marlin)
  export KT_GPU_MASK_ZERO=1
fi

# --- kt-kernel and the streamed prefill ---------------------------------------
export KT_STREAM_MEMOPS="${KT_STREAM_MEMOPS:-1}"
export KT_TASKQUEUE_SPIN_US="${KT_TASKQUEUE_SPIN_US:-300}"
export KT_MXFP4_BACKEND="${KT_MXFP4_BACKEND:-avx2}"
export KT_EXPERT_SHM="${KT_EXPERT_SHM:-1}"
export KT_GPU_STREAM_ZEROCOPY="${KT_GPU_STREAM_ZEROCOPY:-1}"
export KT_GPU_STREAM_GRAPH="${KT_GPU_STREAM_GRAPH:-1}"
export KT_GPU_STREAM_W16="${KT_GPU_STREAM_W16:-1}"
export KT_GPU_STREAM_PREFILL="${KT_GPU_STREAM_PREFILL:-512}"
export KT_GPU_STREAM_GROUP="${KT_GPU_STREAM_GROUP:-4}"
export KT_GPU_STREAM_EARLY_INIT="${KT_GPU_STREAM_EARLY_INIT:-1}"
export KT_GPU_STREAM_DROP_CACHE="${KT_GPU_STREAM_DROP_CACHE:-$model}"
export KT_LOAD_PREFETCH="${KT_LOAD_PREFETCH:-4}"
export KT_FAST_PINNED="${KT_FAST_PINNED:-0}"
export KT_SWIGLU_CLAMP="${KT_SWIGLU_CLAMP:-1}"
export SGLANG_UNBALANCED_LOAD_TIMEOUT_S="${SGLANG_UNBALANCED_LOAD_TIMEOUT_S:-1800}"
# the first launch of a new environment compiles its kernels inside the warmup request
# (over 10 minutes here), past SGLang's 600 s default; later launches reuse the cache
export SGLANG_WARMUP_TIMEOUT="${SGLANG_WARMUP_TIMEOUT:-3600}"
if (( tp > 1 )); then
  # all-reduce through host memory: decode in NCCL's LL form, prefill staged
  export SGLANG_LL_ALLREDUCE="${SGLANG_LL_ALLREDUCE:-1}"
  export SGLANG_HOST_STAGED_ALLREDUCE="${SGLANG_HOST_STAGED_ALLREDUCE:-1}"
fi
if grep -q '\[never\]' /sys/kernel/mm/transparent_hugepage/shmem_enabled 2>/dev/null; then
  echo "WARNING: shmem THP is 'never'; the expert arenas get 4K pages and pinning them takes most of an hour." >&2
  echo "         echo advise | sudo tee /sys/kernel/mm/transparent_hugepage/shmem_enabled" >&2
fi
[[ "${SGLANG_SWA_BOUNDED_REPLAY:-0}" == "1" ]] && extra_args+=(--enable-decoder-swa-bounded-replay)

# --- engram from disk ---------------------------------------------------------
export DSV41_ENGRAM_DIR="$engram_dir"
export DSV41_ENGRAM_BASE="${DSV41_ENGRAM_BASE:-$model}"
export DSV41_CACHE_GIB="${DSV41_CACHE_GIB:-8}"
export DSV41_ENGRAM_MODE="${DSV41_ENGRAM_MODE:-nvme}"

autotune_args=()
[[ "${SGLANG_DISABLE_FLASHINFER_AUTOTUNE:-1}" == "1" ]] && autotune_args=(--disable-flashinfer-autotune)
spec_args=()
if [[ "${SGLANG_ENABLE_DSPARK:-0}" == "1" ]]; then
  spec_args=(--speculative-algorithm DSPARK --speculative-attention-mode "${SGLANG_SPEC_ATTENTION_MODE:-decode}")
fi

if pgrep -af -- "-m sglang[. ](serve|launch_server)" | grep -F -- "--port $port " >/dev/null; then
  echo "an SGLang server already listens on port $port" >&2
  exit 6
fi

# --- prefill lend: the decoder's weights and the KV pool's free pages lent to the
# prefill, so the chunk can grow without taking the KV pool
KT_PREFILL_LEND="${KT_PREFILL_LEND:-1}"
export KT_PREFILL_LEND
log_name="$(basename "$model")"
if [[ "$KT_PREFILL_LEND" == 1 ]]; then
  lend_max_chunk="${DSV41_LEND_MAX_CHUNK:-6144}"
  export KT_PREFILL_LEND_DYNAMIC_MAX="${KT_PREFILL_LEND_DYNAMIC_MAX:-$lend_max_chunk}"
  export KT_PREFILL_LEND_MAX_CHUNK="$lend_max_chunk"
  export KT_PREFILL_LEND_PROBE_CONTEXT="${KT_PREFILL_LEND_PROBE_CONTEXT:-$max_total_tokens}"
  source "$here/lib/kt-lend-auto.sh"
  export KT_PREFILL_LEND_KV_CAP="${KT_PREFILL_LEND_KV_CAP:-$max_total_tokens}"
  # the first launch of a setup calibrates the chunk once (one more server start)
  kt_lend_auto "$0" SGLANG_CHUNKED_PREFILL_SIZE "$port" "$model" "$context_length" "$max_total_tokens" \
    "$mem_fraction" "${gpus[*]}" "$gpu_experts" "$placement" "${SGLANG_SWA_PREFIX_TAILS:-6}" || exit $?
  chunked_prefill="$KT_LEND_CHUNK"
  if [[ -z "${SGLANG_CHUNKED_PREFILL_SIZE:-}" ]] && (( chunked_prefill > lend_max_chunk )); then
    chunked_prefill="$lend_max_chunk"
  fi
  KT_LEND_CHUNK="$chunked_prefill"
  kt_prefill_plan_setup swa
  chunked_prefill="$KT_LEND_CHUNK"
  echo "prefill lend: chunk $chunked_prefill, SWA prefix tails $swa_prefix_tails" >&2
fi
extra_args+=(--swa-prefix-tails "$swa_prefix_tails")

# --- environment ----------------------------------------------------------------
run_dir="$log_root/$log_name/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$run_dir"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${gpus[*]}")"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}" NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}" NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
arch_a="$cap"
case "$cap" in 9.0|10.0|10.3|12.0|12.1) arch_a="${cap}a" ;; esac
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$cap+PTX}"
export FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-$arch_a}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$venv/bin:$PATH"
# compiled-kernel caches kept per environment (another venv's builds do not match)
export SGLANG_CACHE_DIR="${SGLANG_CACHE_DIR:-$HOME/.cache/sglang-dsv41-1017}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$HOME/.cache/torch_extensions-dsv41-1017}"
export OMP_NUM_THREADS="$cpu_threads"
# SGLang's NUMA bind v2 is a hard membind that can OOM the CUDA JIT while the experts
# fill the nodes; v1 prefers the local node and can spill
export SGLANG_NUMA_BIND_V2="${SGLANG_NUMA_BIND_V2:-0}"
# the patched tree shadows any installed sglang; the adapter's sitecustomize.py
# installs the engram hook in every worker
export PYTHONPATH="$source_dir:$adapter_dir${PYTHONPATH:+:$PYTHONPATH}"

command=(
  "$python" -m sglang.launch_server
  --host "${SGLANG_HOST:-0.0.0.0}"
  --port "$port"
  --model-path "$model"
  --load-format safetensors
  --trust-remote-code
  --served-model-name "$alias_name"
  --tensor-parallel-size "$tp"
  --numa-node $numa_list
  --disable-custom-all-reduce
  --kt-weight-path "$model"
  --kt-method MXFP4
  --kt-num-gpu-experts "$gpu_experts"
  --kt-cpuinfer "$cpu_threads"
  --kt-threadpool-count "$threadpools"
  --kt-max-deferred-experts-per-token "$deferred_experts"
  --context-length "$context_length"
  --max-total-tokens "$max_total_tokens"
  --kv-cache-dtype "$kv_cache_dtype"
  --mem-fraction-static "$mem_fraction"
  --max-running-requests 1
  --chunked-prefill-size "$chunked_prefill"
  --max-prefill-tokens "$chunked_prefill"
  --sampling-backend pytorch
  --watchdog-timeout 1200
  --disable-shared-experts-fusion
  "${radix_args[@]}"
  --enable-metrics
  "${autotune_args[@]}"
  "${extra_args[@]}"
  "${spec_args[@]}"
)

printf '%s\n' "DeepSeek-V4.1-Flash on SGLang + KTransformers: $model"
printf '%s\n' "GPUs ${CUDA_VISIBLE_DEVICES} (sm $cap), TP $tp on NUMA nodes $numa_list; CPU experts: $cpu_threads threads in $threadpools pools; GPU experts $gpu_experts a layer"
printf '%s\n' "context $context_length; attention output FP8 before wo_a: block ${SGLANG_DSV4_O_FP8_QDQ_BLOCK:-0}"
printf '%s\n' "wo_a native FP8: $SGLANG_DSV4_WO_A_FP8_NATIVE; vision tower on demand: $SGLANG_DSV4_VISION_ON_DEMAND; MXFP4 MoE fp32: $SGLANG_MXFP4_MOE_FP32; Marlin MoE fp32 glue: $SGLANG_MARLIN_MOE_FP32_GLUE; hot lend spare floor: $KT_KV_HOT_SPARE_FLOOR_MB MiB"
printf '%s\n' "Run directory: $run_dir"
ln -sfn "$run_dir" "$log_root/$log_name/current"
if [[ "$mode" == "--background" ]]; then
  nohup "${command[@]}" >>"$run_dir/server.log" 2>&1 &
  pid=$!
  printf '%s\n' "$pid" >"$run_dir/server.pid"
  echo "Started PID $pid"
else
  "${command[@]}" 2>&1 | tee -a "$run_dir/server.log"
fi
