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
# the tier setup.sh applied (an empty marker, from before tiers, is the full tree)
tier="$(cut -d' ' -f1 "$here/third_party/sglang/.dsv41-1017-patched" 2>/dev/null || true)"
tier="${tier:-2}"
# the tier-1 features setup.sh applied (one per line; without the file, all of them)
features_file="$here/third_party/sglang/.dsv41-1017-features"
has() {
  (( tier >= 1 )) || return 1
  [[ ! -f "$features_file" ]] || grep -qx "$1" "$features_file"
}
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
# Tier 0 supplies a portable Triton attention fallback for Ada (SM89).
# Other pre-Hopper architectures remain unsupported; do not load the model first.
if [[ "$cap" != "8.9" && "${cap%%.*}" =~ ^[0-9]+$ ]] && (( ${cap%%.*} < 9 )); then
  echo "GPU ${gpus[0]} is compute capability $cap: attention needs SM89 (Ada) or SM90+" >&2
  exit 3
fi
if [[ "$cap" == "8.9" ]]; then
  # Ada uses the portable indexer and mHC paths as well as sparse attention.
  # The DeepGEMM and TileLang alternatives require newer GPU architectures.
  export SGLANG_OPT_USE_TOPK_V2="${SGLANG_OPT_USE_TOPK_V2:-0}"
  export SGLANG_OPT_USE_TILELANG_MHC_PRE="${SGLANG_OPT_USE_TILELANG_MHC_PRE:-0}"
  export SGLANG_OPT_DEEPGEMM_HC_PRENORM="${SGLANG_OPT_DEEPGEMM_HC_PRENORM:-0}"
  export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH="${SGLANG_FP8_PAGED_MQA_LOGITS_TORCH:-1}"
fi
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
# experts that always stay on each GPU per layer: one; the hot lend fills the GPU's free
# memory with more, swapped for what the text routes to (more here only takes memory the
# lend would use, and must fit at load)
gpu_experts="${KT_GPU_EXPERTS:-1}"
cpu_threads="${KT_CPU_THREADS:-$phys_cores}"
threadpools="${KT_THREADPOOL_COUNT:-$n_nodes}"
deferred_experts="${KT_DEFERRED_EXPERTS:-0}"  # >0 drops expert work for speed; it hurts the text
mem_fraction="${SGLANG_MEM_FRACTION:-0.81}"
kv_cache_dtype="${SGLANG_KV_CACHE_DTYPE:-fp8_e4m3}"
chunked_prefill="${SGLANG_CHUNKED_PREFILL_SIZE:-2048}"
swa_prefix_tails="${SGLANG_SWA_PREFIX_TAILS:-6}"
# split on whitespace on purpose: SGLANG_EXTRA_ARGS="--a 1 --b 2" gives four words (no quoting inside)
extra_args=(${SGLANG_EXTRA_ARGS:-})
radix_args=(--disable-radix-cache)
[[ "${DSV41_RADIX:-0}" == "1" ]] && radix_args=()

# DeepSeek-V4.1-Flash-1017 rounds the attention output to FP8 before wo_a; the block
# comes from the checkpoint's own inference code (off for the earlier release)
source "$here/tools/o-fp8-qdq-block.sh"
# V4.1 keeps its KV in its own formats (tier 0's v41-kv-layout): the window KV as all 512
# values in FP8 blocks of 32, the RoPE part too, and the compressed KV in FP4 blocks of 16.
# SGLang reads them on SM100 and, with this repository, on SM120 / SM121; other GPUs keep the
# V4 layout (v4: the V4 layout everywhere)
if [[ -z "${SGLANG_DSV4_KV_LAYOUT:-}" ]] && [[ "${cap%%.*}" == 10 || "${cap%%.*}" == 12 ]]; then
  export SGLANG_DSV4_KV_LAYOUT=v41
fi
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
# decode's block-FP8 GEMMs (up to four rows) through a GEMV that steps K by several scale
# groups (0: the block kernel)
export SGLANG_FP8_GEMV="${SGLANG_FP8_GEMV:-1}"
# decode's engram row lookups start with the token on a side stream, not at their layers
# (0: at the layers)
export SGLANG_DSV41_ENGRAM_PREFETCH_ROWS="${SGLANG_DSV41_ENGRAM_PREFETCH_ROWS:-1}"

if has hot-lend; then
  # hot experts: the resident ones, and with the hot lend as many more as the KV pool's free
  # pages and the driver's idle memory hold outside prefill (KT_KV_HOT_EXTRA=<n> caps them a
  # layer; KT_KV_HOT_LEND=0 turns it off)
  source "$here/lib/kt-kv-hot-default.sh"
  kt_kv_hot_default "$here/placements/dsv41-expert-placement-hot64.json" 5 "$gpu_experts" auto "$source_dir" swa
  # in decode the hot lend's slots also take the driver's idle memory above a 1 GiB floor
  # (-1: only the KV pool's free pages)
  export KT_KV_HOT_SPARE_FLOOR_MB="${KT_KV_HOT_SPARE_FLOOR_MB:-1024}"
  # free KV pages kept back for 65,536 tokens: a new request's pages come from them, so the
  # slots are not all handed back (and loaded again) for each request (4,096 did that)
  export KT_KV_HOT_RESERVE="${KT_KV_HOT_RESERVE:-65536}"
  # the slots follow what is routed: every 8 decode steps each layer swaps experts that
  # lose for ones that are routed more, copied from kt-kernel's expert memory (0: the
  # placement's next-hottest experts, recorded on Japanese prose)
  export KT_KV_HOT_DYNAMIC_SLOTS="${KT_KV_HOT_DYNAMIC_SLOTS:-1}"
  # the resident experts swap too, from the same pool (kt-kernel then loads every expert;
  # 0: the placement's first ones stay)
  export KT_KV_HOT_DYNAMIC_RESIDENT="${KT_KV_HOT_DYNAMIC_RESIDENT:-1}"
else
  export KT_KV_HOT_LEND=0
  if has kv-lend-plan; then
    # the KV pool on remappable memory, its free pages lent to the prefill, the chunk planned
    export KT_KV_VMM="${KT_KV_VMM:-1}" KT_PREFILL_KV_LEND="${KT_PREFILL_KV_LEND:-1}"
    export KT_PREFILL_PLAN="${KT_PREFILL_PLAN:-1}" KT_PREFILL_SWA_LEND="${KT_PREFILL_SWA_LEND:-1}"
  fi
fi
if has placement; then
  # the experts hottest first: any resident count takes the first ones
  placement="${KT_EXPERT_PLACEMENT:-${KT_KV_HOT_PLACEMENT:-$here/placements/dsv41-expert-placement-hot64.json}}"
else
  # without the placement feature the wrapper keeps experts 0..N-1 on the GPU
  placement=none
fi
# the prefill lend needs its feature (calibrated once per setup)
has prefill-lend || export KT_PREFILL_LEND="${KT_PREFILL_LEND:-0}"
if [[ "$placement" != "none" && -f "$placement" ]]; then
  extra_args+=(--init-expert-location "$placement")
elif [[ "$placement" != "none" ]]; then
  echo "WARNING: no placement $placement: the $gpu_experts GPU experts a layer are experts 0..$((gpu_experts - 1)), not the hottest" >&2
  echo "         (record your own with KT_ROUTING_DUMP, or KT_EXPERT_PLACEMENT=none)" >&2
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
# the CPU experts read a one-byte code per 32-value scale group instead of an fp32 scale
# (15% fewer bytes a weight, bit-identical; 0: fp32 scales)
export KT_MXFP4_SCALE_CODES="${KT_MXFP4_SCALE_CODES:-1}"
# the CPU experts' decode converts the activation row with PSHUFB and prefetches the weights
# 1 KB ahead, a line every 4 groups (bit-identical; 0 / 0: as before)
export KT_MXFP4_DEC_VCONV="${KT_MXFP4_DEC_VCONV:-1}"
export KT_MXFP4_DEC_PFS="${KT_MXFP4_DEC_PFS:-1024}"
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
  export KT_PREFILL_LEND_MAX_CHUNK="$lend_max_chunk"
  export KT_PREFILL_LEND_PROBE_CONTEXT="${KT_PREFILL_LEND_PROBE_CONTEXT:-$max_total_tokens}"
  # the prefill plan's cost before this setup's own fit (FIXED MiB, a row B, a sequence token
  # B, CARRY MiB): V4.1-1017's measured on two 16 GB cards, rounded up
  export KT_PREFILL_PLAN_COST_PRIOR="${KT_PREFILL_PLAN_COST_PRIOR:-2600,471040,280,820}"
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
# compiled-kernel caches kept in this checkout: another checkout's torch builds
# do not match, and two of them sharing a directory wait on each other's locks
export SGLANG_CACHE_DIR="${SGLANG_CACHE_DIR:-$here/.cache/sglang}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$here/.cache/torch_extensions}"
# a build killed mid-way leaves its lock, and the next server waits on it for good:
# nothing builds before this server starts, so a lock this old is a leftover
find "$TORCH_EXTENSIONS_DIR" -name lock -mmin +30 -delete 2>/dev/null || true
export OMP_NUM_THREADS="$cpu_threads"
# SGLang's NUMA bind v2 is a hard membind that can OOM the CUDA JIT while the experts
# fill the nodes; v1 prefers the local node and can spill
export SGLANG_NUMA_BIND_V2="${SGLANG_NUMA_BIND_V2:-0}"
# the patched tree shadows any installed sglang; the adapter's sitecustomize.py
# installs the engram hook in every worker
export PYTHONPATH="$source_dir:$adapter_dir${PYTHONPATH:+:$PYTHONPATH}"

# tool calls and reasoning come back as structured fields of the OpenAI API (tool_calls,
# reasoning_content) through V4.1's own detector and reasoning parser; an empty
# SGLANG_TOOL_CALL_PARSER= or SGLANG_REASONING_PARSER= leaves one out
tool_parser="${SGLANG_TOOL_CALL_PARSER-deepseekv41}"
reasoning_parser="${SGLANG_REASONING_PARSER-deepseek-v41}"
[[ -n "$tool_parser" ]] && extra_args+=(--tool-call-parser "$tool_parser")
[[ -n "$reasoning_parser" ]] && extra_args+=(--reasoning-parser "$reasoning_parser")

# SGLANG_HOST=0.0.0.0 serves other machines too; then set SGLANG_API_KEY, which every
# request must send as "Authorization: Bearer <key>"
[[ -n "${SGLANG_API_KEY:-}" ]] && extra_args+=(--api-key "$SGLANG_API_KEY")
command=(
  "$python" -m sglang.launch_server
  --host "${SGLANG_HOST:-127.0.0.1}"
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
printf '%s\n' "patch tier $tier; context $context_length; attention output FP8 before wo_a: block ${SGLANG_DSV4_O_FP8_QDQ_BLOCK:-0}; KV layout ${SGLANG_DSV4_KV_LAYOUT:-v4}"
(( tier >= 1 )) && printf '%s\n' "tier-1 features: $( [[ -f "$features_file" ]] && tr '\n' ' ' < "$features_file" || echo all)"
(( tier >= 1 )) && printf '%s\n' "wo_a native FP8: $SGLANG_DSV4_WO_A_FP8_NATIVE; vision tower on demand: $SGLANG_DSV4_VISION_ON_DEMAND; MXFP4 MoE fp32: $SGLANG_MXFP4_MOE_FP32; Marlin MoE fp32 glue: $SGLANG_MARLIN_MOE_FP32_GLUE; CPU scale codes: $KT_MXFP4_SCALE_CODES; hot lend spare floor: ${KT_KV_HOT_SPARE_FLOOR_MB:-off} MiB; dynamic hot slots: ${KT_KV_HOT_DYNAMIC_SLOTS:-0}, resident rows too: ${KT_KV_HOT_DYNAMIC_RESIDENT:-0}"
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
