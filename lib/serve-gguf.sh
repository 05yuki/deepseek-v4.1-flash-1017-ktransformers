#!/usr/bin/env bash
# Serve a GGUF build of DeepSeek-V4.1-Flash-1017 with SGLang + KTransformers from what setup.sh built.
# serve.sh runs this when GGUF is set:
#
#   GGUF=<dir with the GGUF shards and the Engram file> ./serve.sh [--foreground|--background]
#
# One GPU and the CPU socket next to it. The routed experts run on the CPU from the GGUF rows
# (kt-kernel's LLAMAFILE backend, held in an expert cache in RAM); attention and the dense layers
# run on the GPU in their GGUF types, and the GPU's free memory holds as many hot experts as fit,
# swapped for what the text routes to. The prefill streams the experts over PCIe to the GPU. The
# engram tables are read from the Engram GGUF on disk (a small RAM cache).
#
# The machine is read at start: the first visible GPU (CUDA_VISIBLE_DEVICES picks it), the NUMA
# node it hangs off, that node's physical cores, its memory (and this process's cgroup limit)
# for the expert cache. Every default below can be overridden by the variable named next to it.
# Logs go to logs/<build>/<timestamp>/.
set -Eeuo pipefail
export LC_ALL=C PYTHONUNBUFFERED=1
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
root="$here"  # read by lib/*.sh
mode="${1:---foreground}"
if [[ "$mode" != "--foreground" && "$mode" != "--background" ]]; then
  echo "Usage: GGUF=<GGUF dir> ./serve.sh [--foreground|--background]" >&2
  exit 2
fi
[[ -n "${GGUF:-}" && -d "$GGUF" ]] || { echo "set GGUF to the directory with the GGUF shards and the Engram file" >&2; exit 4; }
gguf_dir="$(cd "$GGUF" && pwd)"
state="$here/state/gguf-$(basename "$gguf_dir")"
# config.json and the tokenizer that setup.sh wrote for this build (no weights)
model="$state/config"
[[ -f "$model/config.json" && -f "$state/experts.json" ]] || { echo "run GGUF=$GGUF ./setup.sh first ($model missing)" >&2; exit 3; }
# the dense weights in their GGUF types, the routed experts through kt-kernel
export KT_GGUF_V41_DENSE="$gguf_dir"
# one resident GPU expert a layer and the hot lend with dynamic slots; the lends make the room,
# so the static fraction keeps none
export KT_GPU_EXPERTS="${KT_GPU_EXPERTS:-1}" KT_KV_HOT_LEND="${KT_KV_HOT_LEND:-1}"
export KT_KV_HOT_EXTRA="${KT_KV_HOT_EXTRA:-auto}" KT_KV_HOT_DYNAMIC_SLOTS="${KT_KV_HOT_DYNAMIC_SLOTS:-1}"
export KT_HOT_DYNAMIC="${KT_HOT_DYNAMIC:-0}"
export SGLANG_MEM_FRACTION="${SGLANG_MEM_FRACTION:-1.0}"
# the GGUF hot method takes -1 for a CPU route: no Marlin runner and no KT_GPU_MASK_ZERO
export KT_HOT_BACKEND="${KT_HOT_BACKEND:-gguf}"
export KT_KV_VMM="${KT_KV_VMM:-1}" KT_PREFILL_KV_LEND="${KT_PREFILL_KV_LEND:-1}"
export KT_PREFILL_PLAN="${KT_PREFILL_PLAN:-1}" KT_PREFILL_SWA_LEND="${KT_PREFILL_SWA_LEND:-1}"
# V4.1 runs layers 20-39 of a long prefill on the last 2048 rows only
export KT_PREFILL_PLAN_TAIL_ROWS="${KT_PREFILL_PLAN_TAIL_ROWS:-2048}"
# the streamed prefill reads the experts' GGUF rows from the file (O_DIRECT) and multiplies them
# with llama.cpp's MMQ on the GPU
export KT_GGUF_STREAM_SOURCE="${KT_GGUF_STREAM_SOURCE:-nvme}"
export KT_GPU_STREAM_PREFILL="${KT_GPU_STREAM_PREFILL:-1024}" KT_GPU_STREAM_EARLY_INIT="${KT_GPU_STREAM_EARLY_INIT:-0}"
export KT_GPU_STREAM_GROUP="${KT_GPU_STREAM_GROUP:-8}"
export KT_GGUF_STREAM_MMQ="${KT_GGUF_STREAM_MMQ:-1}"
export KT_GGUF_STREAM_RING="${KT_GGUF_STREAM_RING:-16}"
# wo_a is a GGUF tensor here, not the checkpoint's FP8
export SGLANG_DSV4_WO_A_FP8_NATIVE="${SGLANG_DSV4_WO_A_FP8_NATIVE:-0}"
# no expert placement: an expert map makes kt-kernel copy the experts into its arenas instead
# of computing them from the file's rows
export KT_EXPERT_PLACEMENT="${KT_EXPERT_PLACEMENT:-none}"
# wq_a and wkv as two layers: GGUF bytes do not go through the fused load
export SGLANG_OPT_FUSE_WQA_WKV="${SGLANG_OPT_FUSE_WQA_WKV:-0}"
# kt-kernel computes from the GGUF's rows (mapped, then held in the expert cache), no copy
export KT_GGUF_NOCOPY="${KT_GGUF_NOCOPY:-1}" KT_GGUF_ZEROCOPY="${KT_GGUF_ZEROCOPY:-1}"
export KT_LOAD_PREFETCH="${KT_LOAD_PREFETCH:-0}"
# the prefill lend keeps a copy of the dense weights it lends: read back from the GGUF instead
export KT_PREFILL_LEND_MIRROR="${KT_PREFILL_LEND_MIRROR:-gguf}"
export DSV41_LEND_MAX_CHUNK="${DSV41_LEND_MAX_CHUNK:-16384}"
export DSV41_CACHE_GIB="${DSV41_CACHE_GIB:-2}"
# glibc hands large frees back to the node instead of keeping them in the arenas
export MALLOC_MMAP_THRESHOLD_="${MALLOC_MMAP_THRESHOLD_:-1048576}" MALLOC_TRIM_THRESHOLD_="${MALLOC_TRIM_THRESHOLD_:-4194304}"
venv="$here/.venv"
python="$venv/bin/python"
source_dir="$here/third_party/sglang/python"
adapter_dir="$here/engram-adapter"
engram_dir="${DSV41_ENGRAM_DIR:-$state/engram}"
export KT_KV_VMM_LIB="${KT_KV_VMM_LIB:-$here/build/libkv_vmm.so}"
export KT_LEND_PY="$source_dir/sglang/srt/layers/moe/kt_lend.py"
port="${SGLANG_PORT:-8080}"
log_root="$here/logs"
alias_name="${SGLANG_SERVED_NAME:-$(basename "$gguf_dir")}"
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
  "$python" "$here/tools/make-engram-manifest-gguf.py" "$gguf_dir" "$engram_dir" >&2
fi

# --- the machine ------------------------------------------------------------
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found" >&2; exit 3; }
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -ra gpus <<<"$CUDA_VISIBLE_DEVICES"
else
  mapfile -t gpus < <(nvidia-smi --query-gpu=index --format=csv,noheader)
fi
# Every visible GPU, tensor parallel (KT_GGUF_TP=<n>: the first n). The dense layers split over the
# GPUs; the routed experts do not (a GGUF block cannot be cut), so each GPU streams and holds whole
# experts, its own share (the patch's KT_GGUF_STREAM_EP and KT_GGUF_HOT_EP). What the GPUs share is
# set by the weakest of them: the oldest architecture, the smallest memory.
if [[ "${KT_GGUF_TP:-}" =~ ^[0-9]+$ ]] && (( KT_GGUF_TP >= 1 && KT_GGUF_TP < ${#gpus[@]} )); then
  gpus=("${gpus[@]:0:KT_GGUF_TP}")
fi
n_gpus="${#gpus[@]}"
tp="$n_gpus"
caps=() vrams=() gpu_nodes=()
for g in "${gpus[@]}"; do
  caps+=("$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i "$g" | head -1)")
  vrams+=("$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$g" | head -1)")
  bus="$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i "$g" | head -1 | tr 'A-F' 'a-f')"
  n="$(cat "/sys/bus/pci/devices/${bus#0000}/numa_node" 2>/dev/null || cat "/sys/bus/pci/devices/${bus:4}/numa_node" 2>/dev/null || echo -1)"
  (( n >= 0 )) 2>/dev/null || n=0
  gpu_nodes+=("$n")
done
# cards of different sizes: SGLang refuses ranks whose free memory differs by more than 10%
# unless told this is meant (each rank then takes the smaller card's share)
if (( $(printf '%s\n' "${vrams[@]}" | sort -u | wc -l) > 1 )); then
  export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK="${SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK:-0}"
fi
# the lowest compute capability, and whether any GPU is Turing or Ada
cap="$(printf '%s\n' "${caps[@]}" | sort -t. -k1,1n -k2,2n | head -1)"
any_cap() { local c; for c in "${caps[@]}"; do [[ "$c" == "$1" ]] && return 0; done; return 1; }
# Tier 0 supplies a portable Triton attention fallback for Ada (SM89); the GGUF patch takes it to
# Turing (SM75). Other pre-Hopper architectures remain unsupported; do not load the model first.
for i in "${!gpus[@]}"; do
  c="${caps[$i]}"
  if [[ "$c" != "8.9" && "$c" != "7.5" && "${c%%.*}" =~ ^[0-9]+$ ]] && (( ${c%%.*} < 9 )); then
    echo "GPU ${gpus[$i]} is compute capability $c: attention needs SM75 (Turing), SM89 (Ada) or SM90+" >&2
    exit 3
  fi
done
if any_cap 8.9 || any_cap 7.5; then
  # Ada and Turing use the portable indexer and mHC paths as well as sparse attention.
  # The DeepGEMM and TileLang alternatives require newer GPU architectures.
  export SGLANG_OPT_USE_TOPK_V2="${SGLANG_OPT_USE_TOPK_V2:-0}"
  export SGLANG_OPT_USE_TILELANG_MHC_PRE="${SGLANG_OPT_USE_TILELANG_MHC_PRE:-0}"
  export SGLANG_OPT_DEEPGEMM_HC_PRENORM="${SGLANG_OPT_DEEPGEMM_HC_PRENORM:-0}"
  export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH="${SGLANG_FP8_PAGED_MQA_LOGITS_TORCH:-1}"
fi
sm75_bf16=0
if any_cap 7.5; then
  # Turing keeps the model in bf16 (DSV4's JIT kernels take no fp16; the tensor-core dots go
  # through fp32), and the GGUF dense layers' prefill runs on MMQ in their GGUF types (the bf16
  # copy path needs a bf16 cuBLAS GEMM, SM80+)
  export KT_SM75_BF16=1
  sm75_bf16=1
  export SGLANG_GGUF_DEQUANT_ROWS="${SGLANG_GGUF_DEQUANT_ROWS:-1000000000}"
  export SGLANG_DSV4_WO_A_DEQUANT_ROWS="${SGLANG_DSV4_WO_A_DEQUANT_ROWS:-1000000000}"
  # a 2048-token prefill chunk without the calibration launch: TileLang's mHC kernels fail to
  # launch on larger chunks there (CUDA_ERROR_INVALID_VALUE at 8192), and the CPU sets the pace
  export SGLANG_CHUNKED_PREFILL_SIZE="${SGLANG_CHUNKED_PREFILL_SIZE:-2048}"
fi
# 131,072 tokens (1,048,576 when the RAM holds every expert: below); a card under 12 GB (an 8 GB
# RTX 20) keeps 32,768, its room after the weights (the smallest card: every rank holds the same
# share of the KV pool)
vram_mib="$(printf '%s\n' "${vrams[@]}" | sort -n | head -1)"
context_vram=131072
[[ "$vram_mib" =~ ^[0-9]+$ ]] && (( vram_mib < 12288 )) && context_vram=32768
n_nodes="$(ls -d /sys/devices/system/node/node[0-9]* 2>/dev/null | wc -l)"
(( n_nodes > 0 )) || n_nodes=1
phys_cores="$(lscpu -p=CORE,SOCKET 2>/dev/null | grep -v '^#' | sort -u | wc -l)"
(( phys_cores > 0 )) || phys_cores="$(nproc)"
# each rank on the NUMA node its GPU hangs off (SGLANG_NUMA_NODES="<n> ..." picks them, one a rank)
numa_list="${SGLANG_NUMA_NODES:-${gpu_nodes[*]}}"
# kt-kernel's CPU experts: one pool on each of those nodes
nodes=()
for n in $numa_list; do [[ " ${nodes[*]} " == *" $n "* ]] || nodes+=("$n"); done
node="${nodes[0]}"
export KT_NUMA_NODES="${KT_NUMA_NODES:-$(IFS=,; echo "${nodes[*]}")}"
# a node's physical cores, each as its first hardware thread (the one kt-kernel pins a thread to)
node_threads_of() { lscpu -p=CPU,NODE,CORE 2>/dev/null | grep -v '^#' | awk -F, -v n="$1" '$2==n' | sort -t, -k3,3n -k1,1n; }
n_node_cores=0
for n in "${nodes[@]}"; do
  c="$(node_threads_of "$n" | awk -F, 'NF && !seen[$3]++' | grep -c . || true)"
  n_node_cores=$(( n_node_cores + c ))
done
have_cores=1
(( n_node_cores > 0 )) || { have_cores=0; n_node_cores="$phys_cores"; }

# --- the expert cache ---------------------------------------------------------
# kt-kernel keeps the experts' GGUF rows it has read in RAM on the node, up to
# KT_GGUF_EXPERT_CACHE_GIB; what does not fit is read from the file again when routed. The
# default takes the node's memory (or this process's cgroup limit, if lower) less
# KT_GGUF_HOST_RESERVE_GIB for everything else. When the node holds every expert and their
# IQ1_M decode copy, the copy is kept too (KT_GGUF_CACHE_IQ1M_T=1: faster CPU decode for the
# IQ1_M layers, at 76 more bytes a block of 256); otherwise the raw rows only.
node_gib=0
for n in "${nodes[@]}"; do
  g="$(awk '/MemTotal/ {printf "%d", $4 / 1048576}' "/sys/devices/system/node/node$n/meminfo" 2>/dev/null || true)"
  node_gib=$(( node_gib + ${g:-0} ))
done
(( node_gib > 0 )) || node_gib="$(awk '/MemTotal/ {printf "%d", $2 / 1048576}' /proc/meminfo)"
cg_max="$(cat "/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup | head -1)/memory.max" 2>/dev/null || echo max)"
if [[ "$cg_max" =~ ^[0-9]+$ ]] && (( cg_max / 1073741824 < node_gib )); then node_gib=$(( cg_max / 1073741824 )); fi
# what everything else takes: 18 GiB, and as much again for each more rank
budget_gib=$(( node_gib - ${KT_GGUF_HOST_RESERVE_GIB:-18} * tp ))
read -r raw_gib copy_gib < <("$python" -c "
import json, math; e = json.load(open('$state/experts.json'))
print(math.ceil(e['raw_bytes'] / 2**30), math.ceil(e['iq1m_copy_bytes'] / 2**30))")
if [[ -z "${KT_GGUF_CACHE_IQ1M_T:-}" ]]; then
  KT_GGUF_CACHE_IQ1M_T=0
  (( copy_gib > 0 && budget_gib >= raw_gib + copy_gib + 4 )) && KT_GGUF_CACHE_IQ1M_T=1
fi
export KT_GGUF_CACHE_IQ1M_T
want_gib=$(( raw_gib + 4 + (KT_GGUF_CACHE_IQ1M_T == 1 ? copy_gib : 0) ))
cache_gib=$(( want_gib < budget_gib ? want_gib : budget_gib ))
export KT_GGUF_EXPERT_CACHE_GIB="${KT_GGUF_EXPERT_CACHE_GIB:-$cache_gib}"
(( KT_GGUF_EXPERT_CACHE_GIB >= 8 )) || { echo "nodes ${nodes[*]} have ${node_gib} GiB: too little for the expert cache (KT_GGUF_EXPERT_CACHE_GIB)" >&2; exit 5; }
# the context: the model's 1,048,576 tokens on one GPU of 12 GB or more when the node holds every
# expert with 2 GiB to spare, the compressed KV past the GPU's share and the RoPE rows past
# 131,072 in host memory (host-kv; about 1.5 GiB at 1M); otherwise context_vram above, all on the
# GPU, so a node that cannot hold the experts keeps every GiB for the expert cache
if (( tp == 1 && context_vram == 131072 && budget_gib >= raw_gib + 4 + 2 )); then
  export SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-1048576}" SGLANG_DSV41_HOST_KV="${SGLANG_DSV41_HOST_KV:-1}"
else
  export SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-$context_vram}" SGLANG_DSV41_HOST_KV="${SGLANG_DSV41_HOST_KV:-0}"
fi
if (( tp > 1 )); then
  # the shared expert whole on every rank: its down projection's 2304 inputs split in two are
  # 1152, not a whole number of 256-value GGUF blocks (+0.5 GB a GPU)
  export SGLANG_SHARED_EXPERT_TP1="${SGLANG_SHARED_EXPERT_TP1:-1}"
  # kt-kernel's cache holds each expert split over its pools, which no rank's prefill can read
  # whole: the prefill reads the GGUF through the page cache when RAM holds the experts once
  # more (from the drive, O_DIRECT, otherwise)
  if [[ -z "${KT_GGUF_STREAM_DIRECT:-}" ]] && (( budget_gib >= KT_GGUF_EXPERT_CACHE_GIB + raw_gib + 8 )); then
    export KT_GGUF_STREAM_DIRECT=0
  fi
fi

# --- model and memory -------------------------------------------------------
context_length="${SGLANG_CONTEXT_LENGTH:-1048576}"
max_total_tokens="${SGLANG_MAX_TOTAL_TOKENS:-$context_length}"
# experts that always stay on each GPU per layer: one; the hot lend fills the GPU's free
# memory with more, swapped for what the text routes to (more here only takes memory the
# lend would use, and must fit at load)
gpu_experts="${KT_GPU_EXPERTS:-1}"
cpu_threads="${KT_CPU_THREADS:-$n_node_cores}"

# --- the threads that read the drive ---------------------------------------------
# The expert cache's readers, the hot swaps and the streamed prefill's readers run on KT_IO_CPUS,
# away from the CPUs the experts compute on (a read that waits for a busy CPU holds up the layer
# that needs its expert). The engram reads are left to the scheduler (DSV41_ENGRAM_IO_CPUS pins
# them; pinned, the decode was a few percent slower on some texts).
# - With SMT: the second hardware threads of the node's last cores, one for every four readers;
#   the experts compute on every core (on one EPYC 7452 socket, +2 to +10% decode against four
#   whole cores for the readers, at 64 GB and with every expert in RAM).
# - Without SMT: the node's last cores, as many as reading the drive at full speed keeps busy;
#   the experts compute on the rest.
# The number of readers that fills the drive, and the CPU time a GB of reads takes, come from
# state/gguf-<name>/io-probe.json, which setup.sh measures (tools/io-probe.py); without it, 16
# readers on one CPU's worth.
io_readers="" io_cores=""
if [[ -f "$state/io-probe.json" ]]; then
  read -r io_readers io_cores < <("$python" -c "
import json, math, os
p = json.load(open('$state/io-probe.json'))
if os.path.realpath(p['path']) == os.path.realpath('$gguf_dir'):  # measured on this GGUF's drive
    c = p['pieces']['512']
    r = c['saturating_threads']
    row = next(x for x in c['curve'] if x['threads'] == r)
    print(r, max(1, math.ceil(row['gb_per_s'] * row['cpu_s_per_gb'])))" 2>/dev/null)
fi
io_readers="${io_readers:-16}" io_cores="${io_cores:-1}"
if [[ -z "${KT_IO_CPUS:-}" ]] && (( have_cores )); then
  # on each node: the readers that node's ranks start
  io=() compute=0
  for n in "${nodes[@]}"; do
    threads="$(node_threads_of "$n")"
    cores="$(echo "$threads" | awk -F, 'NF && !seen[$3]++')"
    nc="$(echo "$cores" | grep -c . || true)"
    if (( nc < 4 )); then compute=$(( compute + nc )); continue; fi
    max_io=$(( nc / 4 ))
    siblings="$(echo "$threads" | awk -F, 'NF && seen[$3]++ == 1 {print $1}')"
    if [[ -n "$siblings" ]]; then
      n_io=$(( (io_readers + 3) / 4 )); (( n_io <= max_io )) || n_io=$max_io
      io+=($(echo "$siblings" | tail -n "$n_io"))
      compute=$(( compute + nc ))
    else
      n_io=$io_cores; (( n_io <= max_io )) || n_io=$max_io
      io+=($(echo "$cores" | tail -n "$n_io" | cut -d, -f1))
      compute=$(( compute + nc - n_io ))
    fi
  done
  (( ${#io[@]} )) && KT_IO_CPUS="$(IFS=,; echo "${io[*]}")"
  [[ -n "${KT_CPU_THREADS:-}" ]] || cpu_threads=$compute
fi
if [[ -n "${KT_IO_CPUS:-}" ]]; then
  export KT_IO_CPUS
  export KT_GGUF_PREFETCH_CPUS="${KT_GGUF_PREFETCH_CPUS:-$KT_IO_CPUS}"
fi
export KT_GGUF_PREFETCH_THREADS="${KT_GGUF_PREFETCH_THREADS:-$io_readers}"
export KT_GGUF_STREAM_READERS="${KT_GGUF_STREAM_READERS:-$io_readers}"
export KT_GGUF_CACHE_PIECE_KIB="${KT_GGUF_CACHE_PIECE_KIB:-512}"
threadpools="${KT_THREADPOOL_COUNT:-${#nodes[@]}}"
deferred_experts="${KT_DEFERRED_EXPERTS:-0}"  # >0 drops expert work for speed; it hurts the text
mem_fraction="${SGLANG_MEM_FRACTION:-0.81}"
kv_cache_dtype="${SGLANG_KV_CACHE_DTYPE:-fp8_e4m3}"
chunked_prefill="${SGLANG_CHUNKED_PREFILL_SIZE:-2048}"
swa_prefix_tails="${SGLANG_SWA_PREFIX_TAILS:-6}"
# split on whitespace on purpose: SGLANG_EXTRA_ARGS="--a 1 --b 2" gives four words (no quoting inside)
extra_args=(${SGLANG_EXTRA_ARGS:-})
(( sm75_bf16 )) && extra_args+=(--dtype bfloat16)
radix_args=(--disable-radix-cache)
[[ "${DSV41_RADIX:-0}" == "1" ]] && radix_args=()

# DeepSeek-V4.1-Flash-1017 rounds the attention output to FP8 before wo_a; the block
# comes from the checkpoint's own inference code (off for the earlier release)
source "$here/tools/o-fp8-qdq-block.sh"
# V4.1 keeps its KV in its own formats (tier 0's v41-kv-layout): the window KV as all 512
# values in FP8 blocks of 32, the RoPE part too, and the compressed KV in FP4 blocks of 16.
# SGLang reads them on SM100 and, with this repository, on SM120 / SM121; other GPUs keep the
# V4 layout (v4: the V4 layout everywhere)
all_v41_kv=1
for c in "${caps[@]}"; do [[ "${c%%.*}" == 10 || "${c%%.*}" == 12 ]] || all_v41_kv=0; done
if [[ -z "${SGLANG_DSV4_KV_LAYOUT:-}" ]] && (( all_v41_kv )); then
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
# cpu-gpu-overlap (tier1.md §19): the shared expert forks after the CPU request, and one kernel
# stages the request in the pinned buffers; both on
if has cpu-gpu-overlap; then
  export KT_SHARED_AFTER_SUBMIT="${KT_SHARED_AFTER_SUBMIT:-1}"
  export KT_HANDOFF_PACK="${KT_HANDOFF_PACK:-1}"
elif [[ "${KT_SHARED_AFTER_SUBMIT:-0}" != 0 || "${KT_HANDOFF_PACK:-0}" != 0 ]]; then
  echo "KT_SHARED_AFTER_SUBMIT/KT_HANDOFF_PACK require the cpu-gpu-overlap tier-1 feature" >&2
  exit 3
fi
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
export DSV41_ENGRAM_BASE="${DSV41_ENGRAM_BASE:-$gguf_dir}"
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
log_name="$(basename "$gguf_dir")"
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
run_dir="$log_root/$log_name/$(date +%Y%m%d-%H%M%S)${SGLANG_PORT:+-p$SGLANG_PORT}"  # two servers started in the same second
mkdir -p "$run_dir"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${gpus[*]}")"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}" NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}" NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
# the kernels built at run time: every GPU's architecture
arch_list=() arch_a_list=()
for c in $(printf '%s\n' "${caps[@]}" | sort -u -t. -k1,1n -k2,2n); do
  arch_list+=("$c")
  case "$c" in 9.0|10.0|10.3|12.0|12.1) arch_a_list+=("${c}a") ;; *) arch_a_list+=("$c") ;; esac
done
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$(IFS=';'; echo "${arch_list[*]}")+PTX}"
export FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-${arch_a_list[*]}}"
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
  --kt-weight-path "$gguf_dir"
  --kt-method LLAMAFILE
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
  --max-prefill-tokens "$(( chunked_prefill + KT_PREFILL_PLAN_TAIL_ROWS ))"
  --sampling-backend pytorch
  --watchdog-timeout 1200
  --disable-shared-experts-fusion
  "${radix_args[@]}"
  --enable-metrics
  "${autotune_args[@]}"
  "${extra_args[@]}"
  "${spec_args[@]}"
)

printf '%s\n' "DeepSeek-V4.1-Flash-1017 GGUF on SGLang + KTransformers: $gguf_dir"
printf '%s\n' "GPU ${CUDA_VISIBLE_DEVICES} (sm $(IFS=,; echo "${caps[*]}"), $(IFS=,; echo "${vrams[*]}") MiB), TP $tp on NUMA nodes $numa_list; CPU experts: $cpu_threads threads in $threadpools pools; GPU experts $gpu_experts a layer + the hot lend"
printf '%s\n' "drive reads: $KT_GGUF_PREFETCH_THREADS cache readers and $KT_GGUF_STREAM_READERS prefill readers on CPUs ${KT_IO_CPUS:-(not pinned)}"
printf '%s\n' "expert cache: $KT_GGUF_EXPERT_CACHE_GIB GiB of nodes ${nodes[*]}'s ${node_gib} GiB for ${raw_gib} GiB of experts; IQ1_M decode copy: $KT_GGUF_CACHE_IQ1M_T (${copy_gib} GiB)"
printf '%s\n' "patch tier $tier; context $context_length; attention output FP8 before wo_a: block ${SGLANG_DSV4_O_FP8_QDQ_BLOCK:-0}; KV layout ${SGLANG_DSV4_KV_LAYOUT:-v4}"
(( tier >= 1 )) && printf '%s\n' "tier-1 features: $( [[ -f "$features_file" ]] && tr '\n' ' ' < "$features_file" || echo all)"
(( tier >= 1 )) && printf '%s\n' "wo_a native FP8: $SGLANG_DSV4_WO_A_FP8_NATIVE; vision tower on demand: $SGLANG_DSV4_VISION_ON_DEMAND; MXFP4 MoE fp32: $SGLANG_MXFP4_MOE_FP32; Marlin MoE fp32 glue: $SGLANG_MARLIN_MOE_FP32_GLUE; CPU scale codes: $KT_MXFP4_SCALE_CODES; hot lend spare floor: ${KT_KV_HOT_SPARE_FLOOR_MB:-off} MiB; dynamic hot slots: ${KT_KV_HOT_DYNAMIC_SLOTS:-0}, resident rows too: ${KT_KV_HOT_DYNAMIC_RESIDENT:-0}; shared expert after submit: ${KT_SHARED_AFTER_SUBMIT:-0}; packed hand-off: ${KT_HANDOFF_PACK:-0}"
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
