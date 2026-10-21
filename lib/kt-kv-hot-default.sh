# Sourced by a launcher: the hot lend on by default (sglang/srt/layers/moe/kt_kv_hot_lend.py
# in the tier-1 hot-lend patches). Outside prefill, the KV pool's free pages
# hold X more hot experts; the pool never shrinks (they leave when KV needs the
# pages). Needs the KV pool on kv-vmm, so the prefill plan's set (KT_KV_VMM,
# KT_PREFILL_KV_LEND, KT_PREFILL_PLAN, KT_PREFILL_SWA_LEND on SWA models) comes on
# with it, and a placement whose physical N..N+X-1 are the next-hottest experts.
#
#   kt_kv_hot_default <hot placement> <resident count it keeps> <gpu experts> <X> <tree python dir> [swa]
#
# Any resident count works: the placement lists the experts hottest first, so the slots
# take the ones after the residents (the count it was built around is kept as an argument).
#
# On when KT_KV_HOT_LEND is unset or 1, KT_EXPERT_PLACEMENT is unset (or names the hot
# placement), the hot placement, libkv_vmm.so and the tree's kt_kv_hot_lend.py exist, and
# KT_KV_VMM is not 0. Then KT_KV_HOT_PLACEMENT is the placement to use. Otherwise
# KT_KV_HOT_LEND=0 and KT_KV_HOT_PLACEMENT is empty.
kt_kv_hot_default() {
  local hot="$1" keep="$2" n="$3" x="$4" tree="$5" swa="${6:-}"
  local lib="${KT_KV_VMM_LIB:-${here:-.}/build/libkv_vmm.so}" why=""
  KT_KV_HOT_PLACEMENT=""
  if [[ "${KT_KV_HOT_LEND:-1}" != 1 ]]; then
    why="KT_KV_HOT_LEND=${KT_KV_HOT_LEND}"
  elif [[ -n "${KT_EXPERT_PLACEMENT:-}" && "${KT_EXPERT_PLACEMENT}" != "$hot" && "${KT_KV_HOT_LEND:-}" != 1 ]]; then
    # a placement given turns the lend off unless KT_KV_HOT_LEND=1 is given too: then
    # the lend runs on that placement (its N..N+X-1 must be the next hottest)
    why="a placement given ($KT_EXPERT_PLACEMENT; KT_KV_HOT_LEND=1 keeps the lend on it)"
  elif [[ ! -f "$hot" ]]; then
    why="$hot missing"
  elif [[ ! -f "$lib" ]]; then
    why="$lib missing (./setup.sh builds it from kv-vmm/kv_vmm.c)"
  elif [[ ! -f "$tree/sglang/srt/layers/moe/kt_kv_hot_lend.py" ]]; then
    why="the tree has no kt_kv_hot_lend.py (tier 1 hot-lend feature not applied?)"
  elif [[ "${KT_KV_VMM:-1}" == 0 ]]; then
    why="KT_KV_VMM=0"
  fi
  if [[ -n "$why" ]]; then
    [[ "${KT_KV_HOT_LEND:-1}" == 1 ]] && echo "hot lend off: $why" >&2
    export KT_KV_HOT_LEND=0
    return 0
  fi
  export KT_KV_HOT_LEND=1 KT_KV_HOT_EXTRA="${KT_KV_HOT_EXTRA:-$x}"
  export KT_KV_VMM=1 KT_PREFILL_KV_LEND="${KT_PREFILL_KV_LEND:-1}" KT_PREFILL_PLAN="${KT_PREFILL_PLAN:-1}"
  [[ "$swa" == swa ]] && export KT_PREFILL_SWA_LEND="${KT_PREFILL_SWA_LEND:-1}"
  KT_KV_HOT_PLACEMENT="$hot"
  echo "hot lend: up to $KT_KV_HOT_EXTRA experts a layer past the $n resident ones (auto: as many as the free memory holds), in the KV pool's free pages (${KT_EXPERT_PLACEMENT:-$hot})" >&2
}
