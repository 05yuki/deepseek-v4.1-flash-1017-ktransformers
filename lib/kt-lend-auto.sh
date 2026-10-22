# Sourced by a launcher under KT_PREFILL_LEND=1 (sglang/srt/layers/moe/kt_lend.py in the
# tier-1 prefill-lend patch): picks the prefill chunk from what the server measured,
# calibrating once when there is nothing to go on.
#
#   kt_lend_auto <launcher> <chunk env var> <port> <key part>...
#
# Sets KT_PREFILL_LEND_HINT (prefix of the per-rank hint files the server
# writes, keyed by the key parts: model, context, mem fraction, GPUs, hot
# experts, ...) and KT_LEND_CHUNK:
#   - the chunk env var when the user gave one;
#   - else the smallest of the ranks' hints from an earlier launch (one file
#     per rank; one rank is enough, so a single-GPU server works too);
#   - else a calibration: the launcher started once in the background with
#     KT_PREFILL_LEND_CALIBRATING=1 at a first guess from the card size, one
#     ~50K-token prompt, the server stopped; out of memory halves the guess
#     (down to 2048). KT_PREFILL_LEND_CALIBRATE=0 skips it (chunk 8192).
# The launcher must print "Run directory: <dir>" (or "log directory: <dir>")
# and leave <dir>/server.pid
# and <dir>/server.log in --background mode.

_kt_lend_read_hint() {
  python3 - "$KT_PREFILL_LEND_HINT" <<'PY'
import glob, json, sys
hints = [json.load(open(f))["chunk"] for f in sorted(glob.glob(sys.argv[1] + "-rank*.json"))]
print(min(hints) if hints else "")
PY
}

_kt_lend_auto_core() {
  local launcher="$1" chunk_var="$2" port="$3"
  shift 3
  local key extra=()
  # the plan's row blocks (KT_GPU_STREAM_ROW_BLOCK: the CUTLASS MoE workspace
  # sized for a block of rows, not the chunk) on by default with it
  if [[ "${KT_PREFILL_PLAN:-0}" == 1 ]]; then
    export KT_GPU_STREAM_ROW_BLOCK="${KT_GPU_STREAM_ROW_BLOCK:-4096}"
  fi
  [[ "${KT_GPU_STREAM_ROW_BLOCK:-0}" != 0 ]] && extra+=("rowblock$KT_GPU_STREAM_ROW_BLOCK")
  # the KV pool's loans change what a window has: their hints are kept apart
  [[ "${KT_KV_VMM:-0}" == 1 ]] && extra+=("kvvmm")
  [[ "${KT_PREFILL_KV_LEND:-0}" == 1 ]] && extra+=("kvlend")
  [[ "${KT_PREFILL_PLAN:-0}" == 1 ]] && extra+=("plan")
  [[ "${KT_PREFILL_SWA_LEND:-0}" == 1 ]] && extra+=("swalend")
  key="$(printf '%s|' "$@" "${extra[@]}" | sha1sum | cut -c1-16)"
  # KT_LEND_PRINT_KEY=1: print this setup's hint key and stop the launcher (to carry
  # hints to a new key, e.g. a placement with the same residents)
  if [[ "${KT_LEND_PRINT_KEY:-0}" == 1 ]]; then
    echo "kt-lend hint key $key"
    exit 0
  fi
  mkdir -p "$HOME/.cache/kt-lend"
  export KT_PREFILL_LEND_HINT="$HOME/.cache/kt-lend/chunk-$key"
  local user="${!chunk_var:-}"
  if [[ -n "$user" ]]; then
    KT_LEND_CHUNK="$user"
    return 0
  fi
  local hint
  hint="$(_kt_lend_read_hint)"
  if [[ -n "$hint" ]]; then
    echo "chunk $hint from the hint $KT_PREFILL_LEND_HINT-rank*.json" >&2
    KT_LEND_CHUNK="$hint"
    return 0
  fi
  if [[ "${KT_PREFILL_LEND_CALIBRATE:-1}" != 1 || -n "${KT_PREFILL_LEND_CALIBRATING:-}" ]]; then
    KT_LEND_CHUNK=8192
    return 0
  fi
  local cal_log="$HOME/.cache/kt-lend/calibrate-$key.log"
  : >"$cal_log"
  local vram_mb cal_chunk cal_run cal_pid cal_ok
  vram_mb="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | head -1)"
  cal_chunk=16384
  (( vram_mb < 15000 )) && cal_chunk=8192
  (( vram_mb > 23000 )) && cal_chunk=24576
  # no wider than the launcher allows (V4.1: the widest chunk that keeps the KV pool)
  [[ -n "${KT_PREFILL_LEND_MAX_CHUNK:-}" ]] && (( cal_chunk > KT_PREFILL_LEND_MAX_CHUNK )) && \
    cal_chunk="$KT_PREFILL_LEND_MAX_CHUNK"
  # KT_PREFILL_LEND_PROBE_CONTEXT=<tokens> (a tree with sglang-kt-lend-probe.py):
  # the calibration server measures the chunk with the lend probe at prefix 0,
  # half the context and the context's end before it serves, instead of a
  # ~50K-token prompt: what the chunk costs at the longest prompt the server
  # will take, in a few minutes
  local probe=""
  if [[ -n "${KT_PREFILL_LEND_PROBE_CONTEXT:-}" ]]; then
    probe="@0,$(( KT_PREFILL_LEND_PROBE_CONTEXT / 2 )),$(( KT_PREFILL_LEND_PROBE_CONTEXT - 8 * cal_chunk ))"
  fi
  while (( cal_chunk >= 2048 )); do
    echo "no chunk hint for this setup yet: calibration launch at chunk $cal_chunk (${vram_mb} MiB cards; $cal_log)" >&2
    [[ -n "$probe" ]] && probe="@${probe#*@}" && probe="$cal_chunk$probe"
    env KT_PREFILL_LEND_CALIBRATING=1 "$chunk_var=$cal_chunk" ${probe:+KT_PREFILL_LEND_PROBE="$probe"} \
      bash "$launcher" --background >>"$cal_log" 2>&1 || {
      echo "calibration launch failed to start; see $cal_log" >&2; return 5; }
    cal_run="$(sed -n -e 's/^Run directory: //p' -e 's/^log directory: //p' "$cal_log" | tail -1)"
    cal_pid="$(cat "$cal_run/server.pid" 2>/dev/null)"
    [[ -n "$cal_pid" ]] || { echo "calibration launch left no pid; see $cal_log" >&2; return 5; }
    cal_ok=0
    for _ in $(seq 1 360); do
      kill -0 "$cal_pid" 2>/dev/null || break
      if curl -sf -m 3 "http://127.0.0.1:$port/health" >/dev/null; then cal_ok=1; break; fi
      sleep 10
    done
    if [[ "$cal_ok" == 1 && -z "$probe" ]]; then
      python3 - "$port" <<'PY' >>"$cal_log" 2>&1 || true
import json, secrets, sys, time, urllib.request
para = ("灯りの落ちた編集部で、彼女は原稿の束を繰っていた。窓の外は雨で、アスファルトの匂いが"
        "換気口から上がってくる。赤を入れる手は速い。迷った箇所には印だけ置いて先へ進む。\n")
text = "識別子 " + secrets.token_hex(24) + "\n" + para * 900 + "\n以上を一行で要約して。"
body = json.dumps({"text": text, "sampling_params": {"max_new_tokens": 4, "temperature": 0.0}}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{sys.argv[1]}/generate", body, {"Content-Type": "application/json"})
t = time.time()
r = json.load(urllib.request.urlopen(req, timeout=3600))
print(f"calibration prompt: {r['meta_info']['prompt_tokens']} tokens in {time.time() - t:.1f} s")
PY
      for _ in $(seq 1 60); do
        if ls "$KT_PREFILL_LEND_HINT"-rank*.json >/dev/null 2>&1; then
          sleep 10  # the other ranks write theirs at the end of the same window
          break
        fi
        sleep 2
      done
    fi
    kill -TERM "$cal_pid" 2>/dev/null
    for _ in $(seq 1 120); do kill -0 "$cal_pid" 2>/dev/null || break; sleep 2; done
    kill -KILL "$cal_pid" 2>/dev/null || true
    # the GPUs, the port and the server processes, as the launchers check them
    for _ in $(seq 1 90); do
      nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . || \
        ss -ltn | grep -q ":$port " || pgrep -f 'sglang[. ](serve|launch_server)' >/dev/null || \
        pgrep '^sglang::' >/dev/null || break
      sleep 5
    done
    hint="$(_kt_lend_read_hint)"
    [[ -n "$hint" ]] && break
    if grep -q -E "OutOfMemoryError|out of memory|cannot come back" "$cal_run/server.log" 2>/dev/null; then
      echo "calibration at chunk $cal_chunk ran out of memory; halving" >&2
      cal_chunk=$((cal_chunk / 2))
    else
      break
    fi
  done
  if [[ -n "$hint" ]]; then
    echo "calibrated: chunk $hint (calibration ran at $cal_chunk)" >&2
    KT_LEND_CHUNK="$hint"
  else
    echo "calibration left no hint; starting at chunk 2048 (see $cal_log)" >&2
    KT_LEND_CHUNK=2048
  fi
}

# The KV guard (KT_PREFILL_LEND_KV_CAP=<the max_total_tokens the launcher asks
# for>): the lend takes nothing from the KV pool. Where a wider chunk widens the
# SWA pool (DeepSeek V4, MiMo), that comes out of the full pool; so when the
# last server of this setup got a smaller full pool than the reference, the
# chunk it ran is remembered as one that cuts the KV pool and this launch goes
# 1024 below it. The reference is the cap, or, where even chunk 2048 does not
# reach it (memory, not the lend: MiMo on one card), the full pool at 2048.
# Kept in <hint>-kv.json; the server rewrites the rank files every window.
_kt_lend_kv_guard() {
  [[ -n "${KT_PREFILL_LEND_KV_CAP:-}" ]] || return 0
  local out
  out="$(python3 - "$KT_PREFILL_LEND_HINT" "$KT_LEND_CHUNK" "$KT_PREFILL_LEND_KV_CAP" <<'EOF'
import glob, json, sys
prefix, chunk, cap = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
side_path = prefix + "-kv.json"
try:
    side = json.load(open(side_path))
except Exception:
    side = {}
recs = []
for f in glob.glob(prefix + "-rank*.json"):
    try:
        recs.append(json.load(open(f)))
    except Exception:
        pass
full = min((r.get("full_tokens", 0) for r in recs), default=0)
ran = max((r.get("current_chunk", 0) for r in recs), default=0)
ref = side.get("ref", cap)
note = ""
if full > 0 and ran > 0 and full < ref * 0.9995:
    if ran <= 2048:
        side["ref"] = ref = full
        note = f"full pool {full} below {cap} even at chunk {ran}: memory, not the lend; {full} is the reference now"
    else:
        side["cut_at"] = min(side.get("cut_at", ran), ran)
        note = f"chunk {ran} left the full pool at {full} of {ref}: the KV pool comes first"
if "cut_at" in side:
    chunk = min(chunk, max(2048, side["cut_at"] - 1024))
json.dump(side, open(side_path, "w"))
print(chunk)
print(note)
EOF
)" || return 0
  local new note
  new="$(sed -n 1p <<<"$out")"
  note="$(sed -n 2p <<<"$out")"
  [[ -n "$note" ]] && echo "KV guard: $note" >&2
  if [[ -n "$new" && "$new" != "$KT_LEND_CHUNK" ]]; then
    echo "KV guard: chunk $KT_LEND_CHUNK -> $new" >&2
    KT_LEND_CHUNK="$new"
  fi
}

kt_lend_auto() {
  _kt_lend_auto_core "$@" || return $?
  [[ -n "${!2:-}" ]] || _kt_lend_kv_guard
}

# The DeepSeek V4 / MiMo SWA pool is full_tokens x ratio and must hold one
# chunk plus a page, or the server refuses to start or caps every prefill
# batch below the chunk: the ratio for a chunk, never below the launcher's own.
# The full pool is the cap only when memory allows. Where it does not (MiMo on
# one card got 97,699 of 131072, so its SWA pool held each batch at 12.1K rows
# of 15360), the KV pool is kept and the chunk is what gives: the launcher's
# default. KT_PREFILL_LEND_SWA_FIT=1 sizes the ratio from the full pool an
# earlier launch recorded instead, and the chunk wins: there MiMo went to
# 13.9K rows and 40K prompts 1,107 -> 1,331 tok/s, but the full pool to 81,552
# (39 sliding layers to 9 full ones: each SWA token costs several full ones).
kt_lend_swa_ratio() {  # chunk max_total_tokens ratio
  # the SWA lend (kt_prefill_plan_setup): the budget holds a small chunk (the server
  # budgets at the same one, and DeepSeek V4 refuses a pool under the chunk it is
  # sized at); the server widens the pool for the cap outside the budget and lends it
  local chunk="$1"
  [[ -n "${KT_PREFILL_SWA_TOKENS:-}" ]] && chunk="${KT_PREFILL_SWA_SIZING_CHUNK:-2048}"
  python3 - "$chunk" "$2" "$3" "${KT_PREFILL_LEND_HINT:-}" "${KT_PREFILL_LEND_SWA_FIT:-0}" <<'PY'
import glob, json, sys
c, m, r, hint, fit = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), sys.argv[4], sys.argv[5] == "1"
full = [json.load(open(f)).get("full_tokens", 0) for f in glob.glob(hint + "-rank*.json")] if hint and fit else []
full = [f for f in full if f > 0]
if full:
    m = min(m, int(min(full) * 0.97))
print(f"{max(r, (c + 512) * 1.02 / m):.4f}")
PY
}

# KT_PREFILL_PLAN=1 (sglang/srt/layers/moe/kt_prefill_plan.py, with KT_KV_VMM=1 and
# KT_PREFILL_KV_LEND=1): the chunk planned per request. Called after kt_lend_auto:
# the launch chunk becomes the cap (KT_PREFILL_PLAN_CAP, 36864) and the room and
# the cost a row come from the lend's measurements of this setup
# (lib/kt-prefill-plan-constants.py over the hint files); without them the plan
# is off for this launch. KT_PREFILL_SWA_LEND=1 (a hybrid SWA model): the
# sliding-window pool is sized for the cap (KT_PREFILL_SWA_TOKENS) outside the KV
# budget, its free pages unmapped outside prefill, and kt_lend_swa_ratio keeps the
# launcher's own ratio. Sets KT_LEND_CHUNK. A launcher of a hybrid SWA model
# passes "swa": without the SWA lend a cap past the lend's chunk would widen the
# SWA pool out of the KV budget, so the cap stays the lend's chunk there.
kt_prefill_plan_setup() {
  [[ "${KT_PREFILL_PLAN:-0}" == 1 && -z "${KT_PREFILL_LEND_CALIBRATING:-}" && -n "${KT_PREFILL_LEND_HINT:-}" ]] || return 0
  local lib="${KT_KV_VMM_LIB:-${here:-.}/build/libkv_vmm.so}"
  if [[ ! -f "$lib" ]]; then
    echo "$lib missing (build kv_vmm.c, or set KT_KV_VMM_LIB): prefill plan off" >&2
    export KT_PREFILL_PLAN=0 KT_PREFILL_SWA_LEND=0
    return 0
  fi
  local cap="${KT_PREFILL_PLAN_CAP:-36864}" pf_free pf_row pf_cost pf_delta
  if [[ -z "${KT_PREFILL_PLAN_FREE_MB:-}" && -z "${KT_PREFILL_PLAN_ROW_KB:-}" && -z "${KT_PREFILL_PLAN_COST:-}" ]]; then
    read -r pf_free pf_row pf_cost pf_delta <<<"$(python3 "$(dirname "${BASH_SOURCE[0]}")/kt-prefill-plan-constants.py" "$KT_PREFILL_LEND_HINT" "$cap")"
    if [[ -n "${pf_free:-}" ]]; then
      export KT_PREFILL_PLAN_FREE_MB="$pf_free" KT_PREFILL_PLAN_ROW_KB="$pf_row"
      # a cost fitted from the windows a setup's serving recorded is opt-in: it is fitted on
      # the lend's allocated bytes, torch's reserved memory runs up to 2 GB above them, and a
      # fitted cost planned 131K in 26,624-row chunks on a second launch and ran out of
      # memory (10-22); the prior below plans that prompt in 11,264-row chunks
      [[ "${pf_cost:--}" != - && "${KT_PREFILL_PLAN_FIT:-0}" == 1 ]] && export KT_PREFILL_PLAN_COST="$pf_cost"
      [[ "${pf_delta:-0}" != 0 ]] && export KT_PREFILL_PLAN_DELTA_MB="$pf_delta"
    fi
  fi
  # no fitted cost (the default, or a new setup's first launches with KT_PREFILL_PLAN_FIT=1):
  # the model's prior, whose sequence term keeps a long prompt's chunks small (a row-only
  # cost planned 13,312 rows at 1M and the indexer's scores ran out of memory, 10-21)
  if [[ "${KT_PREFILL_PLAN_ROW_KB:-0}" != 0 && -z "${KT_PREFILL_PLAN_COST:-}" && -n "${KT_PREFILL_PLAN_COST_PRIOR:-}" ]]; then
    export KT_PREFILL_PLAN_COST="$KT_PREFILL_PLAN_COST_PRIOR"
    echo "prefill plan: the prior cost $KT_PREFILL_PLAN_COST (KT_PREFILL_PLAN_FIT=1 uses this setup's fitted cost once it has one)" >&2
  fi
  if [[ "${KT_PREFILL_PLAN_ROW_KB:-0}" == 0 && -z "${KT_PREFILL_PLAN_COST:-}" ]]; then
    export KT_PREFILL_PLAN=0
    echo "prefill plan: no measurement of this setup yet, off for this launch (chunk $KT_LEND_CHUNK)" >&2
    return 0
  fi
  # the lend's chunk stays the fallback before the first window measures anything
  export KT_PREFILL_PLAN_SAFE="${KT_PREFILL_PLAN_SAFE:-$KT_LEND_CHUNK}"
  if [[ "${1:-}" == swa && "${KT_PREFILL_SWA_LEND:-0}" != 1 ]] && (( cap > KT_LEND_CHUNK )); then
    cap="$KT_LEND_CHUNK"
  fi
  KT_LEND_CHUNK="$cap"
  export KT_PREFILL_PLAN_MARGIN_MB="${KT_PREFILL_PLAN_MARGIN_MB:-128}"
  if [[ "${KT_PREFILL_SWA_LEND:-0}" == 1 ]]; then
    export KT_PREFILL_SWA_TOKENS="${KT_PREFILL_SWA_TOKENS:-$(( (cap + 512) * 102 / 100 ))}"
  fi
  echo "prefill plan: cost ${KT_PREFILL_PLAN_COST:-${KT_PREFILL_PLAN_ROW_KB} KiB a row}, chunk cap $cap${KT_PREFILL_SWA_TOKENS:+, SWA pool $KT_PREFILL_SWA_TOKENS tokens lent outside prefill}" >&2
}
