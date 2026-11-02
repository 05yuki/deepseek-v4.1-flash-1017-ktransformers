#!/usr/bin/env bash
# A qualification report for this machine, to paste into an issue or a pull request: starts ./serve.sh,
# checks the server end to end, measures the README's numbers, stops the server, and writes one Markdown
# file.
#
#   MODEL=<checkpoint dir> tools/qualify.sh [report.md]
#
# Checks (pass or fail):
#   answer      a chat request comes back with text
#   tool calls  tools/tool-call-check.py
#   NLL         tools/nll.py's 24 pieces against the checkpoint's reference implementation's scores
#               (tools/ref-nll.py): within QUALIFY_NLL_TOL (0.01) nats a token. The scores are
#               tools/nll-ref-1017.json, the reference run on an RTX 5070 Ti, or QUALIFY_REF=<file> from a
#               run on this machine. Launches of one build differ by about 0.002; the reference run on an
#               RTX 4080 SUPER scored 0.004 below the one on the RTX 5070 Ti.
#   cache       the same 35,000-token prompt twice: the second reads it from the prefix cache
# Measured, before the cache check: decode by text (tools/bench.py, --other-text) and the 35K prefill.
#
# About 15 minutes after the server is up; the first launch of a setup compiles kernels and may calibrate the
# prefill chunk first (up to an hour; a shipped hint for the GPU model and count skips the calibration). The report names the hardware and software: no host name, user,
# path, address or serial number. Exit status 0 when every check passed.
set -Eeuo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ -n "${MODEL:-}" ]] || { echo "set MODEL to the checkpoint directory" >&2; exit 2; }
port="${SGLANG_PORT:-8080}"
url="http://127.0.0.1:$port"
py="$here/.venv/bin/python"
out="${1:-$here/qualify-$(date +%Y%m%d-%H%M).md}"
work="$(mktemp -d)"
tol="${QUALIFY_NLL_TOL:-0.01}"
ref="${QUALIFY_REF:-$here/tools/nll-ref-1017.json}"

# what may name the machine, replaced in everything the report quotes
scrub_args=(-e "s#$(cd "$MODEL" && pwd)#<model>#g" -e "s#$here#<repo>#g" -e "s#$HOME#~#g")
for name in "$(hostname)" "$(id -un)"; do  # names short enough to be common words stay
  (( ${#name} >= 3 )) && scrub_args+=(-e "s#$name#<name>#g")
done
scrub() { sed "${scrub_args[@]}"; }

checks=()
pass() { checks+=("| $1 | pass | ${2//|/\\|} |"); }
fail() { checks+=("| $1 | **fail** | ${2//|/\\|} |"); failed=1; }
failed=0

# shellcheck source=../lib/server-up.sh
source "$here/lib/server-up.sh"
t0=$(date +%s)
server_up > "$work/launch.out" 2>&1 || { scrub < "$work/launch.out" | tail -20 >&2; exit 1; }
up_s=$(( $(date +%s) - t0 ))
sleep 10

key_header=()
[[ -n "${SGLANG_API_KEY:-}" ]] && key_header=(-H "Authorization: Bearer $SGLANG_API_KEY")

echo "answer"
reply="$(curl -s --max-time 600 "${key_header[@]}" -H "Content-Type: application/json" "$url/v1/chat/completions" \
  -d '{"messages":[{"role":"user","content":"Name three primary colors."}],"max_tokens":64,"temperature":1.0,"top_p":1.0}')"
text="$("$py" -c 'import json,sys
d = json.loads(sys.argv[1]); m = d["choices"][0]["message"]
print((m.get("content") or "") + (m.get("reasoning_content") or ""))' "$reply" 2>/dev/null || true)"
if [[ -n "${text//[[:space:]]/}" ]]; then pass answer "${#text} characters"; else fail answer "no text: ${reply:0:120}"; fi

echo "tool calls"
if "$py" "$here/tools/tool-call-check.py" "$url" > "$work/tools.out" 2>&1; then
  pass "tool calls" "$(tail -1 "$work/tools.out")"
else
  fail "tool calls" "$(grep -a -m3 -i "fail" "$work/tools.out" | tr '\n' ' ' | cut -c1-200)"
fi

echo "NLL"
if URL="$url" "$py" "$here/tools/nll.py" score "$MODEL" "$work/served.json" > "$work/nll.out" 2>&1; then
  "$py" "$here/tools/nll.py" compare "$ref" "$work/served.json" > "$work/nll-cmp.out"
  diff_line="$(grep -a "minus" "$work/nll-cmp.out" || true)"
  d="$(sed -n 's/.*: \([-+][0-9.]*\) nats a token.*/\1/p' <<< "$diff_line")"
  served="$(sed -n 's/.*mean NLL \([0-9.]*\).*/\1/p' "$work/nll.out")"
  msg="served $served, minus reference ${d}${diff_line#*nats a token}"
  if "$py" -c "import sys; sys.exit(0 if abs(float('$d')) <= float('$tol') else 1)"; then
    pass NLL "$msg"
  else
    fail NLL "$msg (tolerance $tol)"
  fi
else
  fail NLL "$(tail -1 "$work/nll.out" | cut -c1-200)"
fi

echo "bench"
"$py" "$here/tools/bench.py" --url "$url" ${SGLANG_API_KEY:+--api-key "$SGLANG_API_KEY"} > "$work/bench.out" 2>&1 || true
"$py" "$here/tools/bench.py" --url "$url" ${SGLANG_API_KEY:+--api-key "$SGLANG_API_KEY"} --other-text > "$work/other.out" 2>&1 || true

echo "prefix cache"
if "$py" - "$url" "${SGLANG_API_KEY:-}" > "$work/cache.out" 2>&1 <<'PY'
import json, random, sys, time, urllib.request
url, key = sys.argv[1], sys.argv[2]
ids = [random.Random(4242).randrange(1000, 129280) for _ in range(35000)]
headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}
times, cached = [], []
for _ in range(2):
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        url + "/generate", json.dumps({"input_ids": ids, "sampling_params": {"max_new_tokens": 1}}).encode(),
        headers), timeout=1800))
    times.append(time.time() - t)
    cached.append(r["meta_info"].get("cached_tokens") or 0)
print("first %.1f s, again %.1f s with %d of 35000 tokens from the cache" % (times[0], times[1], cached[1]))
sys.exit(0 if cached[1] >= 30000 else 1)
PY
then pass "prefix cache" "$(tail -1 "$work/cache.out")"; else fail "prefix cache" "$(tail -1 "$work/cache.out" | cut -c1-200)"; fi

slots="$(grep -a -o "lend 1: [^;]*" "$log" | head -1 | sed -n 's/.* \([0-9]*\) slots .*/\1/p' || true)"
# the GPUs the server saw, read while it still runs (an idle GeForce drops its link generation);
# nvidia-smi ignores CUDA_VISIBLE_DEVICES, -i takes its indices or UUIDs
gpus="$(nvidia-smi ${CUDA_VISIBLE_DEVICES:+-i "$CUDA_VISIBLE_DEVICES"} --format=csv,noheader \
  --query-gpu=name,compute_cap,memory.total,driver_version,pcie.link.gen.current,pcie.link.width.current,pcie.link.gen.max,pcie.link.width.max \
  2>/dev/null | sed -E 's/^/- GPU: /; s/, ([0-9]+), ([0-9]+), ([0-9]+), ([0-9]+)$/, PCIe gen \1 x\2 (max gen \3 x\4)/')"
server_stop

# the machine
cpu="$(LC_ALL=C lscpu | sed -n 's/^Model name: *//p' | head -1)"
sockets="$(LC_ALL=C lscpu | sed -n 's/^Socket(s): *//p')"
cores="$(LC_ALL=C lscpu | sed -n 's/^Core(s) per socket: *//p')"
numa="$(LC_ALL=C lscpu | sed -n 's/^NUMA node(s): *//p')"
isa="$(grep -o -w -E 'avx2|avx512f|avx512_vnni|avx512_bf16|amx_tile' /proc/cpuinfo | sort -u | paste -sd' ' || true)"
ram="$(awk '/MemTotal/ {printf "%.0f", $2 / 1048576}' /proc/meminfo)"
os="$(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME"), kernel $(uname -r)"
sw="$("$py" -c 'import platform, torch, flashinfer
print("Python %s, torch %s (CUDA %s), FlashInfer %s" % (platform.python_version(), torch.__version__, torch.version.cuda, flashinfer.__version__))' 2>/dev/null || echo unknown)"
# the toolkit the server used: the system's, else the pip wheels' under build/cuda-home (lib/cuda-home.sh), else none
nvcc_bin="${CUDA_HOME:-/usr/local/cuda}/bin/nvcc"
[[ -x "$nvcc_bin" ]] || nvcc_bin="$here/build/cuda-home/bin/nvcc"
[[ -x "$nvcc_bin" ]] || nvcc_bin="$(command -v nvcc || true)"
nvcc="$( { [[ -n "$nvcc_bin" ]] && "$nvcc_bin" --version; } 2>/dev/null | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -1 || true)"
[[ "$nvcc_bin" == "$here/build/cuda-home/bin/nvcc" && -n "$nvcc" ]] && nvcc="$nvcc (pip wheels, build/cuda-home)"
recipe="$(git -C "$here" describe --tags --always --dirty 2>/dev/null || echo "not a git checkout")"
sglang="$(sed -n 's/^SGLANG_COMMIT=\([0-9a-f]\{8\}\).*/\1/p' "$here/setup.sh")"
index_sha="$(sha256sum "$MODEL/model.safetensors.index.json" 2>/dev/null | cut -c1-12)"

{
  echo "# Qualification report"
  echo
  echo "$(date -u +%Y-%m-%d), \`tools/qualify.sh\` at recipe \`$recipe\` (SGLang \`$sglang\`)."
  echo
  echo "## Machine"
  echo
  echo "$gpus"
  echo "- CPU: $cpu, $sockets socket(s) x $cores cores, $numa NUMA node(s); $isa"
  echo "- RAM: $ram GiB"
  echo "- OS: $os"
  echo "- $sw; CUDA toolkit ${nvcc:-none found}"
  echo "- checkpoint: model.safetensors.index.json sha256 \`${index_sha:-missing}\`"
  echo
  echo "## Checks"
  echo
  echo "| check | result | detail |"
  echo "|---|---|---|"
  printf '%s\n' "${checks[@]}"
  echo
  echo "## Measured"
  echo
  echo "- server up after $up_s s; hot slots at the first lend: ${slots:-none logged}"
  echo '```'
  grep -a -E "^decode median|^prefill," "$work/bench.out" || echo "bench.py gave no result"
  grep -a -E "^decode," "$work/other.out" || echo "bench.py --other-text gave no result"
  echo '```'
  echo
  echo "## Launch"
  echo
  echo '```'
  grep -a -E "^GPUs |^prefix cache:|^patch tier|^tier-1 features:|^prefill lend: chunk|^chunk [0-9]+ from" "$work/launch.out" |
    sed 's/ from the hint .*/ from the hint/' || true
  echo '```'
} | scrub > "$out"

rm -rf "$work"
echo "report: $out"
(( failed == 0 )) && echo "all checks passed" || { echo "a check failed (see the report)"; exit 1; }
