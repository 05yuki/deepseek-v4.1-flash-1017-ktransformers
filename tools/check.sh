#!/usr/bin/env bash
# Does this setup work? Starts ./serve.sh in the background, waits until the server is up (the first
# launch of a setup compiles kernels and may calibrate the prefill chunk, so up to an hour), sends one
# short chat request, checks that text comes back, and stops the server it started.
#
#   MODEL=<checkpoint dir> tools/check.sh
#
# Exit status 0 when an answer came back; the server log's path is printed either way.
set -Eeuo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ -n "${MODEL:-}" ]] || { echo "set MODEL to the checkpoint directory" >&2; exit 2; }
port="${SGLANG_PORT:-8080}"
log_dir="$here/logs/$(basename "$(cd "$MODEL" && pwd)")"
before="$(readlink -f "$log_dir/current" 2>/dev/null || true)"

"$here/serve.sh" --background
run_dir=""
for _ in $(seq 1 120); do
  cur="$(readlink -f "$log_dir/current" 2>/dev/null || true)"
  if [[ -n "$cur" && "$cur" != "$before" && -f "$cur/server.pid" ]]; then run_dir="$cur"; break; fi
  sleep 5
done
[[ -n "$run_dir" ]] || { echo "serve.sh started no server (see its output above)" >&2; exit 1; }
pid="$(cat "$run_dir/server.pid")"
log="$run_dir/server.log"
stop() {  # and wait for it to go, so the GPUs are free when this returns
  kill "$pid" 2>/dev/null || true
  for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || return 0; sleep 2; done
  echo "server $pid still running after 2 minutes" >&2
}
trap stop EXIT

# up, failed (the process gone, a scheduler exception, an argument error), or stalled:
# its log silent for CHECK_STALL_S (900 s; the first launch compiles kernels in long
# quiet spells). A Traceback alone is not a failure: the warmup logs one on a healthy start.
stall="${CHECK_STALL_S:-900}"
for _ in $(seq 1 720); do
  grep -a -q "The server is fired up" "$log" 2>/dev/null && break
  if ! kill -0 "$pid" 2>/dev/null ||
    grep -a -q "Scheduler hit an exception\|error: unrecognized arguments\|error: argument" "$log" 2>/dev/null; then
    echo "the server stopped before it was up; see $log" >&2
    exit 1
  fi
  if (( $(date +%s) - $(stat -c %Y "$log" 2>/dev/null || date +%s) > stall )); then
    echo "the server log has been silent for over $stall s; stopping it. See $log" >&2
    exit 1
  fi
  sleep 5
done
grep -a -q "The server is fired up" "$log" || { echo "not up after an hour; see $log" >&2; exit 1; }

key_header=()
[[ -n "${SGLANG_API_KEY:-}" ]] && key_header=(-H "Authorization: Bearer $SGLANG_API_KEY")
reply="$(curl -s --max-time 600 "${key_header[@]}" -H "Content-Type: application/json" \
  "http://127.0.0.1:$port/v1/chat/completions" \
  -d '{"messages":[{"role":"user","content":"Name three primary colors."}],"max_tokens":64,"temperature":1.0,"top_p":1.0}')"
text="$("$here/.venv/bin/python" -c 'import json,sys
d = json.loads(sys.argv[1]); m = d["choices"][0]["message"]
print((m.get("content") or "") + (m.get("reasoning_content") or ""))' "$reply" 2>/dev/null || true)"
if [[ -z "${text//[[:space:]]/}" ]]; then
  echo "no answer; the reply was: ${reply:0:300}" >&2
  echo "server log: $log" >&2
  exit 1
fi
printf 'answer: %s\n' "${text:0:200}"
echo "server log: $log"
echo "OK"
