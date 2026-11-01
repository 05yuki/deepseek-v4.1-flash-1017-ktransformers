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
# shellcheck source=../lib/server-up.sh
source "$here/lib/server-up.sh"
server_up || exit 1

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
