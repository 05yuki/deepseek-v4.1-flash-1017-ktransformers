# Start ./serve.sh in the background and wait until the server is up; sourced by tools/check.sh and
# tools/qualify.sh, which set `here` (the repository) and `port` first.
#
#   server_up    sets run_dir, pid and log; stops the server on exit; returns 1 (with a message) when
#                it stopped, failed or stalled before it was up
#
# The first launch of a setup compiles kernels and may calibrate the prefill chunk, so it waits up to an
# hour. A stall is the log silent for CHECK_STALL_S (900 s; the first launch compiles kernels in long quiet
# spells). A Traceback alone is not a failure: the warmup logs one on a healthy start.

server_stop() {  # and wait for it to go, so the GPUs are free when this returns
  [[ -n "${pid:-}" ]] || return 0
  kill "$pid" 2>/dev/null || true
  for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || return 0; sleep 2; done
  echo "server $pid still running after 2 minutes" >&2
}

server_up() {
  local log_dir before cur stall
  log_dir="$here/logs/$(basename "$(cd "$MODEL" && pwd)")"
  before="$(readlink -f "$log_dir/current" 2>/dev/null || true)"
  "$here/serve.sh" --background || { echo "serve.sh failed (see its output above)" >&2; return 1; }
  run_dir=""
  for _ in $(seq 1 120); do
    cur="$(readlink -f "$log_dir/current" 2>/dev/null || true)"
    if [[ -n "$cur" && "$cur" != "$before" && -f "$cur/server.pid" ]]; then run_dir="$cur"; break; fi
    sleep 5
  done
  [[ -n "$run_dir" ]] || { echo "serve.sh started no server (see its output above)" >&2; return 1; }
  pid="$(cat "$run_dir/server.pid")"
  log="$run_dir/server.log"
  trap server_stop EXIT

  stall="${CHECK_STALL_S:-900}"
  for _ in $(seq 1 720); do
    grep -a -q "The server is fired up" "$log" 2>/dev/null && return 0
    if ! kill -0 "$pid" 2>/dev/null ||
      grep -a -q "Scheduler hit an exception\|error: unrecognized arguments\|error: argument" "$log" 2>/dev/null; then
      echo "the server stopped before it was up; see $log" >&2
      return 1
    fi
    if (( $(date +%s) - $(stat -c %Y "$log" 2>/dev/null || date +%s) > stall )); then
      echo "the server log has been silent for over $stall s; stopping it. See $log" >&2
      return 1
    fi
    sleep 5
  done
  echo "not up after an hour; see $log" >&2
  return 1
}
