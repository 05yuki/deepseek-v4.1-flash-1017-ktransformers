#!/usr/bin/env bash
# Source after the ordinary serving environment, before serve.sh.
# Requires SESSION_CONCURRENCY_PATCH=1 at setup time. The ordinary profile
# remains at one active request with retained-session admission disabled.
export KT_STREAMING_SESSION_ADMISSION=1
export SGLANG_EXTRA_ARGS="${SGLANG_EXTRA_ARGS:-} --enable-streaming-session --max-running-requests 4 --prefill-max-requests 4 --max-queued-requests 4 --schedule-policy fcfs --stream-interval 1"
