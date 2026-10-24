#!/usr/bin/env bash
# Source after the ordinary serving environment, before serve.sh.
# Requires SESSION_CONCURRENCY_PATCH=1 at setup time. The ordinary profile
# remains at one active request with retained-session admission disabled.
# Separate four-row hardware profile: cache flush restores SWA pages before
# upstream full-KV clear releases hot loans. Keep SWA resident for session close
# and cold replay. The 3 GiB spare floor keeps GPU memory outside hot-expert
# lending for continuation workspace; it is not memory reserved for hot slots.
# This is a profile workaround, not a change to the optional admission patch.
export KT_PREFILL_SWA_LEND=0
export KT_KV_HOT_SPARE_FLOOR_MB=3072
export KT_STREAMING_SESSION_ADMISSION=1
export SGLANG_EXTRA_ARGS="${SGLANG_EXTRA_ARGS:-} --enable-streaming-session --max-running-requests 4 --prefill-max-requests 4 --max-queued-requests 4 --schedule-policy fcfs --stream-interval 1"
