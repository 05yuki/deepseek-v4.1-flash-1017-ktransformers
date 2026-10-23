# Opt-in retained streaming sessions

The ordinary recipe keeps one active request, disables general radix-cache
reuse, and does not enable streaming sessions. This change leaves those
defaults intact. To build the historical retained-session admission fix, set
`SESSION_CONCURRENCY_PATCH=1` with `TIER=1` or `TIER=2` and
`SGLANG_FEATURES=all` during `setup.sh`. Partial tier-1 selections are not
part of this port's verified source stack and are rejected.
The flag selects one additional SGLang source patch after the existing tier
patches; the setup marker includes its bytes, so a changed selection rebuilds
the patched SGLang tree. The patch can be installed while its behavior remains
off: `KT_STREAMING_SESSION_ADMISSION` defaults to `0`.

For the separate opt-in serving profile, source
`profiles/retained-session-4.sh` after the normal environment and before
`serve.sh`. It sets the admission gate to `1` and appends the historical
`--enable-streaming-session`, four active/prefill and four queued requests,
FCFS scheduling, and `--stream-interval 1` arguments. It does not silently
turn on the general radix cache. Keep the usual model, context, KV and host
memory settings explicit in the deployment profile; this script does not
claim that four active requests fit every host.

The SGLang patch handles retained row admission when idle sessions hold all
request-pool rows, evicts the oldest idle row for a fifth row-needing request,
never evicts an in-flight session, and preserves the running turn's ownership
when an overlapping turn is rejected. A later continuation of an evicted
session re-prefills from its token history. The admission path is selected
only when both the environment gate is on and at least one streaming session
is open. Source changes in the session/cache helpers are otherwise inert:
reclaim demand starts at zero and only the selected scheduler path sets it.

Run the CPU-only tests after setup with
`python -B tools/test_retained_session_admission.py`. These execute the
relevant methods from the patched source AST with fake rows and no model. The
old r4 campaign exercised inheritance at 1/2/4, fifth-session eviction,
overlap rejection, abort/recovery and queued cancellation on an earlier
source tree. This port has not yet been run end to end on 00fd12a or under
the eight-node A/B hardware profile. Before enabling it in service, repeat
those lifecycle cases, explicit FIFO/503 queue limits, and N1/N2/N4
throughput/quality measurements with a bounded four-request memory profile.
