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

With the pinned non-speculative defaults and graph padding enabled, four
running requests resolve to decode graph maximum `4` and batches `[1, 2, 4]`.
An explicit graph JSON or convenience override takes precedence; use the
matching `--cuda-graph-max-bs-decode 4 --cuda-graph-bs-decode 1 2 4` values
if supplying such overrides for this profile.

The opt-in hardware profile also sets `KT_PREFILL_SWA_LEND=0` and
`KT_KV_HOT_SPARE_FLOOR_MB=3072`. SWA stays resident through session close and
cache flush, avoiding an upstream clear-order interaction where SWA pages
are remapped before hot expert loans are released. The larger spare floor
keeps more GPU memory outside hot-expert lending for continuation workspace.
This is a profile workaround, not a lender
fix; flush and cold-replay checks remain required. The ordinary recipe's
lending defaults and the admission patch's default-off gate are unchanged.

SGLang already provides concurrent scheduling, streaming sessions and queues.
This patch handles retained row admission when idle sessions hold all
request-pool rows, evicts the oldest idle row for a fifth row-needing request,
never evicts an in-flight session, and preserves the running turn's ownership
when an overlapping turn is rejected. A later continuation of an evicted
session re-prefills from its token history. The admission path is selected
only when both the environment gate is on and at least one streaming session
is open. Source changes in the session/cache helpers are otherwise inert:
reclaim demand starts at zero and only the selected scheduler path sets it.

Run the CPU-only tests after setup with
`python -B tools/test_retained_session_admission.py`. These execute the
relevant methods from the patched source AST with fake rows and no model.
The A5 correctness run on recipe `00fd12a` / SGLang `9c0f6d6`, one SM89 GPU
and eight NUMA nodes completed three retained turns at N1/N2/N4, fifth-row
admission with exactly one idle-row eviction, overlap rejection, active and
queued abort/recovery, FIFO order and HTTP 503 queue limits. It used the
SWA-off/spare-3072 profile above, four-row decode graphs, and the KV-policy
source patch with its TTL and HiCache policies disabled. The earlier A4
SWA-lending profile failed during cache flush and remains failed evidence.

The functional, numerical, source/resource and fixed-history reference reviews
are complete. Across 24 pieces (11,496 scored positions), A5 minus the checkpoint
reference is **+0.0098927643 nats/token**, with piece-level SE **0.0050181971**.
One of seven retained/cold replay rows (six distinct histories) changes its
selected token. Material probability changes also occur with unchanged winners,
including the separate 18-token recovery witness. These completed descriptive
checks establish neither numerical equivalence nor general quality or production
readiness. The fixed-history reference covers one candidate pair, not every
replay or incremental-session execution.

[Reviewed experimental A5 results and compact evidence](https://github.com/howl9003/deepseek-v4.1-flash-1017-ktransformers/blob/3f36082fc3021998d2a6a1e588b11f94050cbcf6/docs/benchmarks/retained-session-00fd12a-20261024/RESULTS.md)
preserve every replay, the independent reference, NLL/WikiText screening,
prior failure, exact hardware/profile and cleanup limits. Request timings come
from a correctness run with different settings from the canonical single-request
baseline, so they establish no isolated throughput or speedup claim.
The current `e8608a2` / SGLang `8b9f724` port has source/CPU checks only;
the older-pin model results do not establish runtime behavior on this pin.
