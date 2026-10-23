# Optional GPU glue and hot slots

This bundle adds the candidate's GPU-side glue, CPU handoff poller, service-core
affinity, and dynamic hot-slot machinery to the `00fd12a` recipe. It is a separate
patch stack on top of the common SM89/base serving changes. The source bundle is
off by default: `setup.sh` applies it only with `HOT_GLUE_PATCH=1`, and `serve.sh`
activates the glue only with `KT_GPU_GLUE=1`. The tested source selection is
`TIER=1 SGLANG_FEATURES=all`, matching the A/B recipe. `TIER=2` with all tier-1
features is also supported; setup rejects partial selections with this bundle.

The selected candidate source delta is `patches/sglang-optional/01-gpu-glue.patch`,
then `02-hot-slots.patch`, plus `patches/kt-kernel-gpu-glue-only.patch`. The native
patch leaves the already-common physical-NUMA `worker_pool.cpp` fix alone. The
bundle does not include the temporary preferred-memory-policy experiment: the
strict original native policy is common to both arms. `tools/glue/SOURCE_PINS.json`
records the ten selected SGLang files and three native files from the B runtime;
`verify_source_bundle.py` checks all thirteen hashes. Its native comparison
expects the common measured host-NUMA adapter described below, in addition to
the optional B patch; the adapter is not applied by this draft's setup selector.

The glue narrows index top-k work to live decode slots, fuses sparse-output merge
with the attention sink on SM89, packs the CPU handoff into one kernel, lets the
shared expert begin after CPU submission, and changes the native memop poller's
scan hint. Service-core affinity keeps scheduler threads off the 56 singleton
expert CPUs. Hot slots batch whole-part arena loads, defer publication until
transfer completion, and bound swaps with a byte cap, hysteresis, and dwell.

`KT_GPU_GLUE=1` defaults the live index top-k, SM89 fused merge, shared-buffer
submit ordering, handoff pack, and memop poll hint to on; the native task-queue
spin interval defaults to 2000 microseconds. Each setting can be overridden
individually. With `KT_GPU_GLUE=0`, the existing launch defaults apply. Dynamic
hot-slot ports have a separate master gate, `KT_KV_HOT_PORT=1`; enabling glue
alone does not activate the hot-slot port. The combined eight-node profile
enables both gates. An
explicit `KT_SERVICE_CPUS` requires the optional source bundle even when glue is
off. `profiles/eight-node-hot-glue.sh` contains the treatment-only values used on
the single-GPU, eight-NUMA host: 56 expert CPUs (seven per node), eight service
CPUs, dynamic hot-slot limits, and the measured hot-port selection. It assumes
the same common checkpoint, GGUF, placement, SM89 host adapters, and benchmark
scope settings as A. The profile alone is not a full replacement for those shared
inputs or for the reviewed A/B launch controller.

To check the source on an already-built candidate tree:

```bash
python tools/glue/verify_source_bundle.py \
  --sglang third_party/sglang --kt third_party/ktransformers
```

## Completed single-request comparison

The measured A baseline is recipe `00fd12a0b3e59cbf7dcc62b7341415fcdd3fc9a6`
with SGLang `9c0f6d6a610531bd16ad3513572e55226450b244`, SM89 enablement and the
common host/storage adapters. It already has upstream automatic hot lending and
dynamic resident replacement. B adds this entire bundle, changes queue spin
from 300 to 2,000 microseconds, and restricts service threads to
`7,15,23,31,39,47,55,63`. Both use 56 expert workers with strict placement on all
eight NUMA nodes. B's 512 MiB value limits replacement-copy traffic per rebalance
round; it is neither a residency cap nor the 1,024 MiB spare floor.

On one experimental 32 GiB RTX 4080 SUPER and two EPYC 7543 CPUs, TP=1,
max-running=1, radix disabled and a 1M configured cap:

| Measurement | A | B | Observed change |
|---|---:|---:|---:|
| Default p0–p3 decode median | 31.0 tok/s | 35.0 tok/s | +12.9% rate |
| First 35K prefill | 41.63 s | 42.08 s | +1.1% latency |
| Second 35K prefill | 29.65 s | 29.77 s | +0.4% latency |

All ten decode rows increased by 6.0–23.9% in this one ordered A→B pair. Keep
the slightly slower prefill rows and the complete comparison together.
[The public evidence draft](benchmarks/hot-glue-00fd12a-20261024-review-r2/PR_DRAFT.md)
includes every row, unchanged raw scores/stdout and source identities. Rounded
rates, stochastic output and one campaign per arm do not establish a stable
speedup or isolate a component's contribution.

B minus the independent checkpoint reference is +0.003621167 nats/token with
paired-piece SE 0.003984346; B minus A is +0.002636102 with SE 0.001694099 over
24 pieces. The WikiText B−A difference is +0.000788344 with chunk SE 0.001037606.
The owner accepted this exact fixed workload for an observed-comparison draft
and bounded feature trials, without a numerical-equivalence claim. Both arms passed exact upstream
`check.sh`, ten component groups, five tool-call cases and the same three
sequential retrieval checks at approximately 4K/16K/35K. Client-window memory
event deltas and observed swap were zero. A memory PSI was zero; B accumulated
1.624029 s some / 1.623210 s full over 749.566983 s (about 0.2166%). All positive
sampled PSI intervals were in the initial tool-call checks, with no later
increments in NLL, benchmarks, WikiText or retrieval. Zero memory-cap/OOM events
does not mean zero pressure, and the stall cause is not established.

The A-v7/B-v9 harness identities are recorded. B independently calibrates and
preserves its natural planner inputs, then replays A's planner quartet with
FIT=0 to match effective settings. This is not an entirely unmodified launch
comparison. No host-paging, 1M-request, concurrency or optional KV-policy coverage
is implied; short reference pieces also do not exercise competitive index
selection beyond top-k 512.

The measured B runtime used a hash-pinned strict native wheel; installed extension
and Python-module bytes matched that wheel before/after. B's native treatment
changes `cpuinfer.h` and `experts_base.py`. Both arms additionally share a
physical-NUMA mapping adapter in `worker_pool.cpp`; the selected three-file
reconstruction starts from this adapted A tree, not an unadapted clean setup.
On the actual identity mapping of nodes 0–7, the adapter preserves target nodes
and strict policy. It handles sparse/reordered physical IDs safely. The
[native provenance note and exact common adapter](benchmarks/hot-glue-00fd12a-20261024-review-r2/NATIVE_PROVENANCE.md)
separate this common host adaptation from the optional B delta and record the
measured wheel/source identities. No byte-identical fresh-toolchain build is claimed.
The measurements above remain tied to `00fd12a` / `9c0f6d6`; merging or applying
the draft on newer upstream `d7f187` / `8b9f724` is a separate CPU/source check,
not a full-model numerical or performance result on that newer pin.
