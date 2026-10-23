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
`verify_source_bundle.py` checks all thirteen hashes.

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

Source equivalence and CPU-level checks do not establish model quality, memory
fit, or speed. The selected B runtime used the original strict native wheel;
this source draft has not reproduced that wheel binary. Quality and speed claims
will cite the completed, reviewed A/B receipts rather than historical estimates.
