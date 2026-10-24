# CPU/GPU expert overlap

Tier 1 feature `cpu-gpu-overlap` adds independent `KT_SHARED_AFTER_SUBMIT=1`
and `KT_HANDOFF_PACK=1` switches (both default off). Select it with its prerequisites
or use the ordinary all-features setup. The native Python submit hook is rebuilt
only when this feature is selected. The first switch starts the shared expert
after CPU submission; the second packs inputs, weights and exact widened IDs into
the pinned request buffers in one kernel. Unsupported inputs keep the copy path.
Poll order, queue spin, CPU affinity and hot-slot policy are unchanged.

05yuki's SM120 single-launch measurements in PR #5 found 31.5/31.6 tok/s
with both off, 32.8 with shared-after-submit, 32.0 with handoff pack, and 32.7
with both. These are maintainer measurements of the earlier bundle, not fresh
measurements of this split. Scheduling/copy changes are intended to preserve
values; separate switch-on/off full-model logprob identity is not established
by those throughput/NLL results. No fresh model run is included in this split.

The implementation was developed locally for this SGLang/KT recipe on
2026-10-22. The former “Strata gpu_glue bundle” label referred to the local
workspace investigation; no Strata source was copied into this feature.
