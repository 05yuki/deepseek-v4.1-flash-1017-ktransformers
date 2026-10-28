# Optional SM89 indexer and merge kernels

`patches/sglang-optional/01-sm89-indexer-merge.patch` (applied by hand on the full stack, see
[patches.md](patches.md#optional-patches)) adds two independent default-off switches, scoped to compute
capability 8.9: `KT_IDX_LIVE_TOPK=1` and `KT_SM89_FUSED_MERGE=1`.
Live top-k bounds work by device-side sequence lengths and keeps the existing
torch path above `KT_IDX_LIVE_TOPK_MAX_L` (65536). The fused merge preserves
the torch operation order and BF16 rounding in the portable attention fallback;
the default SM89 dual-scope path already merges in its epilogue.

The historical whole-bundle SM89 result (31.0 to 35.0 tok/s) does not isolate
either switch. 05yuki's SM120 measurements do not exercise either SM89 path.
This split contains no new benchmark or isolated SM89 performance claim.
The component parity fixtures are retained in `tools/test_live_topk.py`,
`tools/test_merge_sink.py`, and `tools/test_fused_merge_wiring.py`.

These Python/Triton implementations were developed for this recipe during the
local GPU-glue investigation on 2026-10-22. “Strata gpu_glue bundle” in the old
header named that workspace investigation, not imported Strata source. The
live-length bound extends 05yuki's logits-bound idea. The radix ordering and
merge arithmetic follow the existing torch behavior.
