# Optional SM89 fused merge

`patches/sglang-optional/01-sm89-fused-merge.patch` (applied by hand on the full stack, see
[patches.md](patches.md#optional-patches)) adds one default-off switch, scoped to compute
capability 8.9: `KT_SM89_FUSED_MERGE=1`. It merges the portable attention fallback's partial
results and the attention sink in one kernel. It keeps the torch operation order and BF16 rounding.
The default SM89 dual-scope path already merges in its epilogue.

The switch has no isolated SM89 measurement. The SM120 measurements in this repository do not run
the SM89 path. The parity fixtures are `tools/test_merge_sink.py` and
`tools/test_fused_merge_wiring.py`.

The kernel was developed for this recipe by howl9003 (#7). The merge arithmetic follows the torch
behavior.
