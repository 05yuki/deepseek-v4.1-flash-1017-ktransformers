# Tier 1, one patch per feature

Tier 1 is one patch per feature in `patches/sglang-tier1/`, applied in the order of their numbers on
SGLang main at `58484246` after tiers 0 and 0.5; tier 2 applies on top. `setup.sh` applies all of them by
default; `SGLANG_FEATURES=a,b,...` picks some (the names below), and `serve.sh` turns on only what
was applied.

A patch with no prerequisites applies alone on tiers 0 and 0.5. A patch with prerequisites applies on
them plus those patches (`patches/sglang-tier1/requires.txt`, checked by `setup.sh`). Combinations that

apply: each patch alone, all of them, and all of them but any one (with what needs it); other
combinations may need `git apply -3`. Served and checked end to end: all of them, none, and each
set measured in [Effect of each patch](#effect-of-each-patch).

Defaults are given as *code / serve.sh*: what the Python does with the variable unset, and what
`serve.sh` (with `lib/kt-kv-hot-default.sh` and `lib/kt-lend-auto.sh`) sets. Switches marked "on in
the code" are on as soon as the patch is applied.

| # | patch | lines (+/-) | prerequisites |
|---|---|---|---|
| 01 | [`01-fp8-decode.patch`](#01-fp8-decode) | +222 / -1 | none |
| 02 | [`02-gpu-moe-fp32.patch`](#02-gpu-moe-fp32) | +486 / -1 | none |
| 03 | [`03-cutlass-kt-expert-count.patch`](#03-cutlass-kt-expert-count) | +6 / -1 | none |
| 04 | [`04-engram-row-prefetch.patch`](#04-engram-row-prefetch) | +74 / -7 | none |
| 05 | [`05-wo-a-fp8.patch`](#05-wo-a-fp8) | +174 / -2 | none |
| 06 | [`06-vision-on-demand.patch`](#06-vision-on-demand) | +87 / -11 | `wo-a-fp8` |
| 07 | [`07-thinking-default-parser.patch`](#07-thinking-default-parser) | +6 / -4 | none |
| 08 | [`08-loading.patch`](#08-loading) | +250 / -2 | none |
| 09 | [`09-launch-timing.patch`](#09-launch-timing) | +400 / -0 | none |
| 10 | [`10-pool-swa-guard.patch`](#10-pool-swa-guard) | +17 / -0 | none |
| 11 | [`11-placement.patch`](#11-placement) | +45 / -0 | none |
| 12 | [`12-kv-vmm.patch`](#12-kv-vmm) | +242 / -0 | none |
| 13 | [`13-prefill-lend.patch`](#13-prefill-lend) | +1664 / -1 | none |
| 14 | [`14-stream-prefill.patch`](#14-stream-prefill) | +2062 / -14 | `loading`, `placement`, `prefill-lend` |
| 15 | [`15-kv-lend-plan.patch`](#15-kv-lend-plan) | +1753 / -5 | `kv-vmm`, `prefill-lend` |
| 16 | [`16-hot-lend.patch`](#16-hot-lend) | +736 / -34 | `kv-vmm`, `prefill-lend`, `stream-prefill`, `kv-lend-plan` |
| 17 | [`17-host-kv.patch`](#17-host-kv) | +487 / -10 | `kv-vmm`, `prefill-lend`, `stream-prefill`, `kv-lend-plan`, `hot-lend` |
| 18 | [`18-stream-mmq.patch`](#18-stream-mmq) | +1618 / -3 (llama.cpp's kernels come from `lib/llama-mmq.sh`) | `stream-prefill` |
| 19 | [`19-cpu-gpu-overlap.patch`](#19-cpu-gpu-overlap) | +190 / -2 | `loading`, `placement`, `prefill-lend`, `stream-prefill` |
| 20 | [`20-expert-cache.patch`](#20-expert-cache) | +200 / -3 | `loading`, `placement`, `prefill-lend`, `stream-prefill`, `kv-vmm`, `kv-lend-plan`, `hot-lend`, `stream-mmq` |
| 21 | [`21-hicache.patch`](#21-hicache) | +47 / -3 | `kv-vmm`, `prefill-lend`, `kv-lend-plan` |
| 22 | [`22-prefill-warmup.patch`](#22-prefill-warmup) | +25 / -0 | none |
| 23 | [`23-swa-checkpoint-tail.patch`](#23-swa-checkpoint-tail) | +16 / -2 | none |
| 24 | [`24-rope-tables.patch`](#24-rope-tables) | +160 / -8 | `wo-a-fp8`, `vision-on-demand`, `kv-vmm`, `prefill-lend` |
| 25 | [`25-embed-on-host.patch`](#25-embed-on-host) | +88 / -0 | `wo-a-fp8`, `vision-on-demand`, `kv-vmm`, `prefill-lend`, `rope-tables` |
| 26 | [`26-gguf-lend.patch`](#26-gguf-lend) | +2826 / -38 | `wo-a-fp8`, `vision-on-demand`, `loading`, `placement`, `kv-vmm`, `prefill-lend`, `stream-prefill`, `kv-lend-plan`, `hot-lend`, `stream-mmq`, `cpu-gpu-overlap`, `expert-cache`, `rope-tables` |

## Effect of each patch

Each patch added to a base, one launch each, on the machine in the README (2 × EPYC 7452, 2 × RTX
5070 Ti), `TIER=2` with the kt-kernel patch and `SGLANG_FEATURES` set to the base plus the patch.
Decode is the median of the server's generation throughput over four 1,024-token Japanese prose
answers; free VRAM is what the server reports after loading (`available_gpu_mem`, GPU 0). "For"
says what the patch is meant to change; a patch whose number does not move here has its reason
in its section.

Decode and memory, on tiers 0 and 2 with no tier-1 patch (`SGLANG_FEATURES=none`):

| added | for | decode (tok/s) | free VRAM (GB) |
|---|---|---|---|
| nothing (base) | | 15.9 | 3.16 |
| [`fp8-decode`](#01-fp8-decode) | decode | 27.3 | 3.16 |
| [`placement`](#11-placement) | decode | 18.4 | 3.16 |
| [`engram-row-prefetch`](#04-engram-row-prefetch) | decode | 16.2 | 3.16 |
| `fp8-decode`, `engram-row-prefetch` | | 28.2 | 3.16 |
| [`wo-a-fp8`](#05-wo-a-fp8) | decode, memory | 16.1 | 3.74 |
| `fp8-decode`, `wo-a-fp8` | | 28.3 | 3.74 |
| `wo-a-fp8`, [`vision-on-demand`](#06-vision-on-demand) | memory | 16.1 | 4.69 |
| [`gpu-moe-fp32`](#02-gpu-moe-fp32) | numerics | 15.8 | 3.20 |
| [`loading`](#08-loading), [`launch-timing`](#09-launch-timing), [`pool-swa-guard`](#10-pool-swa-guard) (each alone) | startup | 15.9 to 16.0 | 3.16 |
| [`kv-vmm`](#12-kv-vmm), [`prefill-lend`](#13-prefill-lend), [`kv-lend-plan`](#15-kv-lend-plan) (each alone) | prefill (below) | 15.9 to 16.0 | 3.07 to 3.16 |
| [`cutlass-kt-expert-count`](#03-cutlass-kt-expert-count), [`thinking-default-parser`](#07-thinking-default-parser) (each alone) | fixes | 15.9 to 16.0 | 3.16 |

The gains do not add up. A token waits for the slower of the CPU and GPU experts in each layer,
so a cut on one side shows only where that side was the slower; a fixed cut a token shows more
as the token gets shorter.

A 35,000-token prompt, on the streamed-prefill set (`loading`, `placement`, `prefill-lend`,
`stream-prefill`; without it a prompt's experts run on the CPU, about 40 tok/s). The first prompt
after launch pays a one-time warm-up; the second is a new 35,000-token prompt on the warm server:

| added | 35K, first (s) | 35K, second (s) | decode (tok/s) | free VRAM (GB) |
|---|---|---|---|---|
| the streamed-prefill set (base) | 56.8 | 38.8 | 18.1 | 2.59 |
| `gpu-moe-fp32` | 55.3 | 37.2 | 18.4 | 2.62 |
| `wo-a-fp8` | 56.9 | 38.8 | 18.3 | 3.17 |
| `kv-vmm`, `kv-lend-plan` | 43.9 | 24.5 | 18.4 | 2.69 |
| `kv-vmm`, `kv-lend-plan`, `hot-lend` | 44.4 | 24.8 | 19.2 | 2.66 |
| all sixteen | 41.5 | 21.0 | 46.2 | 4.24 |
| `cpu-gpu-overlap` on the sixteen plus `stream-mmq` | unchanged | unchanged | 45.5-46.1 → 47.5-48.3 | 4.24 |

Numerics: the 2-token NLL check gives the base's value for every patch above but four.
`fp8-decode` and `gpu-moe-fp32` change the GPU arithmetic (see their Numerics lines), and
`placement` and `hot-lend` change which experts run on the GPU.

## 01 fp8-decode

Faster block-FP8 dense GEMMs for decode-sized inputs. With up to four rows, `w8a8_block_fp8_matmul` runs a Triton GEMV (`fp8_block_gemv.py`) that steps K over several scale groups at a time. With up to 16 rows and no tuned config for the shape, the Triton matmul uses a split-K tiling.

Switches (code / serve.sh):

- `SGLANG_FP8_GEMV`: 1 (on in the code) / 1
- `SGLANG_FP8_SKINNY`: 1 (on in the code); SGLANG_FP8_SKINNY_SPLIT_K 4, _BLOCK_N 128, _LOG / unset (on)

Numerics: Not bit-identical when on: the same FP8 operands summed in a different fp32 order (GEMV, split-K). Prefill shapes are unchanged.

Effect: decode 15.9 → 27.3 tok/s alone (a token 62.7 → 36.6 ms). Without it the dense FP8 GEMMs are the longer side of a token.

Prerequisites: none; applies alone on tier 0.

Files (under `python/sglang/`):

- `kernels/ops/quantization/fp8_block_gemv.py` (new)
- `kernels/ops/gemm/fp8_kernel.py`

## 02 gpu-moe-fp32

The GPU-resident MXFP4 experts in fp32 glue. `SGLANG_MXFP4_MOE_FP32=1` (with `--moe-runner-backend marlin`) replaces the Marlin MoE method with `Mxfp4Fp32MoEMethod` (`mxfp4_fp32_moe.py`): Triton kernels that read the checkpoint's MXFP4 layout and round to bf16 once, at the weighted sum; decode uses a per-route GEMV, larger batches the tensor cores on the routes with a weight. When the experts run through Marlin instead, `SGLANG_MARLIN_MOE_FP32_GLUE=1` keeps the steps around Marlin's two GEMMs in fp32.

Switches (code / serve.sh):

- `SGLANG_MXFP4_MOE_FP32`: 0; SGLANG_MXFP4_MOE_FP32_SYNC (debug) / 1
- `SGLANG_MARLIN_MOE_FP32_GLUE`: 0 / 1

Numerics: Not bit-identical to Marlin: fewer bf16 roundings, closer to an fp32 reference. The fp32 MoE repeats bit for bit from launch to launch.

Effect: no speed change in decode (15.8 against 15.9): the experts do the same work with fewer roundings. The first 35K prompt on the streamed-prefill set goes 56.8 → 55.3 s. The long NLL of single launches moved both ways (3.1479 → 3.1645 alone, 3.1346 → 3.1144 on the streamed-prefill set), within what launches differ by; the rounding itself is checked against an fp64 reference by `tools/mxfp4-fp32-moe-test.py`.

Prerequisites: none; applies alone on tier 0.

Files (under `python/sglang/`):

- `srt/layers/moe/fused_moe_triton/fused_marlin_moe.py`
- `srt/layers/moe/mxfp4_fp32_moe.py` (new)
- `srt/layers/quantization/fp8.py`
- `srt/layers/quantization/mxfp4_marlin_moe.py`

## 03 cutlass-kt-expert-count

A fix in the FlashInfer CUTLASS MXFP4 MoE method: the expert count is taken from the weight tensor (`layer.w13_weight.shape[0]`) instead of `layer.num_local_experts`, because under the kt-kernel wrapper the method holds only the GPU-resident experts.

Numerics: No switch. Identical without KTransformers (both counts agree); under KTransformers the CUTLASS MXFP4 path used to fail.

Effect: nothing to measure with `serve.sh`'s default, which runs the resident experts through Marlin's method or `gpu-moe-fp32`'s kernels. With `KT_HOT_BACKEND` set to anything but `marlin` they run through FlashInfer's CUTLASS MXFP4 MoE, which under KTransformers fails without this fix.

Prerequisites: none; applies alone on tier 0.

Files (under `python/sglang/`):

- `srt/layers/quantization/mxfp4_flashinfer_cutlass_moe.py`

## 04 engram-row-prefetch

Each engram layer's rank-local row lookup starts at the beginning of the forward on a side stream instead of when its layer comes: in decode, and in a prefill chunk, where the later engram layer's rows are read while the layers before it run. The all-reduce, `wkv` and the gate stay where they were. Only with one pipeline stage, no DP attention and engram tables that are not shared; a prefill chunk also not under context parallelism or a breakable CUDA graph.

Switches (code / serve.sh):

- `SGLANG_DSV41_ENGRAM_PREFETCH_ROWS`: 1 (on in the code) / 1
- `SGLANG_DSV41_ENGRAM_PREFETCH_PREFILL`: 1 (on in the code) / 1

Numerics: Bit-identical: only the lookup moves (a prefill's teacher-forced log-probs identical token for token).

Effect in prefill: a 35K prompt in 9,216-token chunks 972 → 1,084 tok/s alone; with the engram adapter's next-chunk warm-up most of the rest of the read time goes (docs/optimizations.md). In decode: a token about 1 ms shorter, alone (15.9 → 16.2 tok/s, 62.7 → 61.8 ms) and with `fp8-decode` (27.3 → 28.2 tok/s, 36.6 → 35.5 ms). The cut is a fixed time a token, so it shows more as the token gets shorter.

Prerequisites: none; applies alone on tier 0.

Files (under `python/sglang/`):

- `srt/layers/engram.py`
- `srt/models/deepseek_v4.py`

## 05 wo-a-fp8

DeepSeek V4 attention's `wo_a` in FP8 (`deepseek_common/wo_a_fp8_gemv.py`). `SGLANG_DSV4_WO_A_FP8_NATIVE=1` keeps the checkpoint's FP8 `wo_a` as loaded (no bf16 copy): decode reads it with an FP8 GEMV, prefill dequantizes it into one shared bf16 buffer per device. SGLang main's own FP8 `wo_a` GEMM takes 128 x 128 blocks; 1017's are 32 x 32, so this path is still needed.

Switches (code / serve.sh):

- `SGLANG_DSV4_WO_A_FP8_NATIVE`: 0 / 1

Numerics: Decode GEMV not bit-identical (fp32 summation order; 99.98% of outputs equal, the rest 1 ulp). The prefill dequantize path is bit-identical (the same bf16 values).

Effect: 0.58 GB more free VRAM on each GPU (3.16 → 3.74 GB), the bf16 copy of `wo_a` that is no longer made. Decode: a token 0.6 ms shorter alone (15.9 → 16.1 tok/s), 1.3 ms with `fp8-decode` (27.3 → 28.3 tok/s). Prefill does not change (56.9 against 56.8 s on the streamed-prefill set): it computes on the same bf16 values.

Prerequisites: none; applies alone on tier 0.

Files (under `python/sglang/`):

- `srt/models/deepseek_common/wo_a_fp8_gemv.py` (new)
- `srt/models/deepseek_v4.py`

## 06 vision-on-demand

The vision tower's GPU memory when serving text. `SGLANG_DSV4_VISION_ON_DEMAND=1` moves the tower and aligner to pinned host memory after loading and puts them on the GPU only while a batch's images are encoded (if the KV plan of `kv-lend-plan` is applied, it accounts for the copy). To serve text only, SGLang main's own `--language-model-only` builds and loads no tower at all.

Switches (code / serve.sh):

- `SGLANG_DSV4_VISION_ON_DEMAND`: 0 / 1

Numerics: Bit-identical: placement only (image batches pay a host-to-device copy).

Effect: 0.95 GB more free VRAM on each GPU (3.74 → 4.69 GB with `wo-a-fp8`). Decode and text prefill do not change: the tower is not used without images. In the full setup the hot lend fills the freed memory with more experts.

Prerequisites: `wo-a-fp8`.

Files (under `python/sglang/`):

- `srt/models/deepseek_v4.py`

## 07 thinking-default-parser

When a request sends no thinking toggle in `chat_template_kwargs`, the reasoning parser falls back to `SGLANG_DEFAULT_THINKING`, the same fallback the DeepSeek V4 encoder uses to decide whether the prompt opens a thinking block, so the parser and the prompt agree. Before, a missing toggle always meant no thinking for the parser.

Switches (code / serve.sh):

- `SGLANG_DEFAULT_THINKING (existing)`: 0 / unset

Numerics: No switch of its own. Identical unless `SGLANG_DEFAULT_THINKING=1`.

Effect: none on speed; it changes how the output is split into reasoning and content.

Prerequisites: none; applies alone on tier 0.

Files (under `python/sglang/`):

- `srt/entrypoints/openai/serving_chat.py`

## 08 loading

Weight loading for the hybrid MoE. The GPU side's safetensors loader skips the experts that only the CPU runs, by name, and DeepSeek V4's `load_weights` skips them too (`KT_LOAD_SKIP_CPU_EXPERTS`); it still reads the hot lend's experts when `hot-lend` is applied. `KT_LOAD_PREFETCH=<threads>` reads the dense tensors of each shard ahead in 16 MiB pieces on a thread pool, so the loaders' mmap faults hit the page cache. In the kt-kernel wrapper: `KT_GPU_STREAM_DROP_CACHE=<dir>` drops the page cache of the checkpoint per layer so kt-kernel's arenas find 2 MB pages; `KT_NUMA_NODES` passes kt-kernel the NUMA nodes to use; with `--kt-num-gpu-experts 0` the empty GPU weights are not repacked (dsv41-flash-dual-epyc#2).

Switches (code / serve.sh):

- `KT_LOAD_SKIP_CPU_EXPERTS`: 1 (on in the code); KT_LOAD_SKIP_CPU_EXPERTS_RE / unset (on)
- `KT_LOAD_PREFETCH`: 0; KT_LOAD_PREFETCH_PIECE_MB 16, KT_LOAD_PREFETCH_SKIP, KT_LOAD_PREFETCH_TAIL 4 / 4
- `KT_GPU_STREAM_DROP_CACHE`: unset (off) / the model directory
- `KT_NUMA_NODES`: unset (kt-kernel's default) / unset

Numerics: Bit-identical: the skipped tensors are ones the loader dropped anyway; the rest is I/O and placement.

Effect: ready 149 s after start, against 173 to 224 s (median 177 s) for the other launches of the decode table (one launch each). It does not change decode.

Prerequisites: none; applies alone on tier 0.

Files (under `python/sglang/`):

- `srt/layers/moe/kt_ep_wrapper.py`
- `srt/model_loader/weight_utils.py`
- `srt/models/deepseek_v4.py`

## 09 launch-timing

`KT_LAUNCH_TIMING=1` prints a per-step timeline of the weight load on each rank, with the disk bytes each step read, and sums the steps that repeat (tensor fetches, a layer's CPU experts) in a table at the end. `KT_LAUNCH_TIMING_STACKS=<seconds>` also samples the loading thread's stack. It wraps the load's functions in place; the ones from patches that are not applied are skipped.

Switches (code / serve.sh):

- `KT_LAUNCH_TIMING`: 0; KT_LAUNCH_TIMING_STACKS / unset (off)

Numerics: No effect on outputs.

Effect: none unless `KT_LAUNCH_TIMING=1`; it only records.

Prerequisites: none; applies alone on tier 0.

Files (under `python/sglang/`):

- `srt/layers/moe/kt_launch_timing.py` (new)
- `srt/model_executor/model_runner.py`

## 10 pool-swa-guard

A startup check in the pool configurator for DeepSeek V4: an SWA pool smaller than one prefill chunk plus a page raises a ValueError that names the flags to change (`--swa-full-tokens-ratio`, `--max-total-tokens`, `--chunked-prefill-size`), instead of a scheduler that re-queues the request forever with the GPU idle.

Numerics: No switch. No effect on outputs; a setup that used to hang now fails at startup.

Effect: none on a setup that fits; it is one check at startup.

Prerequisites: none; applies alone on tier 0.

Files (under `python/sglang/`):

- `srt/model_executor/pool_configurator.py`

## 11 placement

A routing dump to build a placement from a workload: `KT_ROUTING_DUMP=<prefix>` saves a per-layer histogram of the routed physical ids every 30 s (a captured `index_add_`, so it counts under CUDA graph replay). A non-identity `--init-expert-location` (`placements/`) needs nothing more: SGLang's TopK resolves it for every router.

Switches (code / serve.sh):

- `--init-expert-location (existing CLI)`: identity / a file from placements/
- `KT_ROUTING_DUMP`: unset (off) / unset

Numerics: The dump does not change outputs.

Effect: none on its own now; the placement it records puts the hottest experts in the resident rows and the hot lend's first slots.

Prerequisites: none; applies alone on tier 0.

Files (under `python/sglang/`):

- `srt/layers/moe/kt_ep_wrapper.py`

## 12 kv-vmm

`KT_KV_VMM=1` puts the KV pool on memory whose physical pages can be unmapped and mapped again while the addresses, and the CUDA graphs that read them, stay the same (`kt_kv_vmm.py`, hooked into the torch_memory_saver adapter). Needs `libkv_vmm.so` from the repo's `kv-vmm/kv_vmm.c` (`KT_KV_VMM_LIB`).

Switches (code / serve.sh):

- `KT_KV_VMM`: 0; KT_KV_VMM_LIB, KT_KV_VMM_DIRECT / 1

Numerics: Bit-identical.

Effect: none alone (decode 16.0); it is what `kv-lend-plan` lends from (35K below).

Prerequisites: none; applies alone on tier 0.

Files (under `python/sglang/`):

- `srt/layers/moe/kt_kv_vmm.py` (new)
- `srt/utils/torch_memory_saver_adapter.py`

## 13 prefill-lend

`KT_PREFILL_LEND=1`: during a streamed prefill the GPU-resident weights give their memory to the prefill's scratch and come back layer by layer, at the same addresses, just before each layer runs (`kt_lend.py`, hooked into the loader and the model runner's forward). `KT_PREFILL_LEND_PROBE=<chunks>@<prefixes>` measures how large a chunk fits before serving: for each chunk size, four chunks ending at each prefix, from prefix 0 when the prefix is shorter (`kt_lend_probe.py`, used by `lib/kt-lend-auto.sh`). `KT_FAST_PINNED=1` pins the lend's host buffers as huge-page memory registered with `cudaHostRegister` instead of `cudaHostAlloc` (`kt_pinned.py`). DeepSeek V4's padded query buffer is allocated in the prefill's scratch. V4's hc mixing weights stay resident, packed in one buffer (`KT_PREFILL_LEND_RESIDENT`, `hc_`): SGLang main's layer boundary reads the next layer's before that layer runs. The tail source KV write, which reads its layer's weights before the layer runs, enters the layer first, and the decoder replay tail's CUDA graphs are not used while the lend is on (they would keep the addresses of their capture). The lend calls into the streamed prefill, the KV lend and the hot lend only when those patches are applied and on.

Switches (code / serve.sh):

- `KT_PREFILL_LEND`: 0; KT_PREFILL_LEND_HINT, _MARGIN_MB, _SKIP, _LONG_SEQ, _MAX_CHUNK, ... / 1 (via lib/kt-lend-auto.sh)
- `KT_PREFILL_LEND_PROBE`: unset / set by lib/kt-lend-auto.sh on the calibration launch
- `KT_LEND_SEED_DIR`, `KT_LEND_SEED`, `KT_LEND_SEED_FREE_MARGIN_MB`: unset / hints/kt-lend at the launcher's defaults, 1, 512 (lib/kt-lend-auto.sh: a shipped hint for the GPU model and count seeds the setup's rank files, with that much less free memory, instead of the calibration launch; tools/lend-hint-export.py writes one)
- `KT_FAST_PINNED`: 0 / 0
- `KT_PREFILL_LEND_RESIDENT`: hc_ / hc_

Numerics: Bit-identical at a given chunk size. It lets the launcher pick a larger chunk, and the chunk size moves NLL.

Effect: none in decode (15.9). It is part of the streamed-prefill set, whose 35K prompt takes 56.8 s against about 15 minutes on the CPU.

Overlap scheduling can leave GPU readers in flight when the next phase begins.
Both phase transitions wait for those readers before unmapping weights or scratch;
ordinary decode and disabled lending do not add a device synchronization.
`python tools/lend-transition-fences-test.py` checks this ordering against the
module embedded in the patch, without SGLang, model weights or a GPU.

Prerequisites: none; applies alone on tier 0.

Files (under `python/sglang/`):

- `srt/layers/moe/kt_lend.py` (new)
- `srt/layers/moe/kt_lend_probe.py` (new)
- `srt/layers/moe/kt_pinned.py` (new)
- `srt/managers/scheduler.py`
- `srt/model_executor/model_runner.py`
- `srt/model_loader/loader.py`
- `srt/models/deepseek_v4.py`
- `srt/models/deepseek_v4_replay_tail.py`

## 14 stream-prefill

`KT_GPU_STREAM_PREFILL=<tokens>`: above that many tokens the CPU experts are not computed by kt-kernel; their weights are copied to the GPU in groups and computed there, the copy of the next group overlapping the compute of the current one (`kt_stream_prefill.py`). `KT_GPU_STREAM_W16=1` dequantizes each group to bf16 and runs FlashInfer's bf16 CUTLASS MoE (SM89+); otherwise the SM120 MXFP8 x MXFP4 kernel. `KT_GPU_STREAM_ZEROCOPY` reads kt-kernel's shared expert memory directly and `KT_GPU_STREAM_GRAPH` launches each group as one CUDA graph. One CUTLASS MoE workspace per GPU is preallocated before the decode graphs are captured, for the resident experts as well (this part is always on), and `KT_GPU_STREAM_ROW_BLOCK` splits long chunks into row blocks. Under the prefill lend the graphs' shared input and output buffers grow in steps of `KT_STREAM_GIN_STEP` rows as chunks get larger, instead of being sized for the plan's cap at once. Also: early streamer init, CPU groups (`KT_STREAM_CPU_GROUPS`), a reference check (`KT_HOT_CHECK`), and comment updates in the wrapper. With SGLang main's decoder bounded replay, the layers from the replay's start run over each chunk's last 128 rows only; below the threshold those calls stay on the CPU, which is faster than streaming them (`KT_GPU_STREAM_IN_WINDOW=1` streams every call of a streamed forward, for comparison).

With GPUs and kt-kernel's NUMA parts that differ in number (one GPU on two sockets, two GPUs on one socket, NPS2 or NPS4 nodes), each rank reads every part its rows of the intermediate size sit in. Checked on one GPU with two parts: an 11,769-token document read and summarised correctly, and the 35K prompt at 210 tok/s; four parts on two GPUs read correctly too.

A short prompt streams every expert all the same: 1,024 tokens at six experts each touch nearly all 384 of a layer, so the stream costs what a long prompt's does. `KT_STREAM_CPU_SHORT=<tokens>:<k>,...` lets a forward of at most `<tokens>` rows leave `k` of a layer's streamed groups to kt-kernel on the CPU, computed while the rest crosses PCIe. `serve.sh` sets `1024:64` on one GPU (64 of the 96 groups of 4) and `1024:32` on more. On one RTX 5070 Ti a fresh 1,024-token prompt takes 6.2-6.8 s instead of 9.2 s. On two it takes 5.4 s instead of 6.0 s. Longer prompts are unchanged: their forwards stream every group. Scored in one server with the share switched on and off, 23 pieces of 1,024 tokens give the same NLL: -0.0006 nats a token (z -0.32) on one GPU and -0.0006 (z -0.46) on two.

Switches (code / serve.sh):

- `KT_GPU_STREAM_PREFILL`: 0 (off) / 512
- `KT_GPU_STREAM_GROUP`: 32 / 4
- `KT_GPU_STREAM_W16`: 0 / 1
- `KT_GPU_STREAM_ZEROCOPY`: 0 / 1
- `KT_GPU_STREAM_GRAPH`: 1 (on in the code) / 1
- `KT_GPU_STREAM_EARLY_INIT`: 0 / 1
- `KT_GPU_STREAM_ROW_BLOCK`: 0 / 4096 (lib/kt-lend-auto.sh)
- `KT_STREAM_CPU_SHORT`: unset / `1024:64` on one GPU, `1024:32` on more (with a resident GPU expert)
- `KT_STREAM_CPU_GROUPS, KT_HOT_CHECK`: 0 / unset
- `KT_GPU_STREAM_IN_WINDOW`: 0 / 0
- `KT_STREAM_GIN_STEP`: 4096 / unset (4096)

Numerics: Not bit-identical when on: the streamed CPU experts run on the GPU (W16: bf16 rounding instead of kt-kernel's; the MXFP8 mode has a larger error). The shared workspace changes memory only. ZEROCOPY needs the kt-kernel tree patch (`KT_EXPERT_SHM`).

Effect: a 35K prompt in 56.8 s (first) and 38.8 s (second) with `loading`, `placement` and `prefill-lend`, against about 15 minutes with the experts on the CPU.

Prerequisites: `prefill-lend`, `loading`, `placement`.

Files (under `python/sglang/`):

- `srt/layers/moe/kt_ep_wrapper.py`
- `srt/layers/moe/kt_stream_prefill.py` (new)
- `srt/layers/moe/moe_runner/flashinfer_cutlass.py`
- `srt/layers/quantization/mxfp4_flashinfer_cutlass_moe.py`

## 15 kv-lend-plan

The KV pool's free pages during prefill. `KT_PREFILL_KV_LEND=1` lends the pool's free pages to the prefill (pages in use are never lent; the pool never shrinks), in every pool tensor laid out by the allocator's pages: the compressed KV of each compress ratio and the indexer keys, whose rows are a page's share, lend whole pages too. `KT_PREFILL_PLAN=1` plans each request's chunk size from the memory free at that moment, so a long prompt takes fewer, larger chunks (scheduler and schedule_batch hooks). `KT_PREFILL_SWA_LEND=1` sizes the sliding-window pool for the largest planned chunk and releases its free pages outside prefill. The file `kt_kv_hot_lend.py` ships here because the plan imports it; it does nothing until `hot-lend` wires it in.

Switches (code / serve.sh):

- `KT_PREFILL_KV_LEND`: 0; KT_PREFILL_KV_LEND_* / 1 (lib/kt-kv-hot-default.sh)
- `KT_PREFILL_PLAN`: 0; KT_PREFILL_PLAN_* / 1
- `KT_PREFILL_PLAN_COST_PRIOR`: unset / 2600,471040,280,820 (the cost the plan uses; lib/kt-lend-auto.sh)
- `KT_PREFILL_PLAN_FIT`: - / 0 (1: the cost fitted from this setup's recorded windows instead of the prior; a fitted cost ran out of memory at 131K)
- `KT_PREFILL_SWA_LEND`: 0; KT_PREFILL_SWA_* / 1

Numerics: The lend is bit-identical. The plan picks per-request chunk sizes, so the prefill chunking (and NLL) moves against a fixed chunk. The SWA lend changes memory only.

Effect: the 35K prompt on the streamed-prefill set goes 56.8 → 43.9 s (first) and 38.8 → 24.5 s (second), 4 chunks of 9,216 rows each time: each chunk streams the CPU experts to the GPU once, so fewer chunks take less time. The stream's buffers the first prompt grew (1.4 GB) stay mapped, and the plan counts them as free for the next prompt, whose chunks reuse them. Decode does not change (15.9 alone). The plan learns its cost a row from the setup's earlier launches, so this row and the next are measured after the setup has launched before.

Prerequisites: `kv-vmm`, `prefill-lend`.

Files (under `python/sglang/`):

- `srt/layers/moe/kt_kv_hot_lend.py` (new)
- `srt/layers/moe/kt_kv_prefill_lend.py` (new)
- `srt/layers/moe/kt_prefill_plan.py` (new)
- `srt/managers/schedule_batch.py`
- `srt/managers/scheduler.py`
- `srt/mem_cache/kv_cache_configurator.py`

## 16 hot-lend

`KT_KV_HOT_LEND=1`: outside prefill the KV pool's free pages hold up to `KT_KV_HOT_EXTRA` more experts a layer on the GPU, on a shadow layer; their routes go to the GPU and to the CPU at weight 0. The experts leave whenever the pool needs the pages or a prefill starts. `KT_KV_HOT_SPARE_FLOOR_MB` lets the slots also use idle driver memory above the floor. A lend needs `KT_KV_HOT_MIN_MB` (256) of free pages and idle driver memory together: with V4.1's KV formats the free pages alone hold half as much, and counting them alone left the slots empty. `KT_KV_HOT_EXTRA=auto` (serve.sh) gives a layer a slot for every non-resident expert in address space only, so how many hold memory follows the free memory at each lend (on two 16 GB cards 14-15 a layer with one resident expert). `KT_KV_HOT_RESERVE` keeps free KV pages back for 65,536 tokens: a new request's pages come from them, instead of every slot being handed back and loaded again for each request. Every rank holds the same slots, the fewest any rank has room for: the CPU runs on rank 0 with its routing.

`KT_KV_HOT_DYNAMIC_SLOTS=1`: the slots follow what is routed. Every routed expert is counted on the GPU; every 8 decode steps, outside the CUDA graph, each layer swaps up to 16 of its slots' experts for ones routed more often (counts halved every 256 steps). An expert that loses leaves the slot table at once and runs on the CPU again; the new one is copied in on a side stream straight from kt-kernel's expert memory (the zero-copy mapping `stream-prefill` registers, no host copy) and enters the table on the next round. A round waits for nothing the GPU is still doing: its table updates go up from pinned memory, split by table on the host, and the swapped-out experts leave by `index_fill_`, so the decode forward in flight runs on (on two GPUs a round takes 12 ms of host time instead of 24, decode +7-9%). At the first lend the slots check that experts copied this way match the placement's, byte for byte, and stay with the placement's experts if not (the lend then gives a layer only as many slots as there are banks). With `KT_KV_HOT_DYNAMIC_RESIDENT=1` the resident experts swap too, from one pool with the slots: kt-kernel then loads every expert, so one that leaves a resident row runs on the CPU, and the route tells kt-kernel which routes the GPU takes. When a prefill window opens, before the weights give up their memory, the resident rows are loaded with the placement's first experts again from kt-kernel's expert memory, weights and scales, and the next lend fills them again. Until 2026-10-31 the rows were left to the prefill lend's pinned copy, which holds only tensors of at least 1 MiB: the rows' scales stayed those of the expert swapped in, so every streamed prefill after a swap, and the first forward after it, ran those layers with another expert's scales (eight 1,536-token pieces after an English reply: NLL 0.8362, now 0.5328; 0.5297 on a fresh server). A pool clear (`/flush_cache`) gives the slots back before the SWA lend maps its pages again: until 2026-11-01 the SWA allocator, which the pool clears first, mapped them while the slots still held the memory, and both ranks went down. The route runs as one Triton kernel a layer. Without the dynamic slots the slots hold the placement's next-hottest experts, and the placement in `placements/` was recorded on Japanese prose: it catches 3-6% of the routes of code, JSON or math.

Switches (code / serve.sh):

- `KT_KV_HOT_LEND`: 0 / 1 (lib/kt-kv-hot-default.sh)
- `KT_KV_HOT_EXTRA`: 0 / auto (a slot for every non-resident expert, address space only)
- `KT_KV_HOT_RESERVE`: 4096 / 65536 (tokens of free KV pages kept back)
- `KT_KV_HOT_SPARE_FLOOR_MB`: -1 (off) / 1024
- `KT_KV_HOT_DYNAMIC_SLOTS`: 0 / 1
- `KT_KV_HOT_DYNAMIC_RESIDENT`: 0 / 1
- `KT_KV_HOT_ROUTE_TORCH`: 0 (the route as separate torch ops) / unset
- `KT_KV_HOT_DYN_EVERY, _SWAPS, _HALF_LIFE, _HYST, _LOG`: 8, 2 (a step and layer), 256, 1.0, 512 / unset
- `KT_KV_HOT_SLOTS_MAX, _MIN_MB, _RETRY, _MARGIN_MB, _CHECK*`: unset / unset

Numerics: Not bit-identical: the lent experts run in the GPU MoE kernel instead of kt-kernel, and which and how many are lent depends on the free KV pages and on what was routed before, so decode output depends on load and history. With the dynamic slots, every call's slot output against the same routes on the CPU (`KT_KV_HOT_CHECK=16`, 527 calls with lent routes) stayed within 1.6x of the CPU's own bf16 rounding; the paired NLL was 3.1317 against 3.1546 with the placement's slots. With the resident rows swapping too, each eager call's whole output against every route computed on the CPU (602 calls) differed by 4.1e-3 at the median and 5.0e-3 at most, the size of one bf16 rounding; the paired NLL was 3.1503 and 3.1576 against 3.1258 with the slots alone, inside the 3.11-3.16 that launches of the same setup gave on the same day.

Effect: decode 18.4 → 19.2 tok/s on the streamed-prefill set with `kv-vmm` and `kv-lend-plan`, on Japanese prose. The 35K prompt stays where it was (43.9 → 44.4 s first, 24.5 → 24.8 s second, 4 chunks of 9,216 rows in both): the slots give their memory back when a prefill window starts. On text the placement was not recorded on (all sixteen, `tools/bench.py --other-text`, one launch each), the placement's slots → dynamic slots → the resident rows swapping too: Python code 27.9 → 32.0 → 33.7, JSON 28.2 → 32.8 → 35.1, math 28.1 → 34.7 → 37.6, English technical prose 28.1 → 32.4 → 33.7, English prose 31.8 → 35.4 → 36.0, Chinese prose 34.3 → 38.5 → 39.1 tok/s; Japanese prose 43.7 → 43.8 → 44.2. Code and JSON spread their routes over more experts (a layer's 21 most used take 48-55% of its routes, against 71-75% in Japanese prose), so more slots help them most: `KT_KV_HOT_EXTRA` raises the cap where the GPU has the memory (9.4 MB a slot a layer on each GPU here).

Prerequisites: `kv-vmm`, `prefill-lend`, `stream-prefill`, `kv-lend-plan`.

Files (under `python/sglang/`):

- `srt/layers/moe/fused_moe_triton/layer.py`
- `srt/layers/moe/kt_ep_wrapper.py`

## 17 host-kv

On by default (`SGLANG_DSV41_HOST_KV=0` turns it off). When the GPU budget cannot hold the KV pool for the whole context, the pool is made at the context length anyway (1M tokens on 1017). The ratio-1/2 compressed KV it cannot hold lives in host memory.

**What stays on the GPU.**

- the window cache;
- the indexer keys, which decode scores in full at every index source;
- as much of the compressed KV as the budget leaves.

The compressed KV is `KVLayout.V41_FP4`, 288 bytes an entry, in the kv_source layers 2, 8, 14 and 20: 720 bytes a full token.

**Where the rest goes.**

- Each ratio-1/2 pool layer is one `kv-vmm` range (`kv_vmm_malloc_split` in `kv-vmm/kv_vmm.c`).
- Its first pages are device chunks; the rest are host chunks on the GPU's NUMA node.
- Every writer and reader addresses the range as before.
- A device chunk lent to a prefill or to hot experts reads as zeros.
- Without `kv-vmm`, a layer that cannot fit goes wholly to pinned host memory.

**Who uses which pages.**

- The allocator hands out the lowest free pages first, so requests fill the GPU's part before host memory.
- The hot lend and the prefill windows lend only device pages, and the prefill plan counts only those in a window's loan. When none are free, the hot lend runs on the driver's idle memory alone.

**Decode.**

- After each index source's top-k, a Triton kernel gathers its 512 rows into a device staging buffer in the pool's page layout. The rows are renumbered for attention. The kernel takes the data, scale and page bytes from the pool's own `KVLayout`, so V4, V41 and V41_FP4 pools all work.
- The following layers reuse the staging buffer, as they reuse the top-k.
- The gather costs about 10 µs an index source. Attention reading host memory in place would cost about 100 µs a layer.

**Prefill.**

- The pool layer's pages up to the batch's highest slot are copied to the GPU through an SM kernel, at the first attention that reads them in the forward.
- The copy waits for every writer. The bounded-replay tail rewrites the kv_source's entries after its own pass, and a copy taken earlier missed that.
- Only one layer's copy is held at a time: 288 MB at 1M tokens.

Switches (code / serve.sh):

- `SGLANG_DSV41_HOST_KV`: 1 / unset (0: the pool is what the GPU budget holds)
- `SGLANG_DSV41_HOST_KV_MAX_ROWS`: 16 (query rows a decode step gathers for; more read host memory in place)

Numerics: Bit-identical to the pools on the GPU:

- With the hot experts held fixed, the 24-piece NLL with the compressed KV in host memory equals the run with it on the GPU (2.5979).
- Greedy outputs are the same on three prompts up to 8,417 tokens.
- On two 16 GB cards the 1M pool fits the GPUs: no host part, NLL equal to the previous build (2.5826).
- The V4 layout (`SGLANG_DSV4_KV_LAYOUT=v4`) on one card: the 1M pool, with 1.43 GiB in host memory, gives the same greedy outputs on prompts up to 8,417 tokens as the pool the budget holds (89,344 tokens).

Effect (one RTX 5070 Ti, `serve.sh` defaults; the compressed KV all in host memory, 0.70 GiB):

- The KV pool goes from 180,480 to 1,048,576 tokens.
- Decode runs at 31.6 tok/s on Japanese prose (32.0 before).
- The 35K prompt prefills at 776 tok/s (774 before).
- A 300K-token prompt prefills at 868 tok/s and decodes at 24.6 tok/s, against 24.7 tok/s at 8K (random token ids).
- A 1M-token prompt prefills in 24.8 min and decodes at 23.8 tok/s.
- With the first 203,776 tokens' compressed KV on the GPU (mem fraction 0.82), a 300K-token prompt across the boundary prefills at 764 tok/s and decodes at 24.6 tok/s.

Prerequisites: `kv-vmm`, `prefill-lend`, `stream-prefill`, `kv-lend-plan`, `hot-lend`.

Files (under `python/sglang/`):

- `srt/layers/attention/deepseek_v4_backend.py`
- `srt/layers/moe/kt_kv_prefill_lend.py`
- `srt/layers/moe/kt_kv_vmm.py`
- `srt/mem_cache/deepseek_v4_memory_pool.py`
- `srt/mem_cache/dsv41_host_kv.py` (new)
- `srt/model_executor/pool_configurator.py`

## 18 stream-mmq

On in `serve.sh` (`KT_GPU_STREAM_MMQ=1`; off in the code). The streamed prefill multiplies its MXFP4 experts with llama.cpp's MMQ instead of a bf16 copy and the bf16 CUTLASS MoE.

**Why.** At the windows the prefill plan picks, the bf16 path was GEMM-bound.

- A layer took 38 ms plus 12 ms a thousand rows on each of two RTX 5070 Ti.
- The PCIe link needs 159 ms for the layer's 3.6 GB, so the math was the longer part from 10,000 rows on.
- Wider windows therefore saved little: one window for a 35K prompt instead of three was 2 s faster.

**What it does.**

- Each group's MXFP4 bytes are regrouped on the GPU into ggml `block_mxfp4`, by one Triton kernel.
- The E8M0 byte comes from kt-kernel's fp32 power-of-two scale. Values j and j + 16 share a byte.
- The regrouped group goes through `kt_gguf_moe_mmq` (the GGUF recipe's extension, which gains the MXFP4 type).
- MMQ is int8 mma. The activations and the clamped SwiGLU intermediate are quantized per 32 values with an fp16 scale. The weights stay exact.
- The routes are sorted by expert once a layer, with one host sync for their bounds.
- A group's slot goes back to the DMA as soon as it is regrouped.
- Neither the bf16 MoE output nor the CUTLASS workspace is allocated, so `serve.sh` lets the plan's windows grow: 300 KiB a row and 36,864 rows at most (460 KiB and 6,144 without it).
- A rank whose rows come from several kt-kernel parts (one GPU over two NUMA nodes) regroups part by part.
- llama.cpp's own files (33 of `23b0202`) are not in the patch. `setup.sh` runs `lib/llama-mmq.sh`, which fetches them, applies `patches/llama-mmq.patch` (15 changed files and the IQ1_M instance, 679 lines) and checks all 36 against `patches/llama-mmq.sha256`.
- `setup.sh` builds the extension for the GPU's arch.

**Effect** (two RTX 5070 Ti and two EPYC 7452 unless noted):

- A layer at 12,288 rows takes 154-168 ms, against 182-192 ms. That is the PCIe time now.
- At 35,000 rows the MoE math alone takes 107 ms, against 462 ms.
- With the wider windows: 35K prompt 1,806 tok/s (1,290 before), 131K 1,881 (1,452), 1M 1,087 (941).
- One GPU: 35K 1,056 tok/s (862 before), 65K 1,154 (811).
- No OOM and no allocator retries in any of these runs. Decode does not use the streamer.

Numerics: the 24-piece NLL with every piece through the streamer (stream threshold 256, the hot slots held fixed):

- Two GPUs: reference + 0.0005 with the bf16 path, - 0.0004 with MMQ; MMQ minus bf16 -0.0009 (SE 0.0033).
- One GPU: + 0.0034 and - 0.0013; MMQ minus bf16 -0.0047 (SE 0.0044).
- Greedy answers are the same as the bf16 path's.

Switches (code / serve.sh):

- `KT_GPU_STREAM_MMQ`: 0 / 1 (needs `KT_GPU_STREAM_ZEROCOPY=1` and `KT_GPU_STREAM_W16=1`, the defaults)
- `KT_GPU_STREAM_MMQ_ROUTES`: 8192 (a group's routes through the kernels at a time)
- `KT_GGUF_MOE_MMQ_PREBUILT`: unset / `prebuilt/mmq` when setup.sh installed a prebuilt archive (a built `kt_gguf_moe_mmq_<arch>.so` is imported from there, nothing compiled)

Prerequisites: `stream-prefill`.

Files (under `python/sglang/`):

- `srt/layers/moe/kt_stream_prefill.py`
- `srt/layers/moe/kt_gguf_moe_mmq.py` (new)
- `srt/layers/moe/kt_gguf_moe_mmq/` (new: the binding and the glue around llama.cpp 23b0202's MMQ kernels, MIT; the kernels themselves from `lib/llama-mmq.sh`)

## 19 cpu-gpu-overlap

On in `serve.sh` (`KT_SHARED_AFTER_SUBMIT=1 KT_HANDOFF_PACK=1`; off in the code). Two
changes to the decode step's hand-off to the CPU experts, contributed in #5 by howl9003.

- `KT_SHARED_AFTER_SUBMIT`: the shared expert's GEMMs fork from an event recorded right
  after the CPU request is submitted, instead of before the router, so they stop contending
  with the router → top-k → hand-off chain that feeds the CPU. Scheduling only.
- `KT_HANDOFF_PACK`: one Triton kernel writes the hidden rows (bf16), the routing weights
  (fp32) and the expert ids (widened to int64) straight into kt-kernel's pinned host buffers
  through their device-mapped addresses, in place of the ids cast and three DtoH copies the
  graph serialised before the request flag. Decode and verify batches only (256 rows at most);
  a buffer that is not device-mapped turns the pack off for the process, with a warning.
  `patches/kt-kernel-cpu-handoff.patch` gives `submit_forward` the `packer` argument.

**Effect** (two RTX 5070 Ti, TP 2, the `serve.sh` defaults, two launches each way):

- Japanese prose decode 46.1 / 45.5 → 48.3 / 47.5 tok/s (+4.6%).
- The six other texts rise 2-7% in both rounds: Chinese prose 39.7-40.7 → 42.2-43.5,
  English prose 38.1-38.2 → 39.6-40.2, math 38.8-39.3 → 40.8-41.4, JSON 36.4-36.8 →
  37.8-38.2, Python code 34.9-35.1 → 36.2-36.3, English technical prose 34.7-34.8 →
  35.8-36.3.
- The 35K prefill is unchanged (1,474-1,484 tok/s first, 1,834-1,853 second).
- One GPU (SM89, howl9003): 31.5 → 32.7 tok/s.

Numerics: bit-identical. The 24-piece NLL with the hot slots held fixed is 0.7139 with both switches off and on, 11,496 of 11,496 tokens identical (two RTX 5070 Ti, TP 2). howl9003's check on one SM89 GPU: all four switch combinations bit-identical on selected-token and top-8 logprobs, 112 cases.

Switches (code / serve.sh):

- `KT_SHARED_AFTER_SUBMIT`: 0 / 1
- `KT_HANDOFF_PACK`: 0 / 1

## 20 expert-cache

On in `serve.sh` when one CPU expert pool's NUMA node has less than 370 GiB (`KT_MXFP4_FULL_GIB`);
off in the code (`KT_MXFP4_EXPERT_CACHE_GIB` unset). With `patches/kt-kernel-expert-cache.patch`,
kt-kernel keeps a RAM cache of the routed experts instead of all of them, and reads the others from
the checkpoint when they are routed. This is what runs the model on one socket with 256 GB of RAM.

**Why.** The MXFP4 experts are 269 GiB in the checkpoint and 316 GiB in kt-kernel's arenas (their
scales widened to fp32). One NUMA node of 256 GB does not hold them: kt-kernel's threads allocate
on their own node, and the load ends in the OOM killer.

**What it does** (kt-kernel, `operators/avx2/mxfp4-expert-cache.hpp`):

- One memfd arena of fixed 18 MiB slots, one expert a slot: the gate, up and down weights as the
  checkpoint packs them, then their ue8m0 scale bytes, each at a 4 KiB boundary.
- The kernels read the scale bytes as their one-byte scale codes against one fixed table
  (ue8m0 e → 2^(e−127), the values the bf16 load gives bit for bit), so no fp32 scales are kept.
  A forward of more than one row expands the fp32 scales its kernels read into a scratch.
- At load, each layer reads its share of the slots (its CPU experts in id order) at the drive's
  speed. A decode step (fewer than 16 rows, `KT_MXFP4_CACHE_KEEP_ROWS`) keeps what it reads, least
  recently used out first; wider forwards read their misses for themselves and leave the cache alone.
- Reads are O_DIRECT, 1 MiB at a time, on `KT_MXFP4_CACHE_READERS` threads (16) on `KT_IO_CPUS`.
  A forward computes its cached experts while the missing ones are read
  (`KT_MXFP4_CACHE_OVERLAP`, 1); the weighted sum is one pass over the routing order either way.

**What it does** (SGLang, this patch):

- The streamed prefill maps the cache's memfd and registers it as pinned. A group's cached experts
  are copied from their slots; the others are read from the checkpoint into a pinned bounce first
  (`KT_MXFP4_STREAM_READERS`, 8). The scale bytes become fp32 scales on the GPU (e << 23).
- The dynamic hot slots copy a swapped-in expert from its cache slot, which stays pinned in the
  cache until the copy is done (an expert the cache does not hold is read from the checkpoint).

**Effect** (one EPYC 7452 and its 251 GiB, one RTX 5070 Ti, `numactl -N 0 -m 0`, one launch each):

- Without the cache the server does not start on one node.
- Cache 160 GiB: Japanese prose 16.1 tok/s, 35K prefill 451-470 tok/s.
- Cache 200 GiB: 18.1 tok/s and 519-598 tok/s; decode reads half as many experts again
  (2.9% of the expert uses miss, against 5.1%).
- Reading while computing: 18.7 tok/s with the decode's reads, 19.1 with the prefill's too.
- The dynamic hot slots from the cache: 19.8 tok/s; the other texts 14.8-16.7 (13.6-15.7 without).
- The wall is the one socket's memory bandwidth in decode, and the drive (6.5 GB/s here) for the
  experts a prefill touches that the cache does not hold: a 400-token prompt takes 7.8 s.

Numerics: bit-identical. Two layers loaded both ways (`KT_MXFP4_EXPERT_CACHE_GIB` 0 and 4) give
byte-identical outputs for 34 forwards of 1, 8 and 128 rows, through misses, evictions and the
reads of wide forwards, with and without the overlap. The server's 24-piece NLL is 0.7224 with
the hot slots held fixed (cache 160 and 200 GiB: 0.7223, 0.7224; the hot slots differ a little
between launches).

Switches (code / serve.sh):

- `KT_MXFP4_EXPERT_CACHE_GIB`: unset (off) / the node's memory less `KT_MXFP4_HOST_RESERVE_GIB`
  (50), and 6 GiB a GPU for HiCache's host copy of the prefix cache when it is on ([21](#21-hicache)),
  when the node has less than `KT_MXFP4_FULL_GIB` (370) and one pool; 0 keeps every expert in RAM
- `KT_MXFP4_CACHE_KEEP_ROWS`: 16
- `KT_MXFP4_CACHE_READERS`: 16; `KT_IO_CPUS`: unset / the second hardware threads of the pool's cores
- `KT_MXFP4_CACHE_OVERLAP`: 1
- `KT_MXFP4_STREAM_READERS`: 8
- `KT_MXFP4_CACHE_REPORT=<forwards>`, `KT_MXFP4_STREAM_REPORT=<layers>` (40): the cache's counters

## 21 hicache

`serve.sh` runs the prefix cache with SGLang's HiCache: `--enable-hierarchical-cache
--hicache-write-policy write_back --hicache-ratio 2`. A prompt that starts like an earlier one reads
that part from the cache, and the cache's pages are also kept in host memory, so a prefix the GPU
pool has given up is loaded back instead of computed again. The host copy takes twice the GPU
pool, about 5 GiB a GPU with the 1M pool. This patch makes HiCache work with the lends:

- A load-back maps its SWA pages first. The SWA lend keeps the SWA pool's free pages beyond
  `KT_PREFILL_SWA_LEND_KEEP` (4,096 tokens) unmapped outside prefill and maps the ones an
  allocation hands out at the next forward's start. HiCache allocates a load-back's SWA slots in
  the scheduler and starts the copy before that forward, into memory not mapped yet: the third
  prefix loaded back faulted both ranks. The cache controller's `start_loading` now maps them first.
- The write-back staging buffer takes at most `SGLANG_HICACHE_STAGING_MB` (32) MiB of GPU memory.
  A V4 host pool kept 64 pages of all its layers on the GPU, 330 MiB for the SWA pool, memory the
  hot lend's slots did not get. The staged kernel already loops over the buffer.
- With HiCache attached the SWA trim keeps `KT_PREFILL_SWA_LEND_KEEP` tokens outside prefill
  instead of the pool's budget (34,048 tokens); what the cache gives back goes to host memory and
  comes back on the next match. A cached 35K prompt's SWA kept on the GPU (0.72 GiB) left the next
  decode's lend 592 slots instead of 684.

**Effect** (the machine in the README, two GPUs, against the prefix cache off; launches alternated):

- Decode on all seven texts and the 35K prefill as without the cache (the first lend: 690 hot
  slots against 695).
- The same 35K prompt again: 2.0 s, 34,816 tokens from the cache, against 20 s.
- Six 4,096-token documents trimmed from the GPU all come back from host memory on their next read.
- Four sessions of 900K, 131K, 66K and 33K tokens (1.13M, past the 1M GPU pool) with fresh
  200K prompts between rounds, four requests at once: 28 of 28 facts read back, the 900K session
  again in 5-17 s instead of 855 s.
- One GPU (the KV past it in host memory): the 35K prefill (37.9 s, 33.2 s) and decode as without
  the cache; a 131K prompt again in 2.3 s instead of 127 s.

The chunk calibration of a setup's first launch (`lib/kt-lend-auto.sh`) runs without the cache:
its one long prompt is chunked before any plan is known, and the cache would hold every chunk's
SWA pages (on one GPU every calibration ran out of SWA pages and the chunk fell to 2,048).

Numerics: a prefix read from the cache scores as computed (six documents, NLL 0.6267-0.6328 first
read, 0.6280-0.6341 from the cache, against 0.011 a document between two uncached reads). With the
hot slots held fixed (`KT_KV_HOT_SLOTS_MAX=600`) the 24-piece NLL is 0.7169 with the cache off, on,
and on with HiCache; without the cap HiCache's 5 fewer slots give 0.7208 instead of 0.7185.

Switches (code / serve.sh):

- the prefix cache: SGLang's / on; `DSV41_RADIX=0` turns it off
- HiCache: off / on with this patch; `DSV41_HICACHE=0` keeps the cache on the GPU only (it then
  holds memory the hot lend would use); `DSV41_HICACHE_RATIO`: 2
- `SGLANG_HICACHE_STAGING_MB`: 32
- `KT_PREFILL_SWA_HOST_TRIM`: 1 (only with HiCache attached)

Prerequisites: `kv-vmm`, `prefill-lend`, `kv-lend-plan`.

Files (under `python/sglang/`):

- `srt/layers/moe/kt_kv_prefill_lend.py`
- `srt/mem_cache/memory_pool_host.py`

## 22 prefill-warmup

`--warmups prefill_lengths` runs one fresh random prompt of each length in
`SGLANG_WARMUP_PREFILL_LENGTHS` (comma separated) before the server reports ready, with SGLang's
custom warmups. `serve.sh` sets `1024,12288`; an empty value skips it.

`cuModuleLoadData` synchronizes the context. The first launch of a Triton kernel variant (Triton
specializes on whether the row count divides by 16, and the mHC kernels pick tiles by row count)
loads its cubin from Triton's cache and waits for every stream to drain, and a CUDA kernel loaded
lazily at its first launch does the same. A streamed prefill keeps a deep queue (the PCIe expert
copies, the MoE GEMMs, the engram row lookups as host callbacks), so the first request of a launch
paid those drains inside its forward: on one RTX 5070 Ti a 1,024-token prompt took 6.1 s instead
of 4.4 s, and the first 35,000-token prompt 37.4 s instead of 33.3 s, 6.9 s of it in one load of
the mix-stats kernel's 32-row tile behind a 12,288-row chunk's engram lookup. With the two warmup
prompts the first 1,024-token prompt takes 4.4 s and the first 35,000-token prompt 33.5 s (the
second 33.3 s); the server is ready 20 s later. `CUDA_MODULE_LOADING=EAGER` is not an alternative:
every kernel resident at load left no memory for the KV pool.

- `SGLANG_WARMUP_PREFILL_LENGTHS`: unset / `1024,12288`

## 23 swa-checkpoint-tail

A prompt sent again after its first request has answered is read from the prefix cache up to the
end of the prompt, as after a request that ended inside its prefill.

V4.1 runs with decoder SWA bounded replay, so a prefix match stops a sliding window (128 tokens)
short of the prompt's end, and the match needs live SWA in the window before that point. When
SGLang puts a running request's prefix in the tree (at the end of its prefill), it first frees the
SWA slots out of the window, measured from the prompt's end. That freed the window the later match
needs, and the match fell back to the previous chunk boundary. This patch starts the free one
window earlier; a checkpoint keeps one more page of SWA.

On two RTX 5070 Ti, a 32,625-token prompt sent again after a request that generated 2 to 512
tokens read 16,384 tokens from the cache before (16,241 computed again) and 32,256 with the patch
(369). Prompts of 33K, 65K and 131K tokens after 4,096 generated tokens took 4.8-10.6 s the second
time before, against 2.7 s. Four sessions of 786K, 131K, 65K and 33K tokens, each answered and
then sent again together, computed 263-369 tokens again a session, against 3,079-16,241 before.
`tools/qualify.sh` gives the same NLL, decode, 35K prefill and cached 35K time as without it.

- no switch (`SGLANG_FEATURES` without `swa-checkpoint-tail` leaves it out)

## 24 rope-tables

The RoPE tables take GPU memory only under the positions the served context reaches, and the
copy nothing reads is not built; the memory goes to the hot experts.

Each attention layer reads one of two complex64 `freqs_cis` tables (the plain base and the
compressed YaRN base), each built for `max_position_embeddings`: 256 MiB each at 1M positions,
whatever context is served. Each layer also built `rotary_emb`, whose fp32 cos/sin cache holds the
same angles again (512 MiB); on CUDA nothing reads it, so it is not built there. With the switch on,
each `freqs_cis` table moves after loading to a `kv-vmm` range of the same size whose 2 MiB chunks
are mapped only for the rows a forward can reach (the first 65,536, then the batch's longest
sequence plus 16,384), filled from a pinned host copy. Kernels, views and CUDA graphs keep their
addresses. A growth takes 6-32 ms; at 1M the tables are whole again.

Measured with `embed-on-host`; see there.

- `SGLANG_DSV4_ROPE_LAZY`: 0 / 1 (`rotary_emb` is not built on CUDA either way)
- `SGLANG_DSV4_ROPE_LAZY_VERIFY`: 0 / 0 (1 compares the mapped rows with the host copy after each
  growth)

## 25 embed-on-host

The token embedding is read from pinned host memory, one row a decode token; its GPU memory (1.32
GB on one GPU, half on each of two) goes to the hot experts.

After loading, next to the vision tower's offload, the table moves to pinned host memory, and the
embedding becomes a Triton gather that reads the rows through UVA. The values are copies, rows
outside a GPU's vocabulary shard are zero as before, and the all-reduce is unchanged.

`tools/qualify.sh` on the machine in the README, against the full stack without the two patches
(same day):

| GPUs | patches | hot slots | decode median | 35K first | 35K second | NLL minus ref |
|---|---|---|---|---|---|---|
| 1 | without | 61 | 33.3 | 34.15 s | 34.09 s | +0.0026 |
| 1 | `rotary_emb` not built, `embed-on-host` | 162 | 36.9 | 29.83 s | 29.55 s | +0.0013 |
| 1 | both | 189 | 37.9 | 29.88 s | 30.05 s | -0.0020 |
| 2 | without | 661 | 51.5 | 22.38 s | 21.18 s | +0.0002 |
| 2 | `rotary_emb` not built, `embed-on-host` | 780 | 52.8 | 22.43 s | 21.10 s | +0.0004 |
| 2 | both | 835 | 53.9 | 22.65 s | 21.17 s | +0.0046 |

On one GPU the 35K prefill is faster too; the prompt runs as two chunks of 18,432 tokens with and
without the patches.

- `SGLANG_DSV4_EMBED_ON_HOST`: 0 / 1

## 26 gguf-lend

Tier 1's features with a GGUF pack under them (tier 0.5, [gguf.md](gguf.md)):

- The streamed prefill's GGUF edition (`kt_stream_prefill_gguf.py`, `kt_stream_prefill_gguf_nvme.py`):
  the experts from RAM or the drive through llama.cpp's MMQ, added into the MoE output.
- The prefill lend's copy of the dense weights read back from the GGUF shards instead of kept in
  pinned host memory (`KT_PREFILL_LEND_MIRROR=gguf`).
- The hot lend's GGUF slots, their swaps loaded on background threads from kt-kernel's GGUF expert
  cache or the file, and the dynamic hot cache that follows the decode's routing
  (`KT_HOT_DYNAMIC=1`, `kt_hot_cache.py`).
- The prefill plan may run a chunk up to `KT_PREFILL_PLAN_TAIL_STEPS` aligned steps over the launch
  chunk when that spares a chunk.

All of it is gated on a GGUF pack (`KT_GGUF_V41_DENSE`, or kt-kernel's LLAMAFILE method). With the MXFP4
checkpoint the patch changes nothing: the plan's tail steps are 0, the lend keeps its pinned copy,
and the free-memory reads and the swap budget are as before.
