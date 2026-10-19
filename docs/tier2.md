# Tier 2, one patch per feature

Tier 2 is one patch per feature in `patches/sglang-tier2/`, applied in the order of their numbers on
SGLang `1aa0e962` after tiers 0 and 1; tier 3 applies on top. `setup.sh` applies all of them by
default; `SGLANG_FEATURES=a,b,...` picks some (the names below), and `serve.sh` turns on only what
was applied.

A patch with no prerequisites applies alone on tier 1. A patch with prerequisites applies on tier 1
plus those patches (`patches/sglang-tier2/requires.txt`, checked by `setup.sh`). Combinations that
apply: each patch alone, all of them, and all of them but any one (with what needs it); other
combinations may need `git apply -3`. Served and checked end to end: all of them, none, and each
set measured in [Effect of each patch](#effect-of-each-patch).

Defaults are given as *code / serve.sh*: what the Python does with the variable unset, and what
`serve.sh` (with `lib/kt-kv-hot-default.sh` and `lib/kt-lend-auto.sh`) sets. Switches marked "on in
the code" are on as soon as the patch is applied.

| # | patch | lines (+/-) | prerequisites |
|---|---|---|---|
| 01 | [`01-fp8-decode.patch`](#01-fp8-decode) | +241 / -1 | none |
| 02 | [`02-gpu-moe-fp32.patch`](#02-gpu-moe-fp32) | +487 / -1 | none |
| 03 | [`03-cutlass-kt-expert-count.patch`](#03-cutlass-kt-expert-count) | +6 / -1 | none |
| 04 | [`04-engram-row-prefetch.patch`](#04-engram-row-prefetch) | +47 / -0 | none |
| 05 | [`05-wo-a-fp8.patch`](#05-wo-a-fp8) | +222 / -2 | none |
| 06 | [`06-vision-on-demand.patch`](#06-vision-on-demand) | +90 / -12 | `wo-a-fp8` |
| 07 | [`07-thinking-default-parser.patch`](#07-thinking-default-parser) | +6 / -4 | none |
| 08 | [`08-loading.patch`](#08-loading) | +234 / -3 | none |
| 09 | [`09-launch-timing.patch`](#09-launch-timing) | +402 / -0 | none |
| 10 | [`10-pool-swa-guard.patch`](#10-pool-swa-guard) | +17 / -0 | none |
| 11 | [`11-placement.patch`](#11-placement) | +68 / -0 | none |
| 12 | [`12-kv-vmm.patch`](#12-kv-vmm) | +243 / -0 | none |
| 13 | [`13-prefill-lend.patch`](#13-prefill-lend) | +1381 / -2 | none |
| 14 | [`14-stream-prefill.patch`](#14-stream-prefill) | +1890 / -14 | `loading`, `placement`, `prefill-lend` |
| 15 | [`15-kv-lend-plan.patch`](#15-kv-lend-plan) | +1736 / -5 | `kv-vmm`, `prefill-lend` |
| 16 | [`16-hot-lend.patch`](#16-hot-lend) | +26 / -1 | `kv-vmm`, `prefill-lend`, `stream-prefill`, `kv-lend-plan` |

## Effect of each patch

Each patch added to a base, one launch each, on the machine in the README (2 × EPYC 7452, 2 × RTX
5070 Ti), `TIER=3` with the kt-kernel patch and `SGLANG_FEATURES` set to the base plus the patch.
Decode is the median of the server's generation throughput over four 1,024-token Japanese prose
answers; free VRAM is what the server reports after loading (`available_gpu_mem`, GPU 0).

Decode and memory, on tiers 0, 1 and 3 with no tier-2 patch (`SGLANG_FEATURES=none`):

| added | decode (tok/s) | free VRAM (GB) |
|---|---|---|
| nothing (base) | 15.9 | 3.23 |
| `fp8-decode` | 27.7 | 3.23 |
| `placement` | 18.2 | 3.23 |
| `engram-row-prefetch` | 16.3 | 3.22 |
| `wo-a-fp8` | 16.1 | 3.81 |
| `wo-a-fp8`, `vision-on-demand` | 16.1 | 4.76 |
| `gpu-moe-fp32` | 15.9 | 3.27 |
| `cutlass-kt-expert-count`, `thinking-default-parser`, `loading`, `launch-timing`, `pool-swa-guard`, `kv-vmm`, `prefill-lend` (each alone) | 15.8 to 15.9 | 3.16 to 3.23 |

The patches in the last row do not aim at decode: `loading` and `launch-timing` are about startup,
`kv-vmm` and `prefill-lend` about prefill (below), the others fixes and guards. The gains do not
add up: a token waits for the slower of the CPU and GPU experts in each layer, so a cut on one
side shows only where that side was the slower.

A 35,000-token prompt, on the streamed-prefill set (`loading`, `placement`, `prefill-lend`,
`stream-prefill`; without it the prompt's experts run on the CPU). The first prompt after launch
pays a one-time warm-up; the second is a new 35,000-token prompt on the warm server:

| added | 35K, first (s) | 35K, second (s) | decode (tok/s) | free VRAM (GB) |
|---|---|---|---|---|
| no tier-2 patch (the experts on the CPU) | 846.5 | | | |
| the streamed-prefill set (base) | 83.1 | 65.4 | 17.9 | 2.65 |
| `gpu-moe-fp32` | 80.6 | 62.7 | 17.8 | 2.67 |
| `wo-a-fp8` | 83.0 | 65.4 | 18.2 | 3.24 |
| `kv-vmm`, `kv-lend-plan` | 65.3 | 47.5 | 17.9 | 2.75 |
| `kv-vmm`, `kv-lend-plan`, `hot-lend` | 60.6 | 42.2 | 18.7 | 2.72 |
| all sixteen | 54.3 | 35.3 | 46.0 | 4.23 |

Numerics: the 2-token NLL check gives the base's value for every patch above but four.
`fp8-decode` and `gpu-moe-fp32` change the GPU arithmetic (see their Numerics lines; `gpu-moe-fp32`
gives the same value again on a second launch), and `placement` and `hot-lend` change which experts
run on the GPU.

## 01 fp8-decode

Faster block-FP8 dense GEMMs for decode-sized inputs. With up to four rows, `w8a8_block_fp8_matmul` runs a Triton GEMV (`fp8_block_gemv.py`) that steps K over several scale groups at a time. With up to 16 rows and no tuned config for the shape, the Triton matmul uses a split-K tiling. `SGLANG_FP8_DECODE_SPLITK=1` also sends the decode rows (up to 16) of the CUTLASS block-FP8 linear through that Triton matmul, with the operands CUTLASS would have been given; prefill stays on CUTLASS.

Switches (code / serve.sh):

- `SGLANG_FP8_GEMV`: 1 (on in the code) / 1
- `SGLANG_FP8_SKINNY`: 1 (on in the code); SGLANG_FP8_SKINNY_SPLIT_K 4, _BLOCK_N 128, _LOG / unset (on)
- `SGLANG_FP8_DECODE_SPLITK`: 0 / unset (off)

Numerics: Not bit-identical when on: the same FP8 operands summed in a different fp32 order (GEMV, split-K). Prefill shapes are unchanged.

Prerequisites: none; applies alone on tier 1.

Files (under `python/sglang/`):

- `kernels/ops/quantization/fp8_block_gemv.py` (new)
- `kernels/ops/quantization/fp8_kernel.py`
- `srt/layers/quantization/fp8_utils.py`

## 02 gpu-moe-fp32

The GPU-resident MXFP4 experts in fp32 glue. `SGLANG_MXFP4_MOE_FP32=1` (with `--moe-runner-backend marlin`) replaces the Marlin MoE method with `Mxfp4Fp32MoEMethod` (`mxfp4_fp32_moe.py`): Triton kernels that read the checkpoint's MXFP4 layout and round to bf16 once, at the weighted sum; decode uses a per-route GEMV, larger batches the tensor cores on the routes with a weight. When the experts run through Marlin instead, `SGLANG_MARLIN_MOE_FP32_GLUE=1` keeps the steps around Marlin's two GEMMs in fp32.

Switches (code / serve.sh):

- `SGLANG_MXFP4_MOE_FP32`: 0; SGLANG_MXFP4_MOE_FP32_SYNC (debug) / 1
- `SGLANG_MARLIN_MOE_FP32_GLUE`: 0 / 1

Numerics: Not bit-identical to Marlin: fewer bf16 roundings, closer to an fp32 reference. The fp32 MoE repeats bit for bit from launch to launch.

Prerequisites: none; applies alone on tier 1.

Files (under `python/sglang/`):

- `srt/layers/moe/fused_moe_triton/fused_marlin_moe.py`
- `srt/layers/moe/mxfp4_fp32_moe.py` (new)
- `srt/layers/quantization/fp8.py`
- `srt/layers/quantization/mxfp4_marlin_moe.py`

## 03 cutlass-kt-expert-count

A fix in the FlashInfer CUTLASS MXFP4 MoE method: the expert count is taken from the weight tensor (`layer.w13_weight.shape[0]`) instead of `layer.num_local_experts`, because under the kt-kernel wrapper the method holds only the GPU-resident experts.

Numerics: No switch. Identical without KTransformers (both counts agree); under KTransformers the CUTLASS MXFP4 path used to fail.

Prerequisites: none; applies alone on tier 1.

Files (under `python/sglang/`):

- `srt/layers/quantization/mxfp4_flashinfer_cutlass_moe.py`

## 04 engram-row-prefetch

In decode, each engram layer's rank-local row lookup starts at the beginning of the token on a side stream instead of when its layer comes. The all-reduce, `wkv` and the gate stay where they were. Only with one pipeline stage, no DP attention and engram tables that are not shared.

Switches (code / serve.sh):

- `SGLANG_DSV41_ENGRAM_PREFETCH_ROWS`: 1 (on in the code) / 1

Numerics: Bit-identical: only the lookup moves.

Prerequisites: none; applies alone on tier 1.

Files (under `python/sglang/`):

- `srt/models/deepseek_v4.py`

## 05 wo-a-fp8

DeepSeek V4 attention's `wo_a` in FP8 (`deepseek_common/wo_a_fp8_gemv.py`). `SGLANG_DSV4_WO_A_FP8_NATIVE=1` keeps the checkpoint's FP8 `wo_a` as loaded (no bf16 copy): decode reads it with an FP8 GEMV, prefill dequantizes it into one shared bf16 buffer per device. `SGLANG_DSV4_WO_A_FP8_GEMV=1` adds an exact FP8 copy of a bf16 `wo_a` for decode, and `SGLANG_DSV4_WO_A_FP8_ONLY=1` then frees the bf16 weight.

Switches (code / serve.sh):

- `SGLANG_DSV4_WO_A_FP8_NATIVE`: 0 / 1
- `SGLANG_DSV4_WO_A_FP8_GEMV`: 0 / unset (off)
- `SGLANG_DSV4_WO_A_FP8_ONLY`: 0 / unset (off)

Numerics: Decode GEMV not bit-identical (fp32 summation order; 99.98% of outputs equal, the rest 1 ulp). The prefill dequantize path is bit-identical (the same bf16 values).

Prerequisites: none; applies alone on tier 1.

Files (under `python/sglang/`):

- `srt/models/deepseek_common/wo_a_fp8_gemv.py` (new)
- `srt/models/deepseek_v4.py`

## 06 vision-on-demand

The vision tower's GPU memory when serving text. `SGLANG_DSV4_VISION_ON_DEMAND=1` moves the tower and aligner to pinned host memory after loading and puts them on the GPU only while a batch's images are encoded (if the KV plan of `kv-lend-plan` is applied, it accounts for the copy). `--language-model-only` makes DeepSeek V4 build and load no tower at all, as SGLang's flag does for other multimodal models.

Switches (code / serve.sh):

- `SGLANG_DSV4_VISION_ON_DEMAND`: 0 / 1
- `--language-model-only (CLI)`: off / not passed

Numerics: Bit-identical: placement only (image batches pay a host-to-device copy). `--language-model-only` leaves text output unchanged.

Prerequisites: `wo-a-fp8`.

Files (under `python/sglang/`):

- `srt/models/deepseek_v4.py`

## 07 thinking-default-parser

When a request sends no thinking toggle in `chat_template_kwargs`, the reasoning parser falls back to `SGLANG_DEFAULT_THINKING`, the same fallback the DeepSeek V4 encoder uses to decide whether the prompt opens a thinking block, so the parser and the prompt agree. Before, a missing toggle always meant no thinking for the parser.

Switches (code / serve.sh):

- `SGLANG_DEFAULT_THINKING (existing)`: 0 / unset

Numerics: No switch of its own. Identical unless `SGLANG_DEFAULT_THINKING=1`.

Prerequisites: none; applies alone on tier 1.

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

Prerequisites: none; applies alone on tier 1.

Files (under `python/sglang/`):

- `srt/layers/moe/kt_ep_wrapper.py`
- `srt/model_loader/weight_utils.py`
- `srt/models/deepseek_v4.py`

## 09 launch-timing

`KT_LAUNCH_TIMING=1` prints a per-step timeline of the weight load on each rank, with the disk bytes each step read, and sums the steps that repeat (tensor fetches, a layer's CPU experts) in a table at the end. `KT_LAUNCH_TIMING_STACKS=<seconds>` also samples the loading thread's stack. It wraps the load's functions in place; the ones from patches that are not applied are skipped.

Switches (code / serve.sh):

- `KT_LAUNCH_TIMING`: 0; KT_LAUNCH_TIMING_STACKS / unset (off)

Numerics: No effect on outputs.

Prerequisites: none; applies alone on tier 1.

Files (under `python/sglang/`):

- `srt/layers/moe/kt_launch_timing.py` (new)
- `srt/model_executor/model_runner.py`

## 10 pool-swa-guard

A startup check in the pool configurator for DeepSeek V4: an SWA pool smaller than one prefill chunk plus a page raises a ValueError that names the flags to change (`--swa-full-tokens-ratio`, `--max-total-tokens`, `--chunked-prefill-size`), instead of a scheduler that re-queues the request forever with the GPU idle.

Numerics: No switch. No effect on outputs; a setup that used to hang now fails at startup.

Prerequisites: none; applies alone on tier 1.

Files (under `python/sglang/`):

- `srt/model_executor/pool_configurator.py`

## 11 placement

Expert placement in the kt-kernel wrapper. The V4 router hands the wrapper logical expert ids; with a non-identity `--init-expert-location` they are mapped to physical ids once per layer, so a hot-expert placement (`placements/`) routes to the GPU slots. `KT_ROUTING_DUMP=<prefix>` saves a per-layer histogram of the routed ids every 30 s (a captured `index_add_`, so it counts under CUDA graph replay), to build a placement from a workload.

Switches (code / serve.sh):

- `--init-expert-location (existing CLI)`: identity / a file from placements/
- `KT_ROUTING_DUMP`: unset (off) / unset

Numerics: Changes outputs under a placement (without it the wrapper ran logical ids in physical slots, i.e. the wrong experts); identical without one. The dump does not change outputs.

Prerequisites: none; applies alone on tier 1.

Files (under `python/sglang/`):

- `srt/layers/moe/kt_ep_wrapper.py`

## 12 kv-vmm

`KT_KV_VMM=1` puts the KV pool on memory whose physical pages can be unmapped and mapped again while the addresses, and the CUDA graphs that read them, stay the same (`kt_kv_vmm.py`, hooked into the torch_memory_saver adapter). Needs `libkv_vmm.so` from the repo's `kv-vmm/kv_vmm.c` (`KT_KV_VMM_LIB`).

Switches (code / serve.sh):

- `KT_KV_VMM`: 0; KT_KV_VMM_LIB, KT_KV_VMM_DIRECT / 1

Numerics: Bit-identical.

Prerequisites: none; applies alone on tier 1.

Files (under `python/sglang/`):

- `srt/layers/moe/kt_kv_vmm.py` (new)
- `srt/utils/torch_memory_saver_adapter.py`

## 13 prefill-lend

`KT_PREFILL_LEND=1`: during a streamed prefill the GPU-resident weights give their memory to the prefill's scratch and come back layer by layer, at the same addresses, just before each layer runs (`kt_lend.py`, hooked into the loader and the model runner's forward). `KT_PREFILL_LEND_PROBE` measures how large a chunk fits before serving (`kt_lend_probe.py`, used by `lib/kt-lend-auto.sh`). `KT_FAST_PINNED=1` pins the lend's host buffers as huge-page memory registered with `cudaHostRegister` instead of `cudaHostAlloc` (`kt_pinned.py`). DeepSeek V4's padded query buffer is allocated in the prefill's scratch. The lend calls into the streamed prefill, the KV lend and the hot lend only when those patches are applied and on.

Switches (code / serve.sh):

- `KT_PREFILL_LEND`: 0; KT_PREFILL_LEND_HINT, _MARGIN_MB, _SKIP, _LONG_SEQ, _MAX_CHUNK, ... / 1 (via lib/kt-lend-auto.sh)
- `KT_PREFILL_LEND_PROBE`: unset / set by lib/kt-lend-auto.sh on the calibration launch
- `KT_FAST_PINNED`: 0 / 0

Numerics: Bit-identical at a given chunk size. It lets the launcher pick a larger chunk, and the chunk size moves NLL.

Prerequisites: none; applies alone on tier 1.

Files (under `python/sglang/`):

- `srt/layers/moe/kt_lend.py` (new)
- `srt/layers/moe/kt_lend_probe.py` (new)
- `srt/layers/moe/kt_pinned.py` (new)
- `srt/managers/scheduler.py`
- `srt/model_executor/model_runner.py`
- `srt/model_loader/loader.py`
- `srt/models/deepseek_v4.py`

## 14 stream-prefill

`KT_GPU_STREAM_PREFILL=<tokens>`: above that many tokens the CPU experts are not computed by kt-kernel; their weights are copied to the GPU in groups and computed there, the copy of the next group overlapping the compute of the current one (`kt_stream_prefill.py`). `KT_GPU_STREAM_W16=1` dequantizes each group to bf16 and runs FlashInfer's bf16 CUTLASS MoE (SM89+); otherwise the SM120 MXFP8 x MXFP4 kernel. `KT_GPU_STREAM_ZEROCOPY` reads kt-kernel's shared expert memory directly and `KT_GPU_STREAM_GRAPH` launches each group as one CUDA graph. One CUTLASS MoE workspace per GPU is preallocated before the decode graphs are captured, for the resident experts as well (this part is always on), and `KT_GPU_STREAM_ROW_BLOCK` splits long chunks into row blocks. Also: early streamer init, CPU groups (`KT_STREAM_CPU_GROUPS`), a reference check (`KT_HOT_CHECK`), and comment updates in the wrapper.

Switches (code / serve.sh):

- `KT_GPU_STREAM_PREFILL`: 0 (off) / 512
- `KT_GPU_STREAM_GROUP`: 32 / 4
- `KT_GPU_STREAM_W16`: 0 / 1
- `KT_GPU_STREAM_ZEROCOPY`: 0 / 1
- `KT_GPU_STREAM_GRAPH`: 1 (on in the code) / 1
- `KT_GPU_STREAM_EARLY_INIT`: 0 / 1
- `KT_GPU_STREAM_ROW_BLOCK`: 0 / 4096 (lib/kt-lend-auto.sh)
- `KT_STREAM_CPU_GROUPS, KT_HOT_CHECK`: 0 / unset

Numerics: Not bit-identical when on: the streamed CPU experts run on the GPU (W16: bf16 rounding instead of kt-kernel's; the MXFP8 mode has a larger error). The shared workspace changes memory only. ZEROCOPY needs the kt-kernel tree patch (`KT_EXPERT_SHM`).

Prerequisites: `prefill-lend`, `loading`, `placement`.

Files (under `python/sglang/`):

- `srt/layers/moe/kt_ep_wrapper.py`
- `srt/layers/moe/kt_stream_prefill.py` (new)
- `srt/layers/moe/moe_runner/flashinfer_cutlass.py`
- `srt/layers/quantization/mxfp4_flashinfer_cutlass_moe.py`

## 15 kv-lend-plan

The KV pool's free pages during prefill. `KT_PREFILL_KV_LEND=1` lends the pool's free pages to the prefill (pages in use are never lent; the pool never shrinks). `KT_PREFILL_PLAN=1` plans each request's chunk size from the memory free at that moment, so a long prompt takes fewer, larger chunks (scheduler and schedule_batch hooks). `KT_PREFILL_SWA_LEND=1` sizes the sliding-window pool for the largest planned chunk and releases its free pages outside prefill. The file `kt_kv_hot_lend.py` ships here because the plan imports it; it does nothing until `hot-lend` wires it in.

Switches (code / serve.sh):

- `KT_PREFILL_KV_LEND`: 0; KT_PREFILL_KV_LEND_* / 1 (lib/kt-kv-hot-default.sh)
- `KT_PREFILL_PLAN`: 0; KT_PREFILL_PLAN_* / 1
- `KT_PREFILL_SWA_LEND`: 0; KT_PREFILL_SWA_* / 1

Numerics: The lend is bit-identical. The plan picks per-request chunk sizes, so the prefill chunking (and NLL) moves against a fixed chunk. The SWA lend changes memory only.

Prerequisites: `kv-vmm`, `prefill-lend`.

Files (under `python/sglang/`):

- `srt/layers/moe/kt_kv_hot_lend.py` (new)
- `srt/layers/moe/kt_kv_prefill_lend.py` (new)
- `srt/layers/moe/kt_prefill_plan.py` (new)
- `srt/managers/schedule_batch.py`
- `srt/managers/scheduler.py`
- `srt/mem_cache/kv_cache_configurator.py`

## 16 hot-lend

`KT_KV_HOT_LEND=1`: outside prefill the KV pool's free pages hold up to `KT_KV_HOT_EXTRA` more experts a layer on the GPU (the next-hottest after the resident ones, with a placement), on a shadow layer; their routes go to the GPU and to the CPU at weight 0. The experts leave whenever the pool needs the pages or a prefill starts. `KT_KV_HOT_SPARE_FLOOR_MB` lets the slots also use idle driver memory above the floor.

Switches (code / serve.sh):

- `KT_KV_HOT_LEND`: 0 / 1 (lib/kt-kv-hot-default.sh)
- `KT_KV_HOT_EXTRA`: 0 / 16
- `KT_KV_HOT_SPARE_FLOOR_MB`: -1 (off) / 1024
- `KT_KV_HOT_SLOTS_MAX, _RESERVE, _MIN_MB, _RETRY, _MARGIN_MB, _CHECK*`: unset / unset

Numerics: Not bit-identical: the lent experts run in the GPU MoE kernel instead of kt-kernel, and how many are lent depends on the free KV pages at the time, so decode output depends on load.

Prerequisites: `kv-vmm`, `prefill-lend`, `stream-prefill`, `kv-lend-plan`.

Files (under `python/sglang/`):

- `srt/layers/moe/fused_moe_triton/layer.py`
- `srt/layers/moe/kt_ep_wrapper.py`
