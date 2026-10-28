# Tier 1, one patch per feature

Tier 1 is one patch per feature in `patches/sglang-tier1/`, applied in the order of their numbers on
SGLang main at `5d8e90a0` after tier 0; tier 2 applies on top. `setup.sh` applies all of them by
default; `SGLANG_FEATURES=a,b,...` picks some (the names below), and `serve.sh` turns on only what
was applied.

A patch with no prerequisites applies alone on tier 0. A patch with prerequisites applies on tier 0
plus those patches (`patches/sglang-tier1/requires.txt`, checked by `setup.sh`). Combinations that
apply: each patch alone, all of them, and all of them but any one (with what needs it); other
combinations may need `git apply -3`. Served and checked end to end: all of them, none, and each
set measured in [Effect of each patch](#effect-of-each-patch).

Defaults are given as *code / serve.sh*: what the Python does with the variable unset, and what
`serve.sh` (with `lib/kt-kv-hot-default.sh` and `lib/kt-lend-auto.sh`) sets. Switches marked "on in
the code" are on as soon as the patch is applied.

| # | patch | lines (+/-) | prerequisites |
|---|---|---|---|
| 01 | [`01-fp8-decode.patch`](#01-fp8-decode) | +222 / -1 | none |
| 02 | [`02-gpu-moe-fp32.patch`](#02-gpu-moe-fp32) | +487 / -1 | none |
| 03 | [`03-cutlass-kt-expert-count.patch`](#03-cutlass-kt-expert-count) | +6 / -1 | none |
| 04 | [`04-engram-row-prefetch.patch`](#04-engram-row-prefetch) | +71 / -6 | none |
| 05 | [`05-wo-a-fp8.patch`](#05-wo-a-fp8) | +174 / -2 | none |
| 06 | [`06-vision-on-demand.patch`](#06-vision-on-demand) | +87 / -11 | `wo-a-fp8` |
| 07 | [`07-thinking-default-parser.patch`](#07-thinking-default-parser) | +6 / -4 | none |
| 08 | [`08-loading.patch`](#08-loading) | +250 / -2 | none |
| 09 | [`09-launch-timing.patch`](#09-launch-timing) | +402 / -0 | none |
| 10 | [`10-pool-swa-guard.patch`](#10-pool-swa-guard) | +17 / -0 | none |
| 11 | [`11-placement.patch`](#11-placement) | +45 / -0 | none |
| 12 | [`12-kv-vmm.patch`](#12-kv-vmm) | +243 / -0 | none |
| 13 | [`13-prefill-lend.patch`](#13-prefill-lend) | +1664 / -1 | none |
| 14 | [`14-stream-prefill.patch`](#14-stream-prefill) | +2045 / -14 | `loading`, `placement`, `prefill-lend` |
| 15 | [`15-kv-lend-plan.patch`](#15-kv-lend-plan) | +1751 / -5 | `kv-vmm`, `prefill-lend` |
| 16 | [`16-hot-lend.patch`](#16-hot-lend) | +674 / -34 | `kv-vmm`, `prefill-lend`, `stream-prefill`, `kv-lend-plan` |
| 17 | [`17-host-kv.patch`](#17-host-kv) | +477 / -7 | `kv-vmm`, `prefill-lend`, `stream-prefill`, `kv-lend-plan`, `hot-lend` |
| 18 | [`18-stream-mmq.patch`](#18-stream-mmq) | +20478 / -3 (of it +20,300 the vendored llama.cpp kernels) | `stream-prefill` |

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

Switches (code / serve.sh):

- `KT_GPU_STREAM_PREFILL`: 0 (off) / 512
- `KT_GPU_STREAM_GROUP`: 32 / 4
- `KT_GPU_STREAM_W16`: 0 / 1
- `KT_GPU_STREAM_ZEROCOPY`: 0 / 1
- `KT_GPU_STREAM_GRAPH`: 1 (on in the code) / 1
- `KT_GPU_STREAM_EARLY_INIT`: 0 / 1
- `KT_GPU_STREAM_ROW_BLOCK`: 0 / 4096 (lib/kt-lend-auto.sh)
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

The KV pool's free pages during prefill. `KT_PREFILL_KV_LEND=1` lends the pool's free pages to the prefill (pages in use are never lent; the pool never shrinks). `KT_PREFILL_PLAN=1` plans each request's chunk size from the memory free at that moment, so a long prompt takes fewer, larger chunks (scheduler and schedule_batch hooks). `KT_PREFILL_SWA_LEND=1` sizes the sliding-window pool for the largest planned chunk and releases its free pages outside prefill. The file `kt_kv_hot_lend.py` ships here because the plan imports it; it does nothing until `hot-lend` wires it in.

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

`KT_KV_HOT_DYNAMIC_SLOTS=1`: the slots follow what is routed. Every routed expert is counted on the GPU; every 8 decode steps, outside the CUDA graph, each layer swaps up to 16 of its slots' experts for ones routed more often (counts halved every 256 steps). An expert that loses leaves the slot table at once and runs on the CPU again; the new one is copied in on a side stream straight from kt-kernel's expert memory (the zero-copy mapping `stream-prefill` registers, no host copy) and enters the table on the next round. At the first lend the slots check that experts copied this way match the placement's, byte for byte, and stay with the placement's experts if not. With `KT_KV_HOT_DYNAMIC_RESIDENT=1` the resident experts swap too, from one pool with the slots: kt-kernel then loads every expert, so one that leaves a resident row runs on the CPU, and the route tells kt-kernel which routes the GPU takes. In a prefill window the resident rows hold the placement's first experts again (the prefill lend reads every layer from its pinned copy), and the next lend fills them again. The route runs as one Triton kernel a layer. Without the dynamic slots the slots hold the placement's next-hottest experts, and the placement in `placements/` was recorded on Japanese prose: it catches 3-6% of the routes of code, JSON or math.

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
- The hot lend lends only device pages. When none are free, it runs on the driver's idle memory alone.

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

Prerequisites: `stream-prefill`.

Files (under `python/sglang/`):

- `srt/layers/moe/kt_stream_prefill.py`
- `srt/layers/moe/kt_gguf_moe_mmq.py` (new)
- `srt/layers/moe/kt_gguf_moe_mmq/` (new: the binding and the vendored llama.cpp 23b0202 MMQ kernels, MIT)
