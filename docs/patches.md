# What the patches do

`setup.sh` applies the patches up to `TIER` (default 2), each tier on top of the ones below it:

| tier | file | applies to |
|---|---|---|
| 0 | `patches/sglang-tier0/NN-*.patch` | sgl-project/sglang main at `ed0b8f43`, one concern each |
| 1 | `patches/sglang-tier1/NN-*.patch` | the same, after tier 0: one patch per feature ([tier1.md](tier1.md)) |
| 2 | `patches/sglang-tier2-no-p2p.patch` | the same, after tier 1 |
| 1 | `patches/kt-kernel-tree.patch` | kvcache-ai/ktransformers `95009ea` (below tier 1 it is built as pinned) |
| 1 | `patches/kt-kernel-llama-cpp.patch` | ktransformers' `third_party/llama.cpp` submodule |
| 1 | `patches/kt-kernel-load-prefetch.py` | the installed kt-kernel (parallel expert loading) |
| - | `patches/sglang-dsv4-o-fp8-qdq.py`, `patches/dsv4_o_fp8_qdq.py` | another SGLang tree, on their own (tier 0 already has them) |

- **Tier 0** is what DeepSeek-V4.1-Flash-1017 needs to run correctly on the KTransformers
  hybrid MoE, on any GPU, and nothing else. KTransformers `95009ea` needs no change for it: its AVX2
  MXFP4 experts, the SwiGLU clamp and the loader are there.
- **Tier 1** is speed and memory on any GPU this SGLang runs DeepSeek-V4 on; the streamed
  prefill computes through FlashInfer's bf16 CUTLASS MoE (SM89 or newer). It needs the
  kt-kernel patch (shared expert memory). `serve.sh` turns its features on from tier 1.
- **Tier 2** is what depends on the machine: two GPUs without peer access.

The SGLang patches are the commits of branch `dsv41-1017-main` of
[05yuki/sglang-upstream](https://github.com/05yuki/sglang-upstream/tree/dsv41-1017-main) on `ed0b8f43`, one
concern a commit, in `git format-patch` form: each file carries its commit message (what the change
is for, its switch, how it was checked). `git log` on the branch reads as the list below.

A switch marked "on" is on in `serve.sh`.

## SGLang tier 0

What 1017 needs to run correctly. Each patch is one concern; its commit message says how
it was checked. Four are pull requests to SGLang, one concern each; when one is merged, its
patch goes and the pin moves past it.

| patch | upstream |
|---|---|
| `01-o-fp8-qdq` | [#45185](https://github.com/sgl-project/sglang/pull/45185) |
| `02-gpu-experts-mask` | [#45183](https://github.com/sgl-project/sglang/pull/45183) |
| `03-swiglu-clamp` | not sent |
| `04-mask-zero` | not sent (a workaround for the Marlin MoE path) |
| `05-zero-gpu-experts` | [#45188](https://github.com/sgl-project/sglang/pull/45188) |
| `06-load-timeout` | [#45184](https://github.com/sgl-project/sglang/pull/45184) |
| `07-chat-encoding` | not sent |
| `08-sm12x-indexer` | not sent |

### KTransformers wrapper

`layers/moe/kt_ep_wrapper.py`. SGLang's wrapper runs physical experts `0..N-1` of each layer on
the GPU (`--kt-num-gpu-experts`) and the rest on the CPU through kt-kernel, in parallel, and
adds the two results. Tier 0:

- Passes the checkpoint's SwiGLU clamp to the CPU experts (`KT_SWIGLU_CLAMP`, on).
- Passes kt-kernel the boolean mask of GPU experts its inference API takes.
- Sends the CPU's routes to GPU expert 0 at weight 0 instead of `-1`, which the Marlin MoE
  kernels need (`KT_GPU_MASK_ZERO`, set by `serve.sh` with Marlin).
- Runs with zero GPU experts (`--kt-num-gpu-experts 0`).

### 1017's FP8 step

`models/deepseek_v4.py`, `models/deepseek_common/o_fp8_qdq.py`, `models/deepseek_v4_dspark.py`

The attention output in FP8 before `wo_a`, bit-exact against the reference code
(`SGLANG_DSV4_O_FP8_QDQ_BLOCK`, set from the checkpoint by `tools/o-fp8-qdq-block.sh`). See
the README.

### V4.1 indexer on SM12x

`layers/attention/dsv4/v41_indexer/__init__.py`; only GPUs of compute capability 12.x (RTX
50, RTX PRO Blackwell, DGX Spark) are affected.

- `is_sm100_or_newer()` leaves out compute capability 12: DeepGEMM has no SM12x build, and the
  candidate indexer otherwise stops at startup ("the candidate indexer needs DeepGEMM's paged
  sparse MQA logits"). SM12x takes Hopper's torch indexer.
- The sparse attention fixes this tier carried on the earlier base (the extra KV source
  page-split, lanes past `topk_length` masked, the padding page zeroed) are not needed with
  FlashInfer 0.7.2: a 48K-token prefill runs without them, with the same NLL.

### Chat encoding

`entrypoints/openai/encoding_dsv41.py`, `entrypoints/openai/protocol.py`,
`entrypoints/openai/serving_chat.py`, `function_call/deepseekv41_detector.py`,
`function_call/core_types.py`

The prompt format of the checkpoint's current `encoding/encoding.py`
(`tools/dsv41-encoding-check.py` compares the two byte for byte):

- Tool namespaces: a tool may carry `namespace`; the prompt shows `namespace::name`, and a
  parsed call returns the bare name in `function.name` and the namespace in an added
  `namespace` field of the tool call (absent when there is none).

The reference's other later changes are in SGLang main already: reasoning dropped only before
the latest real user request, tool-result images in prompt order (sgl-project/sglang#44751),
and the reasoning-effort budgets 50 / 75 / 100 (#39929).

### Load barrier

`model_executor/model_runner_components/load_model_utils.py`: the wait for the other rank's
weight loading is set by `SGLANG_UNBALANCED_LOAD_TIMEOUT_S` (`serve.sh`: 1800 s).

## SGLang tier 1

One patch per feature. What each does, its switches and their defaults, its numerics, and
what it was measured to change are in [tier1.md](tier1.md), one section a patch. By what they
are for:

| for | patches |
|---|---|
| decode | [`fp8-decode`](tier1.md#01-fp8-decode), [`engram-row-prefetch`](tier1.md#04-engram-row-prefetch), [`wo-a-fp8`](tier1.md#05-wo-a-fp8), [`placement`](tier1.md#11-placement), [`hot-lend`](tier1.md#16-hot-lend) |
| long prompts | [`stream-prefill`](tier1.md#14-stream-prefill), [`prefill-lend`](tier1.md#13-prefill-lend), [`kv-vmm`](tier1.md#12-kv-vmm), [`kv-lend-plan`](tier1.md#15-kv-lend-plan) |
| GPU memory | [`wo-a-fp8`](tier1.md#05-wo-a-fp8), [`vision-on-demand`](tier1.md#06-vision-on-demand) |
| numerics | [`gpu-moe-fp32`](tier1.md#02-gpu-moe-fp32) |
| startup | [`loading`](tier1.md#08-loading), [`launch-timing`](tier1.md#09-launch-timing), [`pool-swa-guard`](tier1.md#10-pool-swa-guard) |
| fixes | [`cutlass-kt-expert-count`](tier1.md#03-cutlass-kt-expert-count), [`thinking-default-parser`](tier1.md#07-thinking-default-parser) |

## SGLang tier 2

`distributed/device_communicators/ll_allreduce.py`,
`distributed/device_communicators/host_staged_allreduce.py`, `distributed/parallel_state.py`

Upstream: [#39605](https://github.com/sgl-project/sglang/pull/39605) (host-staged all-reduce).

Two GPUs without peer access: small all-reduces (decode) through pinned host memory in NCCL's
LL form (`SGLANG_LL_ALLREDUCE`), large ones (prefill) staged through pinned host memory with
the copy engines (`SGLANG_HOST_STAGED_ALLREDUCE`). `serve.sh` turns both on for two GPUs.

## kt-kernel tree patch

Tier 1.

- **MXFP4 experts on AVX2** (`operators/avx2/mxfp4-moe.hpp`, `moe_base.hpp`): the CPU expert
  kernels for this model. Activations are reordered once per expert to match the weight
  decode; the SwiGLU clamp is applied (`KT_SWIGLU_CLAMP`).
- **Scale codes** (`KT_MXFP4_SCALE_CODES`, on): one byte per 32-value scale group instead
  of an fp32 scale, 15% fewer bytes a weight, bit-identical results.
- **Decode reads** (`KT_MXFP4_DEC_VCONV`, `KT_MXFP4_DEC_PFS=1024`, on): each task converts the
  activation row to the kernel's order with PSHUFB instead of a scalar loop, and the weights
  are prefetched 1 KB ahead, a line every 4 groups. Bit-identical results.
- **Shared expert memory** (`KT_EXPERT_SHM`, on): the experts live in shared memory on 2 MB
  pages, which the GPU ranks map to stream from (the streamed prefill).
- **Placement on the CPU side**: each NUMA part loads the experts the placement assigns to
  it, not the staging order.
- **Loader** (`python/utils/loader.py`, `patches/kt-kernel-load-prefetch.py`): safetensors
  experts read in parallel, a layer ahead, through the page cache.
- **Task queue and worker pool**: the GPU stream hands work to the CPU through stream memory
  operations (`KT_STREAM_MEMOPS`, on), and waiting threads spin briefly before sleeping
  (`KT_TASKQUEUE_SPIN_US`).
- **Diagnostics**: `KT_DECODE_PHASES=<calls>` and `KT_TASKQUEUE_TIMING=<syncs>` print where
  the CPU's time goes.
- Code for other models also comes with the tree (the llama.cpp-format GGUF experts and
  `kt-kernel-llama-cpp.patch`, EXL3 experts, Gemma 4 BF16 experts). This repository does
  not use it, but the build needs it.

## Values tuned on the test machine

These were measured on 2 × RTX 5070 Ti (SM120) with Zen 2 CPUs. Other machines run with
them unchanged, which may not be their best:

- the GPU MoE kernels' tiles (`GEMV_BN`, `GEMV_BK`, `GEMM_SMALL`, `GEMM_LARGE` in
  `mxfp4_fp32_moe.py`) and the decode/tensor-core split (`GEMV_MAX_TOKENS`, 8 tokens);
- `KT_GPU_STREAM_PREFILL` (512 tokens) and `KT_GPU_STREAM_GROUP` (4 experts a group);
- the hot lend's `KT_KV_HOT_EXTRA` (16 a layer) and `KT_KV_HOT_SPARE_FLOOR_MB` (1024);
- `SGLANG_MEM_FRACTION` (0.81) and the prefill chunk, which the first launch calibrates.

## Tests

- `tools/dsv4-o-fp8-qdq-test.py`: 1017's FP8 step against the checkpoint's own `act_quant`.
- `tools/e4m3-emulation-test.py`: the fp32 e4m3 rounding for GPUs before sm89.
- `tools/dsv41-encoding-check.py`: SGLang's chat encoding and tool parsing against the
  checkpoint's `encoding/encoding.py` and its tests, byte for byte.
- `tools/kt-mxfp4-decode-bw.py`: the CPU experts' decode call on one socket (time, read rate,
  output hash), to compare kt-kernel switches and against the socket's read bandwidth.
- `tools/mxfp4-fp32-moe-test.py`: the GPU MoE kernels against an fp64 reference and Marlin,
  and their outputs repeating bit for bit.
- `tools/bench.py`: the README's decode and prefill numbers, against a running server.
