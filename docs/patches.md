# What the patches do

`setup.sh` applies the patches up to `TIER` (default 3), each tier on top of the ones below it:

| tier | file | applies to |
|---|---|---|
| 0 | `patches/sglang-tier0-run.patch` | sgl-project/sglang `1aa0e962` (formerly the `dsv4.1` branch) |
| 1 | `patches/sglang-tier1-long-prompts.patch` | the same, after tier 0 |
| 2 | `patches/sglang-tier2-speed.patch` | the same, after tier 1 |
| 3 | `patches/sglang-tier3-no-p2p.patch` | the same, after tier 2 |
| 2 | `patches/kt-kernel-tree.patch` | kvcache-ai/ktransformers `95009ea` (below tier 2 it is built as pinned) |
| 2 | `patches/kt-kernel-llama-cpp.patch` | ktransformers' `third_party/llama.cpp` submodule |
| 2 | `patches/kt-kernel-load-prefetch.py` | the installed kt-kernel (parallel expert loading) |
| - | `patches/sglang-dsv4-o-fp8-qdq.py`, `patches/dsv4_o_fp8_qdq.py` | another SGLang tree, on their own (tier 0 already has them) |

- **Tier 0** is what DeepSeek-V4.1-Flash-1017 needs to run correctly on the KTransformers
  hybrid MoE, on any GPU, and nothing else. KTransformers `95009ea` needs no change for it: its AVX2
  MXFP4 experts, the SwiGLU clamp and the loader are there.
- **Tier 1** keeps the indexer's memory flat on long prompts.
- **Tier 2** is speed and memory on any GPU this SGLang runs DeepSeek-V4 on; the streamed
  prefill computes through FlashInfer's bf16 CUTLASS MoE (SM89 or newer). It needs the
  kt-kernel patch (shared expert memory). `serve.sh` turns its features on from tier 2.
- **Tier 3** is what depends on the machine: two GPUs without peer access.

Each section below says its tier.

Most features have an environment variable. The defaults are the ones `serve.sh` sets. A
switch marked "on" is on in `serve.sh`.

## SGLang patches

### CPU/GPU hybrid MoE (KTransformers wrapper)

`layers/moe/kt_ep_wrapper.py`. Tier 0: the clamp, the `-1` ids, the expert mask and zero GPU
experts. Tier 2: the rest.

- Runs physical experts `0..N-1` of each layer on the GPU (`--kt-num-gpu-experts`) and the
  rest on the CPU through kt-kernel, in parallel, and adds the two results.
- Maps the router's logical expert ids to physical ones, so a hot-expert placement
  (`--init-expert-location`, `placements/`) actually puts the hot experts on the GPU.
- Sends the CPU's routes to GPU expert 0 at weight 0 instead of `-1`, which the Marlin MoE
  kernels need (`KT_GPU_MASK_ZERO`, set by `serve.sh` with Marlin).
- Passes kt-kernel the boolean mask of GPU experts its inference API takes, and runs with
  zero GPU experts (`--kt-num-gpu-experts 0`).
- Passes the checkpoint's SwiGLU clamp to the CPU experts (`KT_SWIGLU_CLAMP`, on).
- `KT_ROUTING_DUMP=<prefix>`: saves a per-layer histogram of the routed experts, to build
  a placement from your own workload.

### GPU MoE kernels

Tier 2.

`layers/moe/mxfp4_fp32_moe.py`, `layers/quantization/mxfp4_marlin_moe.py`, `layers/quantization/fp8.py`

- `SGLANG_MXFP4_MOE_FP32` (on): the GPU experts (resident and hot) stay in the
  checkpoint's MXFP4 layout and run through Triton kernels that round to bf16 once, at
  the weighted sum. Decode uses a per-route GEMV; larger batches compute only the routes
  with a weight, on the tensor cores. Outputs repeat exactly from launch to launch.
- With `SGLANG_MXFP4_MOE_FP32=0` the experts run through Marlin instead.
  `SGLANG_MARLIN_MOE_FP32_GLUE` (on) then keeps the steps around Marlin's two GEMMs in fp32
  (`layers/moe/fused_moe_triton/fused_marlin_moe.py`).

### Streamed prefill

Tier 2.

`layers/moe/kt_stream_prefill.py`, `layers/moe/moe_runner/flashinfer_cutlass.py`,
`layers/quantization/mxfp4_flashinfer_cutlass_moe.py`

- Above `KT_GPU_STREAM_PREFILL` tokens (512), the CPU experts are not computed on the
  CPU. Their weights are copied to the GPU in groups over PCIe and computed there, the
  copy of the next group overlapping the compute of the current one.
- `KT_GPU_STREAM_W16` (on) dequantizes each group to bf16 on the GPU and runs it through
  FlashInfer's bf16 CUTLASS MoE, which FlashInfer builds for SM89, SM90, SM100, SM103 and
  SM120. With `KT_GPU_STREAM_W16=0` the groups stay MXFP4 and run through the SM120
  MXFP8 x MXFP4 kernel instead (SM120 only).
  `KT_GPU_STREAM_ZEROCOPY` and `KT_GPU_STREAM_GRAPH` (on) read straight from kt-kernel's
  shared expert memory and launch each group as one CUDA graph.
- One persistent CUTLASS MoE workspace per GPU, allocated before the decode graphs are
  captured.

### Prefill lend: GPU memory for long prompts

Tier 2.

`layers/moe/kt_lend.py`, `layers/moe/kt_lend_probe.py`, `layers/moe/kt_pinned.py`,
`model_executor/model_runner.py`, `model_loader/loader.py`, `utils/torch_memory_saver_adapter.py`

- `KT_PREFILL_LEND` (on): during a streamed prefill, the GPU-resident weights give their
  memory to the prefill and are brought back layer by layer, at the same addresses,
  just before each layer runs. Decode sees them as before.
- The first launch of a setup measures how large a prefill chunk fits and keeps it as a
  hint (`lib/kt-lend-auto.sh`).

### KV pool on remappable memory

Tier 2.

`layers/moe/kt_kv_vmm.py`, `layers/moe/kt_kv_prefill_lend.py`, `layers/moe/kt_prefill_plan.py`,
`mem_cache/kv_cache_configurator.py`, `model_executor/pool_configurator.py`,
`managers/scheduler.py`, `managers/schedule_batch.py`; the library is `kv-vmm/kv_vmm.c`

- `KT_KV_VMM` (on): the KV pool sits on memory whose physical pages can be unmapped and
  mapped again while its addresses, and the CUDA graphs that read it, stay the same.
- `KT_PREFILL_KV_LEND` (on): during a prefill, the pool's free pages lend their memory to
  the prefill. The KV pool never shrinks: pages in use are never lent.
- `KT_PREFILL_PLAN` (on): each request's chunk size is planned from the memory free at
  that moment, so a long prompt takes fewer, larger chunks.
- `KT_PREFILL_SWA_LEND` (on): the sliding-window pool is sized for the largest planned
  chunk, and its free pages are released outside prefill.

### Hot lend: more GPU experts in decode

Tier 2.

`layers/moe/kt_kv_hot_lend.py`, `layers/moe/fused_moe_triton/layer.py`, `model_loader/weight_utils.py`

- `KT_KV_HOT_LEND` (on): outside prefill, the KV pool's free pages hold up to
  `KT_KV_HOT_EXTRA` (16) more experts a layer on the GPU, the next-hottest after the
  resident ones. They leave whenever the KV pool needs the pages or a prefill starts.
- `KT_KV_HOT_SPARE_FLOOR_MB` (1024): the slots also use the GPU memory the driver has
  idle in decode, above this floor.
- `KT_KV_HOT_SLOTS_MAX`: caps the slots a GPU (for measurements).

### DeepSeek-V4.1-Flash-1017 model changes

Tier 0: the FP8 step. Tier 2: the rest.

`models/deepseek_v4.py`, `models/deepseek_common/o_fp8_qdq.py`, `models/deepseek_common/wo_a_fp8_gemv.py`,
`models/deepseek_v4_dspark.py`

- 1017's FP8 quantize-dequantize of the attention output before `wo_a`, bit-exact against
  the reference code (`SGLANG_DSV4_O_FP8_QDQ_BLOCK`, set from the checkpoint by
  `tools/o-fp8-qdq-block.sh`). See the README.
- `SGLANG_DSV4_WO_A_FP8_NATIVE` (on): `wo_a` stays in the checkpoint's FP8 instead of a bf16
  copy. Decode reads the FP8 directly; prefill dequantizes into one shared buffer.
- `SGLANG_DSV4_VISION_ON_DEMAND` (on): the vision tower waits in pinned host memory and
  is on the GPU only while a batch's images are encoded. The prefill plan accounts for it
  in requests with images.
- `SGLANG_DSV41_ENGRAM_PREFETCH_ROWS` (on): in decode, each engram layer's row lookup starts
  with the token on a side stream instead of when its layer comes. The all-reduce, `wkv` and
  the gate stay in place, so the values are unchanged.
- The padded query buffer of a prefill is part of the prefill's scratch memory.

### SM12x sparse attention

Tier 0. `kernels/ops/attention/flash_mla_sm120.py`, which only GPUs of compute capability 12.x
run (RTX 50, RTX PRO Blackwell, DGX Spark); other GPUs are unaffected.

- The page-split scratch converts the extra KV source (128-token pages) to 64-token pages
  as well, in its own buffer; without it the first prefill stops with "Unsupported
  sparse-MLA prefill configuration".
- Index lanes past `topk_length` are masked (a FlashInfer 0.6.18 bug corrupts prompts over
  64 tokens otherwise); the reserved padding page is zeroed on every conversion
  (dsv41-flash-dual-epyc#3 by f-wunderlich).

### Indexer on long prompts

Tier 1. `layers/attention/deepseek_v4_backend.py`, `layers/attention/dsv4/indexer.py`

- The dense low-ratio indexer scores a long prefill's rows in slices, so its logits never
  hold the whole chunk at once (a long prefill otherwise runs out of memory).

### Dense layers and all-reduce

Tier 2: the GEMMs. Tier 3: the all-reduces.

`kernels/ops/quantization/fp8_kernel.py`, `layers/quantization/fp8_utils.py`,
`distributed/device_communicators/ll_allreduce.py`, `distributed/device_communicators/host_staged_allreduce.py`,
`distributed/parallel_state.py`

- Block-FP8 GEMMs with up to four rows (decode) run through a GEMV kernel that steps K by
  several scale groups at a time (`kernels/ops/quantization/fp8_block_gemv.py`,
  `SGLANG_FP8_GEMV`, on). Same arithmetic up to the fp32 summation order.
- Otherwise, block-FP8 GEMMs with a few rows use a split-K tiling (`SGLANG_FP8_SKINNY`, on by
  default in the code).
- Two GPUs without peer access: small all-reduces (decode) through pinned host memory in
  NCCL's LL form (`SGLANG_LL_ALLREDUCE`), large ones (prefill) staged through pinned host
  memory with the copy engines (`SGLANG_HOST_STAGED_ALLREDUCE`). `serve.sh` turns both on
  for two GPUs.

### Chat encoding

Tier 0. `entrypoints/openai/encoding_dsv41.py`, `entrypoints/openai/protocol.py`,
`entrypoints/openai/serving_chat.py`, `function_call/deepseekv41_detector.py`,
`function_call/core_types.py`

The prompt format of the checkpoint's current `encoding/encoding.py`
(`tools/dsv41-encoding-check.py` compares the two byte for byte):

- Tool namespaces: a tool may carry `namespace`; the prompt shows `namespace::name`, and a
  parsed call returns the bare name in `function.name` and the namespace in an added
  `namespace` field of the tool call (absent when there is none).
- Without a tools declaration, reasoning is dropped only before the latest real user
  request; tool results do not start a new turn. Tool-result images keep prompt order.
- Reasoning effort `low` / `high` / `max` render as 50 / 75 / 100, the reference's budgets
  (the pinned commit had 25 / 50; sgl-project/sglang#39929). The default is `high`.

### Smaller fixes

Tier 0: the load barrier's timeout. Tier 2: the rest.

- `entrypoints/openai/serving_chat.py`, `configs/model_config.py`, `models/deepseek_v2.py`,
  `multimodal/...`: DeepSeek-V4 checkpoints with a vision tower are detected by the tower,
  keep their image in the chat encoding, and route image tokens with the vision bias.
- `entrypoints/warmup.py`: the warmup also runs short extends with sampling
  (`KT_WARMUP_EXTEND`), so the first real request does not pay kernel compiles.
- `model_loader/weight_utils.py`: the GPU side's loader skips reading the experts that only
  the CPU runs (`KT_LOAD_SKIP_CPU_EXPERTS`, on), and reads the hot lend's experts too.
- `model_executor/model_runner_components/load_model_utils.py`: the wait for the other
  rank's weight loading is set by `SGLANG_UNBALANCED_LOAD_TIMEOUT_S` (`serve.sh`: 1800 s).
- `layers/moe/kt_launch_timing.py` (`KT_LAUNCH_TIMING=1`): a timeline of the weight load,
  for diagnosis only.

## kt-kernel tree patch

Tier 2.

- **MXFP4 experts on AVX2** (`operators/avx2/mxfp4-moe.hpp`, `moe_base.hpp`): the CPU expert
  kernels for this model. Activations are reordered once per expert to match the weight
  decode; the SwiGLU clamp is applied (`KT_SWIGLU_CLAMP`).
- **Scale codes** (`KT_MXFP4_SCALE_CODES`, on): one byte per 32-value scale group instead
  of an fp32 scale, 15% fewer bytes a weight, bit-identical results.
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
- `tools/mxfp4-fp32-moe-test.py`: the GPU MoE kernels against an fp64 reference and Marlin,
  and their outputs repeating bit for bit.
- `tools/bench.py`: the README's decode and prefill numbers, against a running server.
