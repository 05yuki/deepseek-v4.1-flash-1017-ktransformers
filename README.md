# DeepSeek-V4.1-Flash-1017 on SGLang + KTransformers (CPU/GPU hybrid inference)

Run [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
on one machine with a lot of RAM and ordinary GPUs. The 384 routed experts per layer
run on the CPU in MXFP4 through [KTransformers](https://github.com/kvcache-ai/ktransformers)'
kt-kernel. Attention, the hottest experts and the prefill run on the GPUs through
[SGLang](https://github.com/sgl-project/sglang). The engram tables (about 200 GB) are
read straight from the checkpoint on disk, with no extracted copy. The context is up
to 1M tokens.

This repository is everything needed: patches against two public upstream commits, a
setup script that builds them, a launcher that reads the machine it runs on, and tests.
It is a snapshot against those commits as of October 2026 (SGLang main at `ed0b8f43`,
KTransformers `95009ea`, torch 2.14.1 with CUDA 13 wheels, FlashInfer 0.7.2, CUDA toolkit
13.1); later upstream versions are not tracked.

```
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 --local-dir /models/dsv41-1017
MODEL=/models/dsv41-1017 ./setup.sh
MODEL=/models/dsv41-1017 ./serve.sh
```

`MODEL=/models/dsv41-1017 tools/check.sh` starts the server once, sends one short request,
checks that an answer comes back, and stops the server: the one command that says a setup works.

The server speaks SGLang's OpenAI-compatible API on `127.0.0.1:8080`, one request at a
time. To serve other machines, set `SGLANG_HOST=0.0.0.0` and `SGLANG_API_KEY=<key>`
(clients then send `Authorization: Bearer <key>`). The same setup also serves the earlier
DeepSeek-V4.1-Flash release: point `MODEL` at it.

## What it needs

RAM and VRAM play different parts here. RAM holds every routed expert and has to be
large enough: nothing in this setup makes the experts smaller, and with too little RAM
the server does not start. VRAM is different. Only the dense layers and the KV pool
have to fit on the GPUs; the rest of the GPU memory is working room, measured at
launch, and each use of it shrinks when there is less:

- the prefill chunk is calibrated on the first launch to the memory that is left, so a
  smaller card runs smaller chunks. Every chunk streams the experts to the GPU again,
  so the prefill takes longer;
- during a long prefill, the memory of the GPU-resident weights and the KV pool's free
  pages is lent to it, and given back before decode;
- in decode, extra hot experts go into the KV pool's free pages and the driver's idle
  memory, as many as fit (`KT_KV_HOT_SPARE_FLOOR_MB` keeps a margin free); with less
  room fewer are kept, and decode is slower;
- the resident experts per layer (`KT_GPU_EXPERTS`, 5) can go down to 0.

The KV pool itself is never made smaller to gain speed: the lends take only its free
pages. On these 16 GB cards, with no resident experts, the weights take 7.9 GB a card
and 4.6 GB is still free after the 1M-token KV pool. A shorter context
(`SGLANG_CONTEXT_LENGTH`) makes the pool smaller. Smaller cards were not tested.

- **RAM: a 512 GB class machine.** The server uses about 420 GiB, most of it the MXFP4
  CPU experts on 2 MB pages. 384 GB does not fit.

  The launcher makes one expert pool per NUMA node and uses every physical core.
  kt-kernel is built for the CPU it is built on (`CPUINFER_CPU_INSTRUCT=NATIVE`). Its
  MXFP4 kernels need AVX2.
- **VRAM: two 16 GB cards, nearly full, with the defaults.** Per card (tensor
  parallel 2): 8.3 GB of weights (half of attention and the other dense layers, plus 5
  resident experts a layer), 1.9 GB for the 1M-token KV pool, and the rest for prefill
  work and extra hot experts in decode.

  One GPU holds all of the dense part and the whole KV pool itself, about 24 GB at a 1M
  context by estimate. `CUDA_VISIBLE_DEVICES` picks the GPUs, one tensor-parallel rank
  each.
- **Disk: about 500 GB, NVMe.**
  - The checkpoint is 476 GiB (510 GB).
  - `.venv` is 8.6 GB and the upstream clones about 1 GB.
  - The load runs at the drive's speed, and the engram rows are read from it during
    inference, so a 1 TB NVMe drive is the practical choice.
- **Software.** Linux, bash 4.4 or newer, the CUDA toolkit (built and run with 13.1), git,
  gcc/g++, the libnuma and hwloc headers (`libnuma-dev libhwloc-dev`), and `uv` or Python 3.12.
  The Python packages are pinned in `requirements.txt` (torch 2.14.1 + cu130).
- **Host settings.** Set these as root, persistently:
  - `echo advise > /sys/kernel/mm/transparent_hugepage/shmem_enabled`. The expert
    arenas want 2 MB pages; on 4 KB pages, pinning them takes most of an hour.
  - `vm.min_free_kbytes = 8388608`. Without it, the CUDA JIT can hit a node-local OOM
    while the experts fill the nodes.
  - Swap off.

## What setup.sh does

1. Clones sgl-project/sglang at `ed0b8f43` on its main branch and applies the SGLang patches
   up to `TIER` (default 2), each tier on top of the ones below it:

   | `TIER` | patch | what it adds |
   |---|---|---|
   | 0 | `sglang-tier0/` (eight patches, one concern each) | what 1017 needs to run correctly on the KTransformers hybrid MoE: the FP8 step below, the SwiGLU clamp passed to the CPU experts, the expert mask kt-kernel takes, `-1` ids kept out of the Marlin MoE, zero GPU experts, a configurable load barrier, the tool namespaces of the current reference's chat encoding, and the V4.1 indexer on its torch path on SM12x (RTX 50, RTX PRO Blackwell, DGX Spark), which has no DeepGEMM build |
   | 1 | `sglang-tier1/` (one patch per feature, [docs/tier1.md](docs/tier1.md); `SGLANG_FEATURES` picks some) | streamed prefill, the lends, hot experts and placement, the KV pool on remappable memory, the GPU MoE and FP8 kernels (the streamed prefill needs SM89 or newer) |
   | 2 | `sglang-tier2-no-p2p.patch` | all-reduce through pinned host memory, for two GPUs without peer access |

   Below tier 1 a long prompt's experts run on the CPU (about 40 tok/s of prefill here, decode
   about 11 tok/s); the speed in Results is tier 2's.

2. Clones kvcache-ai/ktransformers at `95009ea` with its llama.cpp and pybind11
   submodules. From tier 1 it also applies `patches/kt-kernel-tree.patch` (the CPU kernels'
   speed, shared expert memory, the stream hand-over) and `patches/kt-kernel-llama-cpp.patch`;
   below tier 1 kt-kernel is built as pinned.
3. Creates `.venv` from `requirements.txt` and installs the patched SGLang.
4. Builds kt-kernel for this CPU and the visible GPUs' compute capability; at tier 1
   with parallel expert loading (`patches/kt-kernel-load-prefetch.py`).
5. Builds `engram-adapter/librow_store.so` and `build/libkv_vmm.so`.
6. With `MODEL` set, writes the engram manifest from the checkpoint's headers
   (`tools/make-engram-manifest.py`). It then runs the 1017 kernel's test on this GPU.

Re-running skips the finished steps; a different `TIER` re-applies the patches and
rebuilds kt-kernel. `serve.sh` reads the tier and leaves out what needs tier 1. Each patch is checked against its pinned tree before it
is applied; one that does not fit stops the setup with nothing applied. If the libnuma or hwloc headers live outside the
system paths, pass them in through `CPLUS_INCLUDE_PATH` and `LIBRARY_PATH`. CMake
flags go in `CMAKE_ARGS`.

## What serve.sh does

At start it reads the GPUs, their compute capability, the NUMA nodes and the physical
cores, and sets tensor parallelism, the CPU pools and the CUDA arch from them. The
defaults are listed in the script. Each one can be overridden by the environment
variable named next to it. Among them:

- the tier-1 features that were applied, each with its switch and default in
  [docs/tier1.md](docs/tier1.md): the prefill experts streamed to the GPU, the hot lend
  (more hot experts in decode, in memory nothing else is using), `wo_a` in FP8, the vision
  tower in host memory and the rest
- a prefill chunk calibrated once, on the first launch of a setup (the server starts
  one extra time for this)
- FlashInfer autotune off, so outputs reproduce across launches
- V4.1's tool-call and reasoning parsers (below)

Logs go to `logs/<model>/<timestamp>/`. `serve.sh --background` returns once the
server process is started.

The first launch in a new environment compiles SGLang's and FlashInfer's kernels. Here
that took about 15 minutes, before the one-time chunk calibration. Later launches
reuse the cache (`.cache/` in the checkout) and are ready in about 3 minutes, with
the checkpoint on NVMe and loaded once before.

What each patch changes: [docs/patches.md](docs/patches.md) (tiers 0 and 2, kt-kernel) and
[docs/tier1.md](docs/tier1.md) (tier 1, one section a feature, with its switch and measured effect).
What made it this fast, measured change by change, and what was tried and dropped:
[docs/optimizations.md](docs/optimizations.md).

## Tool calls and thinking

`serve.sh` starts the server with V4.1's tool-call parser and reasoning parser
(`--tool-call-parser deepseekv41 --reasoning-parser deepseek-v41`), so the OpenAI API returns
a tool call in `message.tool_calls` and the model's thinking in `reasoning_content`, streamed
or not. `SGLANG_TOOL_CALL_PARSER=` or `SGLANG_REASONING_PARSER=` (empty) leaves one out.

- Thinking is off unless a request turns it on:
  `"chat_template_kwargs": {"thinking": true}`.
- A tool may carry a `namespace` (tier 0): the prompt shows `namespace::name`, and the call
  comes back with the bare name in `function.name` and the namespace in `namespace`.
- `tools/tool-call-check.py [URL]` checks a running server: a call, the same call streamed,
  the answer after the tool's result, a namespaced tool, and a call with thinking on.

## The 1017 change: o in FP8 before `wo_a`

1017's reference inference code quantizes the attention output `o` to FP8 and back
before the bf16 `wo_a` projection:

```python
act_quant(o, fp8_block_size, scale_fmt, scale_dtype, True)  # blocks of 32, power-of-two scales
```

The earlier release did not have this step; 1017's weights were made with it.

- `patches/dsv4_o_fp8_qdq.py`: an in-place Triton kernel, bit-exact against the
  checkpoint's own tilelang `act_quant` (`tools/dsv4-o-fp8-qdq-test.py`). GPUs older
  than sm89 get an fp32 emulation of the e4m3 rounding, also bit-exact
  (`tools/e4m3-emulation-test.py`).
- Tier 0 (`patches/sglang-tier0/01-o-fp8-qdq.patch`) includes it, behind `SGLANG_DSV4_O_FP8_QDQ_BLOCK`.
  `tools/o-fp8-qdq-block.sh` (sourced by `serve.sh`) reads the block size from the
  checkpoint's own `inference/model.py`. The step therefore turns on for 1017 (block
  32) and stays off for the earlier release.
- `patches/sglang-dsv4-o-fp8-qdq.py` installs the same step in another SGLang tree on
  its own. It applies to SGLang main at `ed0b8f43` unmodified.

It works the same at any tensor-parallel size. Its 32-value blocks lie inside a head
(512 values), so splitting heads across GPUs never cuts a block.

Everything else 1017 changes is in [docs/1017-changes.md](docs/1017-changes.md).
None of it needs a server change:
- The tokenizer and `config.json` are identical.
- The reference indexer's `owns_k` fix does not apply to SGLang's port.
- The encoding changes only affect tool calls.
- The card's sampling is `temperature` 1.0, `top_p` 1.0. These are already SGLang's
  defaults for a request that sets neither, since the checkpoint has no
  `generation_config.json`.

## Results

2 × AMD EPYC 7452 (Zen 2, AVX2), 512 GB DDR4, 2 × RTX 5070 Ti 16 GB, the checkpoint on
NVMe; built from this repository and run with the `serve.sh` defaults (1M-token context,
one request at a time), measured with `tools/bench.py`.

Decode, 1,024 tokens a reply (`--other-text` for the rows below the first):

| text | tok/s |
|---|---|
| Japanese prose (median of four prompts) | 43.8 |
| Chinese prose | 38.5 |
| English prose | 35.4 |
| math | 34.7 |
| JSON | 32.8 |
| English technical prose | 32.4 |
| Python code | 32.0 |

Five experts a layer stay on the GPU, and up to 16 more sit in the KV pool's free pages; the
hot lend swaps those 16 for the ones the text being written routes to most, so no routing
has to be recorded for a new kind of text. Code, JSON, math and English spread their routes over more experts than Japanese prose (a layer's 21 most
used take about half of their routes, against 70-75% in Japanese prose), so the CPU computes
more of them: a GPU with more memory can hold more of them with `KT_KV_HOT_EXTRA`
([docs/tier1.md](docs/tier1.md#16-hot-lend)).

Prefill, one prompt of random token ids:

| prompt (tokens) | tok/s | time |
|---|---|---|
| 35,000 | 766 | 46 s |
| 131,072 | 776 | 2.8 min |
| 524,288 | 659 | 13 min |
| 1,000,000 | 558 | 30 min |

Longer prompts run in smaller chunks: the prompt's own KV and the attention's working
memory grow with its length, leaving less GPU memory for a chunk, and every chunk streams
the CPU experts to the GPU once. The prompts are random token ids, which spread over more
experts than real text, so these figures are lower than with text.

## Fixes from dsv41-flash-dual-epyc

Two fixes by f-wunderlich, from the earlier release's repo: `--kt-num-gpu-experts 0` starts
with Marlin (#2, tier 0 for the MoE call and tier 1's `loading` for the loader). The SM120
padding-page fix (#3) is no longer carried: SGLang main's sparse attention with FlashInfer
0.7.2 runs a 48K-token prefill without it, with the same NLL.

## Not done

- The resident-expert placement (`placements/`) was recorded from the earlier
  release's routing on Japanese prose. It loads and runs with 1017, but it is not
  1017's hottest set, and it fits no other workload better. To record your own, use
  `KT_ROUTING_DUMP` and SGLang's `--init-expert-location`.
- A forced `tool_choice` (named or required) does not carry a tool's namespace: the
  constrained decoding asks for the bare name.
- Tested on 2 × RTX 5070 Ti (SM120) with Zen 2 CPUs only.
- CPUs with AVX-512 should work: the experts run on the AVX2 kernels, which those CPUs
  also have.
- One GPU, or more than two: `serve.sh` sets tensor parallelism from the GPUs it sees;
  not tested. One GPU needs about 24 GB at a 1M context.
- Other NVIDIA GPUs: the DeepSeek-V4 attention in this SGLang branch is written for SM90
  and newer, so RTX 30/40 and A100 are not expected to run. With
  `SGLANG_MXFP4_MOE_FP32=0`, the MXFP4 Marlin path also needs SM90 or SM120.
- AMD GPUs: not supported (kv-vmm and kt-kernel use CUDA).

## License

Apache-2.0 (see `LICENSE`), except `engram-adapter/`, which is derived from 0xSero's
engram adapter under the MIT License (see the headers of its files). The model weights
are under DeepSeek's license (see the model card).
