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

```
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 --local-dir /models/dsv41-1017
MODEL=/models/dsv41-1017 ./setup.sh
MODEL=/models/dsv41-1017 ./serve.sh
```

The server speaks SGLang's OpenAI-compatible API on port 8080. The same setup also
serves the earlier DeepSeek-V4.1-Flash release: point `MODEL` at it.

## What it needs

- **RAM: a 512 GB class machine.** The server used 418 GiB, measured while running
  1017:
  - 313 GiB holds the MXFP4 CPU experts, on 2 MB pages.
  - About 40 GiB is the rest of the shared memory: arena pages that ended up on 4 KB,
    the prefill lend's mirror and the all-reduce buffers.
  - Add the 8 GiB engram cache and the OS. 384 GB does not fit. 448 GB leaves no room
    for the page cache the load passes through.

  The launcher makes one expert pool per NUMA node and uses every physical core.
  kt-kernel is built for the CPU it is built on (`CPUINFER_CPU_INSTRUCT=NATIVE`). Its
  MXFP4 kernels need AVX2.
- **VRAM: two 16 GB cards, nearly full, with the defaults.** Per card (tensor
  parallel 2):
  - 9.9 GB of weights: half of attention and the other dense layers, plus 5 resident
    experts a layer.
  - 1.9 GB for the 1M-token KV pool.
  - 0.3 GB of CUDA graphs.
  - 2.7 GB left for prefill work: the streamed experts' buffers and a 0.5 GB CUTLASS
    workspace.

  One GPU holds all of the dense part and the whole KV pool itself. By estimate, that
  needs about 24 GB at a 1M context. A single 16 GB card might fit with
  `KT_GPU_EXPERTS=0` and `SGLANG_CONTEXT_LENGTH` around 262144. Neither is measured.
  `CUDA_VISIBLE_DEVICES` picks the GPUs, one tensor-parallel rank each.
- **Disk: about 500 GB, NVMe.**
  - The checkpoint is 476 GiB (510 GB).
  - `.venv` is 8.6 GB and the upstream clones about 1 GB.
  - The load runs at the drive's speed, and the engram rows are read from it during
    inference, so a 1 TB NVMe drive is the practical choice.
- **Software.** Linux, the CUDA toolkit (built and run with 13.1), git, gcc/g++, the
  libnuma and hwloc headers (`libnuma-dev libhwloc-dev`), and `uv` or Python 3.12.
  The Python packages are pinned in `requirements.txt` (torch 2.13 + cu130).
- **Host settings.** Set these as root, persistently:
  - `echo advise > /sys/kernel/mm/transparent_hugepage/shmem_enabled`. The expert
    arenas want 2 MB pages; on 4 KB pages, pinning them takes most of an hour.
  - `vm.min_free_kbytes = 8388608`. Without it, the CUDA JIT can hit a node-local OOM
    while the experts fill the nodes.
  - Swap off.

## What setup.sh does

1. Clones sgl-project/sglang at `1aa0e962` (the `dsv4.1` branch) and applies
   `patches/sglang-dsv41-tree.patch`. The patch adds:
   - the KTransformers expert wrapper's placement, streaming and lend paths
   - host-memory all-reduce
   - the KV pool on remappable memory (kv-vmm)
   - the indexer and load fixes
   - the 1017 step below
2. Clones kvcache-ai/ktransformers at `95009ea` with its llama.cpp and pybind11
   submodules, and applies `patches/kt-kernel-tree.patch` and
   `patches/kt-kernel-llama-cpp.patch`.
3. Creates `.venv` from `requirements.txt` and installs the patched SGLang.
4. Builds kt-kernel for this CPU and the visible GPUs' compute capability, with
   parallel expert loading (`patches/kt-kernel-load-prefetch.py`).
5. Builds `engram-adapter/librow_store.so` and `build/libkv_vmm.so`.
6. With `MODEL` set, writes the engram manifest from the checkpoint's headers
   (`tools/make-engram-manifest.py`). It then runs the 1017 kernel's test on this GPU.

Re-running skips the finished steps. If the libnuma or hwloc headers live outside the
system paths, pass them in through `CPLUS_INCLUDE_PATH` and `LIBRARY_PATH`. CMake
flags go in `CMAKE_ARGS`.

## What serve.sh does

At start it reads the GPUs, their compute capability, the NUMA nodes and the physical
cores, and sets tensor parallelism, the CPU pools and the CUDA arch from them. The
defaults are listed in the script. Each one can be overridden by the environment
variable named next to it. Among them:

- the prefill experts streamed to the GPU, in bf16
- the resident experts through Marlin
- a "hot lend": 16 more hot experts a layer kept in the KV pool's free pages outside
  prefill, without taking any of the pool
- a prefill chunk calibrated once, on the first launch of a setup (the server starts
  one extra time for this)
- FlashInfer autotune off, so outputs reproduce across launches

Logs go to `logs/<model>/<timestamp>/`. `serve.sh --background` returns once the
server process is started.

The first launch in a new environment compiles SGLang's and FlashInfer's kernels. Here
that took about 15 minutes, before the one-time chunk calibration. Later launches
reuse the cache (`~/.cache/sglang-dsv41-1017`) and are ready in about 4 minutes, with
the checkpoint on NVMe and loaded once before.

## The 1017 change: o in FP8 before `wo_a`

1017's reference inference code quantizes the attention output `o` to FP8 and back
before the bf16 `wo_a` projection:

```python
act_quant(o, fp8_block_size, scale_fmt, scale_dtype, True)  # blocks of 32, power-of-two scales
```

The earlier release did not have this step. The weights are new (all 48 shards
changed) and were made with it. The step moves `o` by 1.8-2.6% (relative L2) in
every layer.

- `patches/dsv4_o_fp8_qdq.py`: an in-place Triton kernel. It is bit-exact against the
  checkpoint's own tilelang `act_quant`: 0 differing bf16 values at 1, 3, 37 and 4,096
  tokens (`tools/dsv4-o-fp8-qdq-test.py`).
  - On sm89 and newer, the e4m3 rounding uses the hardware conversion.
  - On Ampere and older, Triton has no e4m3 type, so the kernel rounds in fp32
    arithmetic. This emulation matches torch's `float8_e4m3fn` cast bit for bit,
    signed zeros included, on 118,343,666 values (`tools/e4m3-emulation-test.py`).
- The tree patch includes it, behind `SGLANG_DSV4_O_FP8_QDQ_BLOCK`.
  `tools/o-fp8-qdq-block.sh` (sourced by `serve.sh`) reads the block size from the
  checkpoint's own `inference/model.py`. The step therefore turns on for 1017 (block
  32) and stays off for the earlier release.
- `patches/sglang-dsv4-o-fp8-qdq.py` installs the same step in another SGLang tree on
  its own. It applies to `dsv4.1` at `1aa0e962` unmodified.

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

One machine, as an example: 2 × AMD EPYC 7452 (Zen 2, AVX2), 512 GB DDR4, 2 × RTX
5070 Ti 16 GB, the checkpoint on NVMe. It ran the `serve.sh` defaults: 5 resident
experts a layer, a 1M-token context, one request at a time. Each column is one
launch.

The last column was built from this repository alone: a new directory, a new venv,
`setup.sh` cloning both upstreams from GitHub, then `serve.sh`. The other columns ran
on the development tree these patches were taken from.

| | earlier release | 1017 | 1017 without the FP8 step | 1017, built from this repo |
|---|---|---|---|---|
| decode, Japanese prose, median tok/s | 30.01 | 30.62 | 30.76 | 31.26 |
| prefill, 35,000-token prompt, tok/s: first / second prompt | 549 / 786 | 522 / 685 ¹ | 580 / 788 | 598 / 738 ² |
| mean NLL, 6,120 tokens of Japanese prose (24 × 256) | 3.0558 | 3.1369 | 3.1605 | 3.1557 |

¹ This launch came straight after copying the checkpoint onto the NVMe, with the page
cache full. Not repeated yet.

² On its first launch the chunk calibration ran out of time while kernels compiled
(fixed since: `SGLANG_WARMUP_TIMEOUT`), so the chunk stayed at its fallback.

**Built from this repo vs the development tree** (same weights, same step): paired
NLL +0.0188 (SE 0.0111, z +1.69). That is inside the launch-to-launch spread below.

**Paired NLL, 1017 with the step minus without it:** −0.0237 (SE 0.0137, z −1.73,
6,120 tokens). That is lower, but within what two launches of one configuration differ
by on this machine (0.020-0.036, z up to −2.2). One launch each cannot tell the two
apart. The reason for the step is that it is the model's own arithmetic, bit-exact
against the reference code, and not this measurement.

The earlier release's NLL comes from different weights, so it is not a quality
comparison between the releases.

## Fixes from dsv41-flash-dual-epyc

Two fixes by f-wunderlich for the earlier release's repo are in the tree patch here too:

- **SM120 sparse MLA padding page** (dsv41-flash-dual-epyc#3). SGLang reserves
  physical KV page 0 for padding, and FlashInfer maps index -1 to entry 0 before
  it masks the score. The page-split scratch is persistent `torch.empty` storage,
  so page 0 could hold NaN, and a masked NaN still reaches the attention output
  through 0 × NaN. The page-split kernel now zeroes that page on every conversion.
- **Zero resident experts with Marlin** (#2). `--kt-num-gpu-experts 0` no longer
  asks Marlin to repack empty resident weights at load.

## Not done

- The resident-expert placement (`placements/`) was recorded from the earlier
  release's routing on Japanese prose. It loads and runs with 1017, but it is not
  1017's hottest set, and it fits no other workload better. To record your own, use
  `KT_ROUTING_DUMP` and SGLang's `--init-expert-location`.
- Tool namespaces and the new `drop_thinking` turn boundary are not in SGLang's
  DeepSeek-V4 chat encoding.
- Not run here: one GPU, more than two GPUs, AMD GPUs, CPUs with AVX-512.

## License

Apache-2.0 (see `LICENSE`), except `engram-adapter/`, which is derived from 0xSero's
engram adapter under the MIT License (see the headers of its files). The model weights
are under DeepSeek's license (see the model card).
