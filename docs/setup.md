# setup.sh and serve.sh

## What setup.sh does

1. Clones sgl-project/sglang at `8c7ec052` on its main branch and applies the SGLang patches
   up to `TIER` (default 2), each tier on top of the ones below it:

   | `TIER` | patch | what it adds |
   |---|---|---|
   | 0 | `sglang-tier0/` (nine patches, one concern each) | what 1017 needs to run correctly on the KTransformers hybrid MoE: the FP8 step below, the SwiGLU clamp passed to the CPU experts, the expert mask kt-kernel takes, `-1` ids kept out of the Marlin MoE, zero GPU experts, a configurable load barrier, the routed experts scaled by `routed_scaling_factor` once, V4.1's own KV cache formats on SM120 (below), and an automatic [SM89 attention fallback](sm89.md) |
   | 1 | `sglang-tier1/` (one patch per feature, [tier1.md](tier1.md); `SGLANG_FEATURES` picks some) | streamed prefill (its experts through llama.cpp's MMQ), the lends, hot experts and placement, the KV pool on remappable memory, the GPU MoE and FP8 kernels (the streamed prefill needs SM89 or newer) |
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

### Variables and what it writes

```
[MODEL=<checkpoint dir> | GGUF=<dir> [BASE=<dir>]] [TIER=0|1|2] [SGLANG_FEATURES=a,b,...] [PREBUILT=<archive>] ./setup.sh
```

- `MODEL`: the MXFP4 checkpoint. Writes `state/engram-<model>/engram-manifest.json`.
- `GGUF`: a GGUF pack, its shards and Engram file side by side ([gguf.md](gguf.md)). Writes
  `state/gguf-<name>/`. `BASE` is a directory with deepseek-ai/DeepSeek-V4.1-Flash-1017's
  `config.json`, tokenizer and `encoding/`; without it those few files are downloaded.
- `TIER`: 0, 1 or 2, default 2 (the table above). Changing it re-applies the patches and rebuilds kt-kernel.
- `SGLANG_FEATURES`: tier-1 features, comma-separated, or `none`; default all. A feature's
  prerequisites (`patches/sglang-tier1/requires.txt`) must be picked too.
- `PREBUILT`: an archive or directory from `tools/prebuilt-pack.sh` (below).
- `PREPARE_ONLY=1`: stops after the trees, the venv and the toolkit choice (for `tools/prebuilt-pack.sh`).
- `CUDA_HOME`: default `/usr/local/cuda`. Without a usable toolkit, NVIDIA's 13.2 toolkit wheels are
  installed under `build/` and serve as `CUDA_HOME` through `lib/cuda-home.sh`. That is enough for the
  run-time JITs and the small builds, not for kt-kernel's CMake build.
- `KT_CUDA_ARCHS`: default the visible GPUs' compute capability.
- `CPUINFER_CPU_INSTRUCT`: kt-kernel's CPU target, default `NATIVE` (this machine's ISA).
- `CMAKE_ARGS`: extra CMake flags for kt-kernel, space separated. Headers and libraries outside the
  system paths go in through `CPLUS_INCLUDE_PATH` and `LIBRARY_PATH`.

It needs git, a C/C++ compiler, the libnuma and hwloc headers (Debian/Ubuntu: `libnuma-dev libhwloc-dev`)
and either uv or python3.12. Everything it makes stays inside this directory: `third_party/sglang`,
`third_party/ktransformers`, `.venv`, `engram-adapter/librow_store.so`, `build/libkv_vmm.so` and `state/`.

### Prebuilt pieces

`PREBUILT=<archive> ./setup.sh` installs the pieces setup.sh otherwise compiles, from an archive
made by `tools/prebuilt-pack.sh` on a machine that has the toolkit: kt-kernel with its six CPU
variants (AVX2, AVX-512 base / VNNI / VBMI / BF16, AMX; one is picked at import), the streamed
prefill's MMQ extension for GPU archs sm89, sm90, sm100 and sm120, kv-vmm, the engram row store
and hwloc. The archive's `manifest.json` names the recipe, the pinned trees, the torch, CUDA and
Python versions and every file's SHA-256; setup.sh checks them all. Nothing is built at setup, and
the FlashInfer CUTLASS MoE of the MMQ-off path is not built. The kernels SGLang compiles at run time
(DeepGEMM's indexer logits, FlashInfer's sparse attention, TileLang's mHC, sgl-kernel's and Triton's
small kernels) are built on the first launch, about nine minutes on an RTX 5070 Ti, with the system
toolkit or, without one, the pip wheels' toolkit assembled by `lib/cuda-home.sh` into `build/cuda-home`.
The archive is for the pinned torch (`requirements.txt`) and Python 3.12. The archive for a tag is
on that tag's [release page](https://github.com/05yuki/deepseek-v4.1-flash-1017-ktransformers/releases),
with its SHA-256 beside it (34 MB). GitHub Actions builds it from the tag
(`.github/workflows/prebuilt.yml`, Ubuntu 24.04, no GPU), so it runs on glibc 2.39 and later, and a
maintainer publishes the release. Only archives built by this repository's CI or its maintainers
are published; please do not send binaries.

## What serve.sh does

At start it reads the GPUs, their compute capability, the NUMA nodes and the physical
cores, and sets tensor parallelism, the CPU pools and the CUDA arch from them. The
defaults are listed in the script. Each one can be overridden by the environment
variable named next to it. Among them:

- the tier-1 features that were applied, each with its switch and default in
  [tier1.md](tier1.md): the prefill experts streamed to the GPU, the hot lend
  (more hot experts in decode, in memory nothing else is using), `wo_a` in FP8, the vision
  tower in host memory and the rest
- an expert cache in RAM when one CPU pool's NUMA node holds less than 370 GiB
  ([One socket, 256 GB](results.md#one-socket-256-gb))
- a prefill chunk calibrated once, on the first launch of a setup (the server starts
  one extra time for this, about 12 minutes here). A shipped hint stands in for it when
  one matches the GPU model and count (`hints/kt-lend/`; RTX 5070 Ti, one or two) and
  the launch is at the defaults; the server replaces it with its own measurement at the
  first long prompt. `KT_LEND_SEED=0` calibrates instead; `tools/lend-hint-export.py`
  turns a calibrated setup's hint into a file to contribute
- FlashInfer autotune off, so outputs reproduce across launches
- V4.1's tool-call and reasoning parsers (below)

Logs go to `logs/<model>/<timestamp>/`. `serve.sh --background` returns once the
server process is started.

The first launch in a new environment compiles SGLang's and FlashInfer's kernels. Here
that took about 15 minutes, before the one-time chunk calibration. Later launches
reuse the cache (`.cache/` in the checkout) and are ready in about 3 minutes, with
the checkpoint on NVMe and loaded once before.

What each patch changes: [patches.md](patches.md) (tiers 0 and 2, kt-kernel) and
[tier1.md](tier1.md) (tier 1, one section a feature, with its switch and measured effect).
What made it this fast, measured change by change, and what was tried and dropped:
[optimizations.md](optimizations.md).
