# DeepSeek-V4.1-Flash-1017 on SGLang + KTransformers (CPU/GPU hybrid inference)

Run [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) on one machine.

- 512 GB of RAM and one or two ordinary GPUs: 50.0 tok/s decode on two RTX 5070 Ti ([Results](#results)).
- One socket with 256 GB: 20.2 tok/s ([One socket, 256 GB](docs/results.md#one-socket-256-gb)).
- 64 GB of RAM and one 16 GB GPU, with a GGUF pack of the model ([GGUF packs](#gguf-packs)).
- A context of up to 1M tokens.
- The reference's accuracy: on 24 public-domain texts the served model scores within the standard
  error of DeepSeek's own `inference/model.py`. A patch stays only while that holds.

How it runs: the 384 routed experts a layer run on the CPU in MXFP4 through
[KTransformers](https://github.com/kvcache-ai/ktransformers)' kt-kernel.
Attention, the hottest experts and the prefill run on the GPUs through
[SGLang](https://github.com/sgl-project/sglang).
The engram tables (about 200 GB) are read from the checkpoint on disk, with no extracted copy.

This repository is everything needed: patches against two public upstream commits, a setup
script that builds them, a launcher that reads the machine it runs on, and tests.
Each tag pins those commits: SGLang main at `58484246` and KTransformers `95009ea` today, with
torch 2.14.1 on CUDA 13 wheels, FlashInfer 0.7.2 and CUDA toolkit 13.1.
The recipe follows SGLang main: every few weeks the patches move to a newer commit, are measured
again, and go out under a new tag.
The SGLang patches are the commits of branch `dsv41-1017-main` of
[05yuki/sglang-upstream](https://github.com/05yuki/sglang-upstream/tree/dsv41-1017-main), one file
a commit ([docs/patches.md](docs/patches.md)).
A patch that SGLang merges is dropped at the next move.

```
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 --local-dir /models/dsv41-1017
MODEL=/models/dsv41-1017 ./setup.sh
MODEL=/models/dsv41-1017 ./serve.sh
```

Or all of it in one command:

```
./run.sh
```

`run.sh` runs `setup.sh`, downloads the checkpoint into `./models/` when it is not there, and starts `serve.sh`.
`MODEL=<dir>` uses a checkpoint already downloaded.
It remembers the directory in `state/run.env`, so the next `./run.sh` only starts the server.

`MODEL=/models/dsv41-1017 tools/check.sh` starts the server once, sends one short request,
checks that an answer comes back, and stops the server: the one command that says a setup works.

`MODEL=<dir> tools/qualify.sh` checks a machine end to end and writes one Markdown report.
It takes about 15 minutes after the server is up ([docs/qualification.md](docs/qualification.md)).
Reports are filed by tag in [reports/](reports/README.md), and one from another GPU or CPU is welcome.

The server speaks SGLang's OpenAI-compatible API on `127.0.0.1:8080`.
It returns tool calls and the model's thinking in their own fields, and keeps a prefix cache with a copy in host memory.
Serving more than one request at a time and other machines: [docs/serving.md](docs/serving.md).

## What it needs

In short, from [docs/requirements.md](docs/requirements.md):

- **RAM:** a 512 GB class machine, or one socket with 256 GB; 64 GB with a GGUF pack. With the MXFP4 checkpoint the server uses about 420 GiB, most of it the CPU experts.
- **GPU:** NVIDIA SM89 (RTX 40) or newer, 16 GB a card, one or two cards.
- **Disk:** about 500 GB on NVMe. The checkpoint is 476 GiB, and its engram rows are read from it while serving.
- **Software:** Linux, gcc/g++, the libnuma and hwloc headers, and a CUDA 13 toolkit; without one, setup.sh fetches NVIDIA's toolkit wheels.
- **Host settings,** as root: `transparent_hugepage/shmem_enabled` set to `advise`, `vm.min_free_kbytes = 8388608`, and swap off.

A prebuilt archive for each tag skips every build at setup: `PREBUILT=<archive> ./setup.sh` ([docs/setup.md](docs/setup.md#prebuilt-pieces)).

## GGUF packs

The same tree serves [ISTA-DASLab's GSQ-RCO GGUF](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF) of the model.
One 16 GB GPU and 64 GB of RAM are enough for it.
`BUILD=IQ2_XXS ./run.sh` downloads the IQ2_XXS build and serves it.
`GGUF=<dir> ./setup.sh` and `GGUF=<dir> ./serve.sh` take a build already downloaded.
On the two-GPU machine below, the IQ2_XXS build decodes Japanese prose at 62.6 tok/s.
On one of its GPUs under a 64 GB memory limit, it decodes Japanese prose at 25.7 tok/s and reads a 35,000-token prompt in 28 s.
Its NLL is 0.327 above the reference, from the pack's quantization.
What it needs, how it runs and its results: [docs/gguf.md](docs/gguf.md).

## Results

2 × EPYC 7452 (Zen 2), 512 GB DDR4, 2 × RTX 5070 Ti 16 GB, the `serve.sh` defaults with a 1M-token context.
Decode is Japanese prose, the median of four prompts.
The prompts are random token ids.

| setup | decode | 35K prompt | 1M prompt |
|---|---|---|---|
| two GPUs | 50.0 tok/s | 20 s | 16 min |
| one GPU | 33.4 tok/s | 33 s | 21 min |
| one socket, 256 GB, one GPU | 20.2 tok/s | 50 s | 27 min |

Other kinds of text, more prompt lengths and how each setup was run: [docs/results.md](docs/results.md).

The served model scores within the standard error of DeepSeek's own `inference/model.py` on the same 24 pieces.
Its mean NLL is 0.7192 on two GPUs and 0.7224 on one, against 0.7185 for the reference ([docs/results.md](docs/results.md#accuracy)).

## Documents

- [docs/glossary.md](docs/glossary.md): the words used here (tier, lend, hot experts, chunk hint and the rest).
- [docs/requirements.md](docs/requirements.md): RAM, VRAM, disk, software, host settings, containers.
- [docs/setup.md](docs/setup.md): what `setup.sh` and `serve.sh` do, and the prebuilt archive.
- [docs/serving.md](docs/serving.md): the API, concurrent requests, the prefix cache, tool calls and thinking.
- [docs/results.md](docs/results.md): speed on two GPUs, one GPU and one socket, and accuracy.
- [docs/qualification.md](docs/qualification.md): `tools/qualify.sh` and the multi-session prefix-cache check.
- [docs/kv-cache.md](docs/kv-cache.md): V4.1's KV formats and the KV pool past the GPU.
- [docs/1017-changes.md](docs/1017-changes.md): what 1017 changes and how the recipe runs its FP8 step.
- [docs/patches.md](docs/patches.md) and [docs/tier1.md](docs/tier1.md): every patch, with its switch and measured effect.
- [docs/optimizations.md](docs/optimizations.md): what made it fast, and what was tried and dropped.
- [docs/sm89.md](docs/sm89.md): Ada (RTX 40) attention.
- [docs/gguf.md](docs/gguf.md): serving a GGUF pack of the model.

## Fixes from dsv41-flash-dual-epyc

Two fixes by f-wunderlich, from the earlier release's repo: `--kt-num-gpu-experts 0` starts
with Marlin (#2, tier 0 for the MoE call and tier 1's `loading` for the loader). The SM120
padding-page fix (#3) is no longer carried: SGLang main's sparse attention with FlashInfer
0.7.2 runs a 48K-token prefill without it, with the same NLL.

## Not done

- Results are on 2 × RTX 5070 Ti (SM120) with Zen 2 CPUs; SM89 was checked by a
  contributor on one card ([docs/sm89.md](docs/sm89.md)).
- CPUs with AVX-512 should work: the experts run on the AVX2 kernels, which those CPUs
  also have.
- More than two GPUs: `serve.sh` sets tensor parallelism from the GPUs it sees; not
  tested.
- Older NVIDIA GPUs: RTX 30 and A100 do not run; DeepSeek-V4's attention here needs SM89
  or newer. With `SGLANG_MXFP4_MOE_FP32=0`, the MXFP4 Marlin path also needs SM90 or SM120.
- AMD GPUs: not supported (kv-vmm and kt-kernel use CUDA).

## License

Apache-2.0 (see `LICENSE`), except `engram-adapter/`, which is derived from 0xSero's
engram adapter under the MIT License (see the headers of its files). The model weights
are under DeepSeek's license (see the model card).
