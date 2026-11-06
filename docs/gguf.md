# GGUF packs

The same tree that serves the MXFP4 checkpoint serves a GGUF pack of it
(ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF: IQ1_M, IQ2_XXS or IQ2_XS, with the Engram file).

```
GGUF=<dir with the GGUF shards and the Engram file> ./setup.sh
GGUF=<the same dir> ./serve.sh
```

`BUILD=IQ2_XXS ./run.sh` downloads a build and serves it (`lib/run-gguf.sh`).
`serve.sh` hands a GGUF pack to `lib/serve-gguf.sh`.

The GGUF code is in these patches, on top of the MXFP4 recipe's ([patches.md](patches.md)):

| file | what |
|---|---|
| `patches/sglang-tier0.5/01-gguf.patch` | tier 0.5, at every tier: the GGUF format on tier 0 (the dense weights from the GGUF, the GGUF `wo_a`, the GPU hot method, the dequantizers, SM75). `setup.sh` first copies SGLang's GGUF `ggml-common.h` and `vecdotq.cuh` into `kt_gguf_moe_multi/`. |
| `patches/sglang-tier1/26-gguf-lend.patch` | tier 1's lends, streamed prefill and hot slots with a GGUF pack ([tier1.md](tier1.md#26-gguf-lend)) |
| `patches/kt-kernel-gguf.patch` | kt-kernel's LLAMAFILE backend for the GGUF experts (tier 1) |
| `patches/kt-llamafile-iq2xs.patch` | IQ2_XS in kt-kernel's x86 kernels (tier 1) |

Serving a GGUF pack needs tier 1 (kt-kernel's GGUF backend) with `gguf-lend` and `stream-mmq`.
Nothing in these patches runs for the MXFP4 checkpoint: the GGUF code takes a GGUF pack only
(`KT_GGUF_V41_DENSE`, which `lib/serve-gguf.sh` sets, or kt-kernel's LLAMAFILE method).
A switch marked "on" is on in `lib/serve-gguf.sh`.

The MMQ extension the GGUF patches use (`kt_gguf_moe_mmq`, llama.cpp's kernels) comes with tier 1's
[`stream-mmq`](tier1.md#18-stream-mmq), which the MXFP4 checkpoint's streamed prefill also uses.

## Downloading a build

```
hf download ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF \
  --revision a419d74cd28d5b9f8c5e2b9077d11a609c17bc32 \
  --include "*IQ2_XXS*" "*Engram*" --local-dir /models/dsv41-gsq-iq2xxs
GGUF=/models/dsv41-gsq-iq2xxs ./setup.sh
GGUF=/models/dsv41-gsq-iq2xxs ./serve.sh
```

`BUILD=IQ2_XXS ./run.sh` does the same into `./models/` (`BUILD=IQ1_M` and `BUILD=IQ2_XS` take the other builds).
It remembers the directory in `state/run-gguf.env`, so the next `BUILD=... ./run.sh` only starts the server.
The revision is the pack in llama.cpp's own layout (six shards a build, the Engram tables as one tensor).
Downloads in the earlier layout (four shards a build) work too.
`GGUF=<dir> tools/check.sh` starts the server once, sends one short request and stops it.
The GGUF has no vision tower, so the model is served as text only.

## What a pack needs

- **GPU:** one 16 GB card is enough; an 8 GB RTX 20 (SM75) runs it too ([Turing](#turing-sm75)).
  The dense layers in their GGUF types take 6.8 GB.
  The rest is the KV pool and the working room of the prefill chunk and the extra hot experts.
  With several GPUs visible, `serve.sh` uses them all ([Two or more GPUs](#two-or-more-gpus)); `CUDA_VISIBLE_DEVICES` or `KT_GGUF_TP` picks fewer.
- **CPU and RAM:** the socket each GPU hangs off. 64 GB of RAM is enough, and more is faster.
  `serve.sh` runs the experts on that NUMA node's physical cores and sizes the expert cache from its memory.
  The cache holds the experts' GGUF rows (124 GiB for IQ2_XXS) as far as the node has room, less 18 GiB (`KT_GGUF_HOST_RESERVE_GIB`).
  What does not fit is read from the file again when it is routed.
  When the node holds every expert and the IQ1_M decode copy too (76 GiB more for IQ2_XXS), the copy is kept ([The expert cache](#the-expert-cache-kt-kernel)).
  A cgroup memory limit counts: `systemd-run --user --scope -p MemoryMax=64G ./serve.sh` runs it as a 64 GB machine.
- **Context:** the model's 1,048,576 tokens on one GPU when the socket's RAM holds every expert; 131,072 tokens otherwise and on two GPUs; 32,768 on a card under 12 GB (`SGLANG_CONTEXT_LENGTH`).
- **Disk:** NVMe, about 260 GB for IQ2_XXS (138 GB of shards and the 111 GB Engram file).
  The engram rows, the experts RAM does not hold and the prefill's experts are read from it.
- **Software:** what the MXFP4 checkpoint needs ([requirements.md](requirements.md)), with a CUDA toolkit: `setup.sh` builds the GGUF kernels for the GPU it runs on.
  The prebuilt archive's kt-kernel has the GGUF backend; the GGUF CUDA kernels are not in it, so `GGUF=<dir> ./setup.sh` builds them.
- **Host settings:** none needed. The RTX 2070 machine runs the pack with swap on, shmem THP `never` and the
  default `vm.min_free_kbytes`. `run.sh` notes swap and a low `vm.min_free_kbytes`, which
  [requirements.md](requirements.md) recommends for the MXFP4 checkpoint.

`GGUF=<dir> ./setup.sh` also writes `state/gguf-<dir name>/`:

- `config/`: the base model's `config.json`, tokenizer and chat encoding from deepseek-ai/DeepSeek-V4.1-Flash-1017 (a few MB, no weights; `BASE=<dir>` takes them from a local copy), with `quantization_config` set to GGUF (`tools/make-gguf-config.py`);
- `engram/engram-manifest.json`: where each engram table lies in the Engram GGUF;
- `experts.json`: the routed experts' bytes, for the cache size;
- `io-probe.json`: how fast the drive under the GGUF reads, and from how many threads (`tools/io-probe.py`, about a minute, reads only).

## Results

AMD EPYC 7452 (Zen 2, AVX2), RTX 5070 Ti 16 GB, the IQ2_XXS build on one NVMe drive, the `lib/serve-gguf.sh` defaults, one request at a time, `tools/bench.py`.

**Two GPUs and two sockets** (64 cores, 512 GB; the expert cache holds every expert and the IQ1_M decode copy), one launch of this tree:

| text | decode, tok/s |
|---|---|
| Japanese prose (median of four prompts) | 62.6 |
| English prose | 46.7 |
| JSON | 41.1 |
| math | 39.7 |
| English technical prose | 37.1 |
| Python code | 36.5 |
| Chinese prose | 31.5 |

A 35,000-token prompt prefills in 34.0 s the first time (1,030 tok/s) and in 18.0 s the second (1,947 tok/s).

**One GPU and one socket** (32 cores, 256 GB), measured with the GGUF recipe's earlier tree (SGLang main `20036b7a`).
The 64 GB and 128 GB columns are that socket under a memory limit, not a desktop of that size.

| RAM | expert cache | of the experts' 124 GiB | expert reads that hit it |
|---|---|---|---|
| 64 GB | 46 GiB | 37% | 90.2% |
| 128 GB | 110 GiB | 89% | 98.2% |
| 251 GiB (the socket) | 195 GiB, with the IQ1_M decode copy | all | 99.1% |

| text, decode tok/s | 64 GB | 128 GB | 251 GiB |
|---|---|---|---|
| Japanese prose (median of four prompts) | 25.7 | 33.4 | 43.5 |
| JSON | 18.3 | 24.3 | 31.0 |
| Chinese prose | 17.9 | 28.6 | 34.1 |
| English prose | 17.3 | 25.2 | 31.1 |
| math | 17.3 | 24.8 | 31.2 |
| Python code | 13.6 | 21.5 | 27.6 |
| English technical prose | 13.3 | 21.2 | 27.9 |

| prefill | 64 GB | 128 GB | 251 GiB |
|---|---|---|---|
| 35,000 tokens | 1,268 tok/s (28 s) | 1,855 tok/s (19 s) | 2,010 tok/s (17 s) |
| 131,072 tokens | | | 1,706 tok/s (77 s) |
| 1,000,000 tokens | | | 1,022 tok/s (16 min) |

**An RTX 2070 8 GB and a Core i5-9600K** (6 cores, two-channel DDR4, a PCIe 3 drive, 64 GB limit, 32,768-token context): decode 1.8 tok/s, an 8,000-token prefill 34 tok/s.
The six CPU cores set the pace.

## Accuracy

Scored as for the MXFP4 checkpoint ([results.md](results.md#accuracy)), against DeepSeek's `inference/model.py` on the FP8 checkpoint (0.7185):

| run | mean NLL | minus the reference (SE) |
|---|---|---|
| IQ2_XXS, two GPUs, this tree | 1.0460 | +0.327 (0.053) |
| IQ2_XXS, one GPU | 1.0479 | +0.329 (0.054) |
| IQ2_XXS, RTX 2070 (`KT_GPU_EXPERTS=0`) | 1.0468 | +0.328 (0.053) |

The difference is the pack's quantization, not the serving.
Served with only the routed experts from the GGUF (the dense layers and the engram from the FP8 checkpoint), the NLL is +0.312 above the reference.
The dense layers in their GGUF types add 0.007, and the IQ4_NL engram tables 0.006.
`tools/nll.py score state/gguf-<dir name>/config served.json` scores a served pack.

## Weights

The pack comes in two layouts, and both load.

- **Earlier** (before 2026-10-27): four shards a build; each Engram layer its own table (`blk.N.engram_embd`); the Engram gate as two BF16 tensors, `engram_q` and `engram_k`.
- **llama.cpp's** (2026-10-27 and later): five shards a build plus the Engram file as shard 6; the Engram layers' rows one after another in `per_layer_token_embd`, their sizes in the first shard's metadata (`deepseek41.ple.*`); the gate folded into one F32 tensor, `engram_gate` = q × k.
- The quantized weights are the same bytes in both.
- `tools/make-engram-manifest-gguf.py` finds each Engram layer's rows in either.
- The F32 gate is split back at load into a bf16 pair whose product is exactly the gate (`_split_gate` in `kt_gguf_dense_v41.py`). The gate is a product of two bf16 values, so its significand is a product of two integers below 256, and any such pair of factors serves.
- So the Engram module and its fused gate kernel run unchanged on either layout. Reading the F32 gate as one weight instead was 0.29 µs a call slower (2.19 against 1.89 µs at one token).
- With the hot slots held fixed, the two layouts give the same NLL on all 11,496 tokens.

- **Dense layers** (`kt_gguf_dense_v41.py`, `quantization/gguf.py`, the loader): every dense tensor is
  read from the GGUF shards and kept in its GGUF type on the GPU (Q5_K and others in this pack). The
  compressor and the indexer are dequantized to bf16 at load (`modules_to_not_convert`). Decode
  multiplies the GGUF rows with vector kernels taken from llama.cpp (`kt_gguf_moe_multi/`).
- **Dense prefill** (64 rows or more): llama.cpp's MMQ multiplies the same GGUF blocks
  (`kt_gguf_moe_mmq/`, Q5_K and Q8_0 here). No bf16 copy of a weight is made. The activations are
  quantized to q8_1 into one 32 MiB buffer, a slice of rows at a time (`SGLANG_GGUF_MMQ_SLICE_MB`),
  and the output is written in bf16. Tensors whose rows are not a multiple of 512 values (q_b and the
  shared expert's down projection here) get 176 zero bytes after their last row at load, which the
  kernel reads. `SGLANG_GGUF_PREFILL_MMQ=0` goes back to a bf16 copy of each weight and a bf16 GEMM.
  Against that path on the RTX 5070 Ti: the window's allocated peak is 0.3 to 0.5 GB lower, a warm
  35K prefill 4 to 6% faster, and the NLL the same (+0.0010 and +0.0018 over two launches, SE 0.003).
- **Embedding**: the token embedding stays in its GGUF type (Q5_K, 434 MiB instead of a 1,262 MiB bf16
  copy); the rows a batch looks up are opened in float32 in gguf-py's order of operations, so the
  embeddings are the same values the bf16 copy held.
- **RoPE tables**: built for the served context (`SGLANG_CONTEXT_LENGTH`) instead of the model's
  1,048,576 positions, one copy for all the layers (two tables of 512 MiB each at 1M; their rows
  depend only on the position). Past `KT_ROPE_DEVICE_ROWS` (131,072) rows a table goes on in host
  memory of the GPU's NUMA node, in the same address range (the kv-vmm split range host-kv uses):
  at 1M that keeps 0.75 GB on the GPU for the lends, and the rows past 131,072 are read over PCIe.
- **Routed experts**: kt-kernel's LLAMAFILE backend computes them from the GGUF's rows
  (`--kt-method LLAMAFILE`, `KT_GGUF_ZEROCOPY=1`, on): no arena copy of the experts at load.
- **Engram**: the two tables are read row by row from the Engram GGUF (IQ4_NL, 144 bytes a row)
  and dequantized on the GPU (`engram-adapter/`).

## The expert cache (kt-kernel)

`KT_GGUF_EXPERT_CACHE_GIB` (sized by `serve.sh` from the node's memory) is kt-kernel's own RAM tier
for the experts: an expert's gate, up and down rows are read once from the file with `O_DIRECT`
into fixed slots on the node and computed from there; when the cache is full, the least recently
used experts give their slots back. Reader threads keep many reads in flight
([Drive reads](#drive-reads) says where they run and how many there are). The reads bypass
the page cache (plain reads only where the file system refuses `O_DIRECT`), so the cache's size is
the RAM the experts take.

`KT_GGUF_CACHE_IQ1M_T=1` (on when the node holds every expert and the copy) keeps, beside each
IQ1_M tensor's raw rows, a copy laid out for the CPU decode: 76 bytes a block of 256 values (IQ1_M:
56). Its GEMM decodes 32 values with a shift, a mask and a byte shuffle and then does the same
integer dot products in the same order as iqk's IQ1_M kernel, so the result is bit-identical. An
IQ1_M layer's CPU work in decode goes from 0.94 to 0.55 ms. The raw rows stay for the GPU readers
below.

`kt-llamafile-iq2xs.patch` adds IQ2_XS to the x86 kernels kt-kernel uses (iqk's matrix
multiplication, from ik_llama.cpp); the pack's IQ2_XS tensors run there.

## Drive reads

Four kinds of threads read the GGUF from the drive: the expert cache's readers, the hot lend's
swaps, the streamed prefill's readers and the engram reads. `lib/serve-gguf.sh` keeps the first three off
the CPUs the experts compute on, and sizes them from the drive.

- **How many readers.** `setup.sh` runs `tools/io-probe.py` once on the GGUF's drive: random 512 KiB
  and 2 MiB `O_DIRECT` reads from 1 to 32 threads. It writes `state/gguf-<name>/io-probe.json`.
  The cache and the prefill each get the fewest readers that reach 90% of the drive's best rate
  (`KT_GGUF_PREFETCH_THREADS`, `KT_GGUF_STREAM_READERS`).
- **Where, with SMT.** The readers and the swaps run on the second hardware threads of the node's
  last cores, one for every four readers (`KT_IO_CPUS`). The experts compute on every core.
- **Where, without SMT.** The readers and the swaps share the node's last cores, as many as
  reading the drive at full speed keeps busy (the probe's rate times its CPU time a GB, rounded
  up). The experts compute on the rest.
- **The engram reads** are left to the scheduler. `DSV41_ENGRAM_IO_CPUS` pins them; pinned to the
  readers' CPUs or to the other second hardware threads, the decode was a few percent slower on
  some texts.
- **Without a probe** (a setup from before it, or a GGUF moved to another drive), `serve.sh`
  takes 16 readers: four second hardware threads with SMT, one core without. Remove
  `state/gguf-<name>/io-probe.json` and run `GGUF=<dir> ./setup.sh` to measure again.

The launch prints the result, e.g. `drive reads: 16 cache readers and 16 prefill readers on CPUs
92,93,94,95`. Every value can be set by hand: `KT_IO_CPUS`, `DSV41_ENGRAM_IO_CPUS`,
`KT_CPU_THREADS` and the two reader counts.

What it was measured on, two launches each:

- **One EPYC 7452 socket (SMT), an SN5100 (16 readers).** The experts on all 32 cores and the
  readers on four second hardware threads decode 2 to 10% faster than with four whole cores given
  to the readers. That holds at 64 GB and with every expert in RAM. The prefill does not change.
- **An i5-9600K (6 cores, no SMT), a KIOXIA EXCERIA PLUS G2 (8 readers).** One core stays with the
  reads. Computing on all six cores decoded 3 to 6% slower.

## Hot experts on the GPU

The MXFP4 recipe's hot lend ([tier1.md](tier1.md#16-hot-lend)) with GGUF slots (`KT_HOT_BACKEND=gguf`,
`kt_gguf_hot.py`): one expert a layer is resident, and the KV pool's free pages and the driver's
idle memory hold as many more as fit, in their GGUF bytes. Decode runs them with the vector
kernels above and passes the routes they do not hold to kt-kernel (`-1`).

Every 8 forwards each layer swaps slots whose experts are routed less for experts routed more
(`KT_KV_HOT_DYNAMIC_SLOTS=1`, on). A swap's rows come from the expert cache, or from the file when
the cache does not hold them. They are read on two background threads into a pinned staging pool
(`KT_KV_HOT_DYN_ASYNC=1`, on; 24 buffers of 9.8 MiB), copied to the slots on a side stream at the
next round, and mapped at the round after: the forward never waits on a read. The table update
goes through pinned memory without a host sync. `KT_KV_HOT_DYN_VERIFY=1` reads every swapped slot
back and compares it with the file.

## Prefill

The MXFP4 recipe's streamed prefill, fed from the GGUF (`kt_stream_prefill_gguf_nvme.py`,
`KT_GGUF_STREAM_SOURCE=nvme`, on above 1024 rows): reader threads (`KT_GGUF_STREAM_READERS`, 16)
read each layer's experts with `O_DIRECT` into a ring of pinned buffers (`KT_GGUF_STREAM_RING`, 16),
taking the ones the expert cache holds from RAM, and the GPU multiplies them in their GGUF types
with llama.cpp's MMQ kernels (`kt_gguf_moe_mmq/`, `KT_GGUF_STREAM_MMQ=1`, on), built for the GPU's
arch. The lends work as in the MXFP4 recipe; the lend's copy of the dense weights is read back from
the GGUF instead of kept in host memory (`KT_PREFILL_LEND_MIRROR=gguf`, on).

The MMQ body keeps no inputs of its own.
It reads the layer's hidden states, ids and weights where they are, and adds into the MoE output
(`KT_GGUF_STREAM_INTO=1`, on).
A group's routes go through the kernels 8,192 at a time (`KT_GGUF_STREAM_MMQ_ROUTES`).
So the group's temporary buffers keep one size however long the window.
On one 16 GB card a window of 38,912 rows fits; before, 30,720 ran out of memory.

The prefill plan still sizes windows at 460 KiB a row, so a 35,000-token prompt runs in two.
A lower cost a row puts it in one window: `KT_PREFILL_PLAN_COST_PRIOR=2600,274432,280,820`.
That only pays when the experts come from the drive.
With the expert cache cold it was 58% faster, at 64 GB up to 17%, with every expert in RAM no faster.
The one window also leaves PyTorch's allocator a few retries at its end, so it is not the default.

## Turing (SM75)

`serve.sh` lets an RTX 20 card through and sets it up as below (all of it inert on SM80 and newer):

- **Attention**: SGLang's Triton sparse decode for Ada (SM89, tier 0) serves SM75 too. Triton has no
  FP8 type and no bf16 dot below SM80, so the kernel opens the e4m3 KV bytes from their bit fields
  (the same values as the FP8 conversion: checked for all 256 bytes) and takes its dots in float32.
  The compressed KV's FP4 scale is rounded to e4m3 with integer ops (ties to even, as the cvt does).
- **bf16**: SGLang switches pre-SM80 GPUs to fp16; `KT_SM75_BF16=1` and `--dtype bfloat16` keep bf16,
  which DSV4's JIT kernels need. The MoE router's bf16 GEMM runs in float32.
- **sgl-kernel ops**: the sgl-kernel wheel carries sm_80 and newer code only. `kt_sm75_ops.py` builds
  its GGUF, activation, RoPE, top-k and `moe_sum` sources (in SGLang's tree) for the GPU's arch with
  torch's JIT loader (`setup.sh` does it once) and registers them as those ops' CUDA kernels.
- **Prefill**: the dense layers' prefill runs on the same llama.cpp MMQ, built for sm_75. The bf16
  copy path is not an option there: it needs a bf16 cuBLAS GEMM (SM80+). Against sgl-kernel's older
  dp4a MMQ (`SGLANG_GGUF_PREFILL_MMQ=0`), an 8K prefill on the RTX 2070 took 149 to 158 s instead of
  181 to 189 s (two launches each).
- **Context and chunk**: a card under 12 GB defaults to 32,768 tokens. The prefill chunk is 2,048
  tokens with no calibration launch: on SM75 the TileLang mHC kernels (`hc_split_sinkhorn`,
  `mhc_post`) fail to launch inside the server on 8,192-token chunks (`CUDA_ERROR_INVALID_VALUE`;
  they run at that size on their own, the cause is not found), and the CPU sets the prefill's pace.

V100 (SM70) is not covered: CUDA 13 dropped it, so it would take a CUDA 12 build of the whole stack.

## Two or more GPUs

`serve.sh` uses every visible GPU, one tensor-parallel rank each (`KT_GGUF_TP=<n>` uses fewer).
Each GPU's CPU work runs on the NUMA node it hangs off, one kt-kernel pool a node.

A GGUF block holds 256 values and cannot be cut.
V4.1's expert intermediate is 2304; split in two it is 1152, not a whole number of blocks.
So the GPUs share the experts, not the rows of each expert:

- **CPU experts.** kt-kernel splits each expert's rows between the nodes by whole blocks
  (1280 and 1024), as it already did.
- **Shared expert.** It is kept whole on every GPU (`SGLANG_SHARED_EXPERT_TP1=1`).
  That costs 0.49 GB a GPU.
- **Hot experts.** Each GPU's slots hold whole experts, only those with `e % world == rank`
  (`KT_GGUF_HOT_EP=1`, on with more than one rank).
  A route goes to the GPU whose slot holds its expert.
  Rank 0's CPU skips what any GPU's slot holds: the GPUs' slot tables are joined with an
  all-reduce whenever one changes.
  The resident expert is whole on every GPU and computed by rank 0 only.
- **Prefill.** Each GPU streams whole experts for its own share of the layer
  (`KT_GGUF_STREAM_EP=1`, on with more than one rank).
  The MoE output's all-reduce adds the shares.
- **Prefill reads.** The expert cache holds each node's row slice, not whole experts.
  So the prefill reads whole experts through the page cache (`KT_GGUF_STREAM_DIRECT=0`) when
  the nodes have room for the file on top of the cache.
  With less memory it reads them with `O_DIRECT`.

GPUs that differ are handled like this:

- Settings every rank shares follow the weakest card: the lowest compute capability, the
  smallest memory.
- Each GPU keeps its own number of hot slots.
- The hot lend runs only when every GPU has something to lend.
- The prefill's share of the experts follows each GPU's SM count.
  `KT_GGUF_STREAM_EP_WEIGHTS=3,2` sets it by hand.
- SGLang's check that the GPUs have the same memory is turned off when they do not.

Two GPUs that differ in memory were run: GPU 1 with 4 GiB taken by another process.
GPUs of two architectures (an RTX 50 with an RTX 20, say) should work but were not run.

## Not carried over from the MXFP4 recipe

- The placement file and the static hot experts: the dynamic slots find the hot experts themselves.
- `wo_a` in FP8: it is a GGUF tensor here.
- The vision tower: the GGUF has none.

## Not done

- GPUs other than SM120 and SM75 were not run with a pack.
- The IQ1_M and IQ2_XS builds: their types are supported, only IQ2_XXS was measured.
- The 1M context on two GPUs: `lib/serve-gguf.sh` keeps 131,072 tokens there.
- A prompt a few dozen tokens short of a 131,072-token context (131,000 token ids) stalls the server instead of being refused.
