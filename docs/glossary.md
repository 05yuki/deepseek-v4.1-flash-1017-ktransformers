# Glossary

The words the README and the docs use for this recipe's parts.
Each entry links to the section that describes the part in full.

**Tier.** A layer of SGLang patches, chosen with `TIER` at setup.
Tier 0 is what 1017 needs to run correctly, tier 1 is the speed features, and tier 2 is the
all-reduce for GPUs without peer access.
Each tier applies on top of the ones below it ([setup.md](setup.md)).

**Feature.** One tier-1 patch, named like `hot-lend` or `stream-prefill`.
`SGLANG_FEATURES` picks some of them at setup ([tier1.md](tier1.md)).

**CPU experts.** The routed experts that kt-kernel runs on the CPU in MXFP4.
They live in RAM on 2 MB pages, one pool per NUMA node.

**Resident expert.** The one expert a layer whose weights stay on the GPU.

**Hot experts.** More experts a layer held on the GPU during decode, next to the resident one.
The GPU computes them, so the CPU skips them.

**Placement.** The file that orders each layer's experts, the most used first.
It decides which experts are resident when the server starts ([tier1.md](tier1.md#11-placement)).

**Hot lend.** The tier-1 feature that holds hot experts in GPU memory nothing else is using:
the KV pool's free pages and the driver's idle memory.
It gives that memory back when the KV pool or a prefill needs it ([tier1.md](tier1.md#16-hot-lend)).

**Dynamic slots.** Hot experts that follow the text being written.
The experts routed to most are copied in from RAM, so no routing has to be recorded beforehand.

**Streamed prefill.** A long prompt's MoE run on the GPU, with the CPU experts copied to it
a group at a time while the previous group computes ([tier1.md](tier1.md#14-stream-prefill)).
Below its threshold, a prompt's experts run on the CPU.

**MMQ.** llama.cpp's integer matrix-multiply kernels for quantized weights.
The streamed prefill runs the MXFP4 experts through them ([tier1.md](tier1.md#18-stream-mmq)).

**Prefill lend.** During a streamed prefill, the GPU-resident weights give their memory to the
prefill's scratch ([tier1.md](tier1.md#13-prefill-lend)).
Each layer's weights come back at the same addresses just before that layer runs.
The room a prefill chunk needs comes from this lend.

**KV lend plan.** The prefill lend's second source: the KV pool's free pages, mapped out for the
prefill and back for decode, with the chunk sizes planned from them ([tier1.md](tier1.md#15-kv-lend-plan)).

**kv-vmm.** A small library that keeps each KV pool layer on remappable GPU memory, so its free
pages can be lent and its tail can live in host memory ([tier1.md](tier1.md#12-kv-vmm)).

**Chunk.** The part of a long prompt prefilled at once.
Its size is calibrated on the first launch to the GPU memory that is left.

**Chunk hint.** The calibrated chunk sizes, kept between launches.
Shipped hints in `hints/kt-lend/` stand in for the calibration on known GPUs ([setup.md](setup.md#what-servesh-does)).

**Host KV.** The compressed KV that does not fit the GPU, kept in host memory so the KV pool
covers the whole context ([kv-cache.md](kv-cache.md#the-kv-pool-past-the-gpu)).

**KV pool.** The GPU memory that holds the KV cache, sized for the whole context.
The lends take only its free pages, so it never shrinks to gain speed.

**Expert cache.** On a NUMA node with less than 370 GiB, the experts kept in RAM, with the
others read from the checkpoint when routed ([tier1.md](tier1.md#20-expert-cache)).

**Engram.** V4.1's n-gram memory tables, about 200 GB.
They are read straight from the checkpoint on disk, a few rows a token.

**Prefix cache.** SGLang's radix cache: a prompt that starts like an earlier one reads that part from the cache.

**HiCache.** SGLang's copy of the prefix cache in host memory, so an evicted prefix can come back
without a prefill ([tier1.md](tier1.md#21-hicache)).

**Prebuilt archive.** The compiled pieces of setup in one file, built by CI for each tag
([setup.md](setup.md#prebuilt-pieces)).

**Qualification report.** The output of `tools/qualify.sh`: checks and measurements of one machine
at one tag ([qualification.md](qualification.md), [reports/](../reports/README.md)).

**Reference NLL.** The scores of DeepSeek's own `inference/model.py` on 24 public-domain pieces.
The served model is compared against them ([results.md](results.md#accuracy)).
