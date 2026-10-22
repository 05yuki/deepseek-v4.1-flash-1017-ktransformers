# How the numbers were reached

What made decode, prefill and the launch as fast as they are, one mechanism at a time,
with what it was measured to change. What each patch does and its switch:
[patches.md](patches.md) and [tier1.md](tier1.md).

All of it ran on the machine in the README (2 × EPYC 7452, 2 × RTX 5070 Ti 16 GB). Much of
it was measured on the earlier DeepSeek-V4.1-Flash release, before 1017 existed; each item
says which. Numbers from different items come from different builds, so compare within an
item, not across items.

How it was measured:

- **Decode**: the server's own generation rate, median over four Japanese writing prompts,
  one request at a time.
- **Quality**: paired NLL of the same Japanese prose text, cut into 24 pieces of 256 tokens,
  against a launch without the change. A change is kept when its NLL lies within the spread of
  repeated launches. A change that only reorders sums is also checked bit for bit where it can be.
- **Prefill**: wall time of one long prompt with one output token. Random token ids route
  over more experts than text, so they come out slower (about 6% at 35K).

## Decode

Where the time goes (1017, nsys, before the CPU experts' decode reads below): a token took
22.5 ms. The first GPU was busy for 13.0 ms of it and waited 9.0 ms for the CPU experts. On
one socket alone (`tools/kt-mxfp4-decode-bw.py`, 32 threads, 4 experts a call) the CPU kernel
read 75 GB/s of the 118 GB/s the socket streams, 83 GB/s after the decode reads.

| change | before → after | release |
|---|---|---|
| Engram rows read through a 16-thread I/O pool instead of two serial `pread`s per miss (`engram-adapter`) | a fetch 4 → 1 ms; decode +8% | earlier |
| The dense FP8 GEMMs at one row: split-K tiling instead of Triton's default (`SGLANG_FP8_SKINNY`) | 38 → 8.7 ms a token; decode +40% | earlier |
| (both together) | 11-12 → 18 tok/s | earlier |
| The hottest experts of each layer resident on the GPU, from a recorded routing (`placements/`, the wrapper's id remap) | CPU 700 → 381 µs a layer; 18 → 23.5 tok/s | earlier |
| kt-kernel's hand-over through CUDA stream memory operations and a spinning poller (`KT_STREAM_MEMOPS`) | 25.4-26.1 → 27.0 tok/s | earlier |
| Hot lend: more hot experts in the KV pool's free pages outside prefill (`KT_KV_HOT_LEND`) | 28.95 → 29.98 tok/s (3-4 more a layer) | earlier |
| … and in the driver's idle memory above a 1 GiB floor (`KT_KV_HOT_SPARE_FLOOR_MB`) | 31.73 → 34.43 tok/s | 1017 |
| The vision tower in host memory between images (`SGLANG_DSV4_VISION_ON_DEMAND`) | 925.7 MiB a GPU back; hot slots 394 → 501 | 1017 |
| The CPU's MXFP4 scales as one-byte codes instead of fp32 (`KT_MXFP4_SCALE_CODES`) | 34.81 → 36.06 tok/s, output bit-identical | 1017 |
| The engram row lookups start with the token on a side stream, not at layers 1 and 14 (`SGLANG_DSV41_ENGRAM_PREFETCH_ROWS`) | 0.7 ms a layer of GPU idle gone; 35.9 → 37.4 tok/s, values unchanged | 1017 |
| Decode's block-FP8 GEMMs through a GEMV stepping K by 512 instead of the 32-value scale group (`SGLANG_FP8_GEMV`) | `wq_b` 78.9 → 33.7 µs, engram `wkv` 552 → 208 µs; 37.4 → 44.2 tok/s | 1017 |
| The CPU experts' activation conversion with PSHUFB and a weight prefetch 1 KB ahead (`KT_MXFP4_DEC_VCONV`, `KT_MXFP4_DEC_PFS`) | a call 502 → 455 µs, output bit-identical; 44.72 → 46.01 tok/s | 1017 |

The GPU experts run on Triton kernels in the checkpoint's own MXFP4 layout
(`SGLANG_MXFP4_MOE_FP32`), rounding to bf16 once, at the weighted sum. Marlin rounded seven
times on the way. Decode speed did not change; the NLL went 3.1478 → 3.1422 and
repeats to the last digit from launch to launch (1017).

## Prefill

| change | before → after | release |
|---|---|---|
| kt-kernel's MXFP4 GEMM permutes an expert's activations once, not on every call | 2,000 tokens 58.8 → 27.0 s, output identical | earlier |
| CPU experts streamed to the GPU per chunk and run there in bf16 (`KT_GPU_STREAM_PREFILL`, `KT_GPU_STREAM_W16`) | about 74 → 270 tok/s at 2,048-token chunks | earlier |
| Prefill lend: the decoder's dense weights leave the GPU while a prompt runs, so the chunk can widen (`KT_PREFILL_LEND`) | 8K 257 → 602, 38K 255 → 636, 114K 259 → 594 tok/s | earlier |
| The indexer scores 512 query rows at a time, with block masks, instead of one matrix growing with the prompt | the peak flat from 38K to 510K (3.3 GB); a 1M prompt fits | earlier |
| KV pool on remappable memory; a chunk borrows the pool's free pages; the chunk planned per request (`KT_KV_VMM`, `KT_PREFILL_KV_LEND`, `KT_PREFILL_PLAN`) | 35K 57.5 → 51.3 s (6 → 4 chunks), KV pool unchanged | earlier |
| The plan charges the sliding-window pool for the chunks in flight, not the rest of the prompt | 512K 256 → 611 tok/s | 1017 |
| The GPU-resident experts of a chunk on the tensor cores, live routes only (`SGLANG_MXFP4_MOE_FP32`) | a chunk's resident call 64.0 → 15.7 ms; 35K +4-6% | 1017 |
| A prefill chunk's engram rows read before it runs: while chunk k runs, the scheduler hashes chunk k+1's tokens on the CPU and each rank's row store reads the missing rows into its RAM cache once chunk k's lookups are done (`DSV41_ENGRAM_WARM`); the later engram layer's lookup starts with the forward (`SGLANG_DSV41_ENGRAM_PREFETCH_PREFILL`) | lookups after the first chunk 2.1 s → 0.04 s (0.9% misses); 35K 1,011 → 1,306, 131K 1,009 → 1,450 tok/s; log-probs identical | 1017 |
| The CPU experts decode each weight group once for 2 to 8 rows on an expert, not once a row (`KT_MXFP4_SMALL_M`) | 3 rows on an expert 1,064 → 719 µs; 35K +0.7%; rows byte-identical to computed alone | 1017 |
| The engram scales of each rank's rows held in RAM (1.4 GiB a table and rank), so a missed row is one NVMe read instead of two (`DSV41_ENGRAM_SCALES_RAM`) | a 12K-token chunk's lookups 4.0 / 6.7 → 2.1 / 2.1 s (layers 1 / 14); 35K 655 → 1011, 131K 652 → 989 tok/s, rows byte-identical | 1017 |

None of these takes tokens from the KV pool. The 1M-token pool comes first.

## Launch

| change | before → after | release |
|---|---|---|
| The host's shmem THP at `advise`, and the checkpoint's page cache dropped before each expert layer loads (`KT_GPU_STREAM_DROP_CACHE`) | pinning the expert arenas: 60+ → 11 min | earlier |
| The expert shards read ahead by parallel `pread`s (`patches/kt-kernel-load-prefetch.py`); the CPU experts skipped in SGLang's loader; the dense shards read in parallel | ready 740 → 243 s | earlier |

## Tried and not kept

- **Deferred experts** (`KT_DEFERRED_EXPERTS`): letting the GPU run ahead of the CPU gave 11.9 →
  19.1 tok/s, but at 6 the output fell apart (wrong facts, Chinese text, loops). On
  V4-Flash-Vision, 4 already made decode drift from the prefill. Kept at 0.
- **More resident experts**: 10 a layer instead of 5 halves the KV pool (559K tokens). Kept at
  5; the hot lend adds experts only in memory nothing else is using.
- **A wider lend chunk**: above 6,144 rows the sliding-window pool grows into the KV pool
  (10,240: 896,000 tokens). Kept at 6,144 by default.
- **A placement from 1017's own routing**: recorded on 3,300 tokens of one topic, it left more
  routes on the CPU on other prose (decode 36.06 → 35.56). The earlier release's placement
  stays. A small static sample is the wrong tool; an adaptive cache is the next step.
- **The CPU kernel's decode, other ways** (one socket, 4 experts a call): whole weight rows
  prefetched at once (down -13 µs, gate/up +12 µs), the activation's conversion done once per
  expert on one thread (502 → 545 µs), row slices sized to fill whole rounds of the threads
  (gate/up 278 → 299 µs).
- **fp32 intermediates in the CPU experts**: no NLL change; the per-expert rounding is not what
  moves the NLL any more.

## SGLang main's decoder bounded replay

On the earlier base the third request of a session came back as garbage with it, and it was
kept off. SGLang main turns it on by default (the same scheme as DeepSeek's DeepInfer, "YoCo
prefill truncation"): the layers from 20 on run only over the prompt's last 128 rows. Three
turns of one chat with the prefix cache on (`DSV41_RADIX=1`, 512 tokens read from the cache in
turns 2 and 3) come back intact. A 131,072-token prompt takes 242.7 s with it and 266.0 s
without (`--no-enable-decoder-swa-bounded-replay`).

## Open

- Launches that should give the same output differ in the long NLL by up to 0.013 (3.1405 to
  3.1536 over ten single-patch launches in [tier1.md](tier1.md), none with the hot lend),
  while the 2-token NLL is the same in all of them. Short prefills repeat; the difference
  comes in somewhere on the longer pieces' path. The cause is not found.
- 9 ms of a decode token is the GPU waiting for the CPU experts.
