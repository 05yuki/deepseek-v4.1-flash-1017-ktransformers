# Results

2 × AMD EPYC 7452 (Zen 2, AVX2), 512 GB DDR4, 2 × RTX 5070 Ti 16 GB, the checkpoint on
NVMe; built from this repository and run with the `serve.sh` defaults (1M-token context,
one request at a time), measured with `tools/bench.py` at tag `2026-10-29-2123` (SGLang main 5d8e90a0), every table from one launch.

Decode, 1,024 tokens a reply (`--other-text` for the rows below the first):

| text | tok/s |
|---|---|
| Japanese prose (median of four prompts) | 50.0 |
| Chinese prose | 44.2 |
| math | 42.4 |
| English prose | 41.1 |
| JSON | 39.0 |
| Python code | 37.4 |
| English technical prose | 37.3 |

One expert a layer always stays on the GPU, and as many more as the free GPU memory holds
sit in the KV pool's free pages and the driver's idle memory (15-16 a layer here). All of
them are swapped for the experts the text being written routes to most, so no routing has
to be recorded for a new kind of text. Code, JSON, math and English spread
their routes over more experts than Japanese prose (a layer's 21 most used take about half
of their routes, against 70-75% in Japanese prose), so the CPU computes more of them: a GPU
with more free memory holds more of them by itself
([tier1.md](tier1.md#16-hot-lend)).

Prefill, one prompt of random token ids:

| prompt (tokens) | tok/s | time |
|---|---|---|
| 35,000 | 1,736 | 20 s |
| 131,072 | 1,872 | 1.2 min |
| 524,288 | 1,460 | 6.0 min |
| 1,000,000 | 1,068 | 16 min |

Longer prompts run in smaller chunks: the prompt's own KV and the attention's working
memory grow with its length, leaving less GPU memory for a chunk, and every chunk streams
the CPU experts to the GPU once. The prompts are random token ids, which spread over more
experts than real text, so these figures are lower than with text.

## One GPU

The same machine with `CUDA_VISIBLE_DEVICES=0`: one RTX 5070 Ti 16 GB (tensor parallel 1),
the CPU experts on both sockets, `serve.sh` defaults, measured at tag `2026-10-29-2123`,
every row from one launch.

The KV pool covers the whole 1M-token context. Its compressed KV, 0.70 GiB, sits in host
memory ([The KV pool past the GPU](kv-cache.md#the-kv-pool-past-the-gpu)). The hot lend holds 1-2 more
experts a layer beside the one that stays, in the driver's idle memory, so every text runs
most of its routes on the CPU.

| text | tok/s |
|---|---|
| Japanese prose (median of four prompts) | 33.4 |
| English prose | 31.0 |
| math | 30.5 |
| Chinese prose | 30.2 |
| English technical prose | 29.9 |
| JSON | 29.9 |
| Python code | 29.4 |

| prompt (tokens) | tok/s | time |
|---|---|---|
| 35,000 | 1,052 | 33 s |
| 65,536 | 1,285 | 51 s |
| 1,000,000 | 807 | 21 min |

## One socket, 256 GB

The same machine held to one socket: `numactl -N 0 -m 0` (one EPYC 7452 and its 251 GiB),
`CUDA_VISIBLE_DEVICES=0`, `KT_THREADPOOL_COUNT=1`, otherwise the `serve.sh` defaults,
measured at tag `2026-10-30-1430`, every row from one launch.

`serve.sh` kept 201 GiB of the experts in a RAM cache (of 269 GiB) and read the others from
the checkpoint, an NVMe drive of about 6.5 GB/s ([tier1.md](tier1.md#20-expert-cache)).
Node 0's free memory never fell below 11 GiB, the 1M-token prompt included. The 24-piece NLL is
0.7209.

| text | tok/s |
|---|---|
| Japanese prose (median of four prompts) | 20.2 |
| Chinese prose | 17.9 |
| English prose | 16.7 |
| math | 16.6 |
| JSON | 15.8 |
| English technical prose | 15.1 |
| Python code | 15.0 |

| prompt (tokens) | tok/s | time |
|---|---|---|
| 2,000 | 147 | 14 s |
| 35,000 | 697 | 50 s |
| 131,072 | 863 | 2.5 min |
| 1,000,000 | 625 | 27 min |

Decode is bound by one socket's memory bandwidth. A prompt touches most experts of every layer,
so a prefill reads the ones the cache does not hold from the drive: short prompts are bound by it.

# Accuracy

The served model is scored against the checkpoint's own reference implementation:
DeepSeek's `inference/model.py`, run one layer at a time on one 16 GB GPU
([tools/ref-nll.py](../tools/ref-nll.py)). Both score the same 24 pieces of 480 tokens, taken
from two public-domain texts in [tools/nll-text/](../tools/nll-text/), one Japanese and one
English, teacher-forced. `serve.sh` defaults, both at tag `2026-10-28-1714`:

| run | mean NLL | minus the reference (SE over the 24 pieces) |
|---|---|---|
| reference (`inference/model.py`) | 0.7185 | |
| served, two GPUs | 0.7192 | +0.0007 (0.0041) |
| served, one GPU (KV past the GPU in host memory) | 0.7224 | +0.0039 (0.0040) |
| served, one RTX 4080 SUPER 32 GB (SM89), against its own reference run (0.7141) | 0.7151 | +0.0010 (0.0041) |

The texts are well known, so the model predicts them closely and the NLL is low. They
serve to compare the two implementations on the same tokens, not to measure the model.

To repeat it:

```
python tools/ref-nll.py <checkpoint dir> ref.json      # about 4 minutes on one 16 GB GPU, no server
MODEL=<checkpoint dir> ./serve.sh --background
python tools/nll.py score <checkpoint dir> served.json
python tools/nll.py compare ref.json served.json
```

`tools/nll-ref-1017.json` holds the reference run's scores on an RTX 5070 Ti; `tools/qualify.sh` compares against them.
