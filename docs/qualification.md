# Qualification

`MODEL=<dir> tools/qualify.sh` writes a report for the machine it runs on, as one Markdown file.
It checks an answer, tool calls, the NLL against the reference implementation's scores, and the prefix cache.
It also measures decode on seven kinds of text and the 35K-token prefill.
It takes about 15 minutes after the server is up.
The report names the hardware and software, and nothing that identifies the machine.
The [Results](results.md) come from one machine, so reports from other GPUs and CPUs are welcome.
Filed reports are listed by tag in [reports/](../reports/README.md), with how to send one.

## Tests

The CI job "patches apply" runs every test that needs no GPU, on each push and once a week.
The Triton kernels run there in Triton's interpreter on the CPU.

| test | needs | where it runs |
|---|---|---|
| `tools/test_sm89_dispatch.py` | the patched SGLang tree | CI |
| `tools/test_qualify_cache.py` | nothing | CI |
| `tools/lend-transition-fences-test.py` | nothing | CI |
| `tools/hot-expert-spare-lending-test.py` | git | CI |
| `tools/e4m3-emulation-test.py` | torch | CI |
| `tools/test_live_topk.py` | torch and Triton; `--gpu` also checks on a GPU | CI on the CPU, by hand on a GPU |
| `tools/test_merge_sink.py` | the same, with `sglang-optional/01` applied | CI on the CPU, by hand on a GPU |
| `tools/dsv4-o-fp8-qdq-test.py` | a GPU | `setup.sh`, on the machine it builds |
| `tools/test_fused_merge_wiring.py` | a GPU | by hand |
| `tools/mxfp4-fp32-moe-test.py` | a GPU | by hand |
| `tools/sm89-attention-test.py` | an SM89 GPU | by hand ([sm89.md](sm89.md#checking-an-sm89-card)) |
| `tools/dsv41-encoding-check.py`, `tools/dsv41-recipe-check.py` | the checkpoint's `encoding/` and tokenizer | by hand |

## Multi-session prefix-cache qualification

`tools/qualify-cache.py` generates synthetic evidence histories from a seed and freezes
both messages and token IDs. Run it against an isolated server with four request slots:

```
python tools/qualify-cache.py --prepare-only --seed 7812 --output prepared
python tools/qualify-cache.py --fixtures prepared/fixtures.json --output cache-on \
  --require-host-hit --flush-cache
```

Use the same fixtures for each configuration. The default four histories fit a 1M-token
device pool; a separate 200K-token history forces eviction. For another pool size, set
`--lengths` and `--pressure-tokens` while preparing: the four histories should fit together
with room for generated tokens, but exceed the pool once the pressure history is added.
The initial replay checks allow only a small recomputed tail (`--replay-tail-tokens`,
default 512). They will fail if the initial histories already exceed a device-only cache.

Each session has independent case, release, document and alert facts. The protocol checks
fresh and concurrent cached answers, a superseding tool result, four overlapping fixed-length
generations, post-decode answers and recovery after fresh eviction pressure. It finishes
with a synthetic tool-call/result exchange; generated calls are never executed. Reasoning
is enabled by default. `--flush-cache` additionally tests cache clearing after decode,
followed by that short tool exchange. It clears the server's entire prefix cache.

The output directory contains fixtures and their hashes, allowlisted serving configuration,
complete responses, token-arrival times, cache-hit details and a summary. First-token latency
and first-to-last-token decode throughput are reported separately. Wrong or stale facts,
retractions, incomplete streams and non-overlapping concurrent generations fail the run.
`--require-host-hit` additionally requires observed host-cache reuse; omit it for a
device-only control. These synthetic checks qualify the exercised cache paths, not general
model quality or arbitrary workloads. API authentication uses `SGLANG_API_KEY` when set.
Server snapshots omit credentials and arbitrary nested configuration.

CPU-only checks for the generator and response validation:

```
python -m unittest discover -s tools -p test_qualify_cache.py
```
