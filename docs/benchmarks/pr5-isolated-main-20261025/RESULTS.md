# PR5 isolated logprob identity check

All four combinations of `KT_SHARED_AFTER_SUBMIT` and `KT_HANDOFF_PACK` produced
bit-identical logprobs in this test. All six pairwise comparisons have zero
differences in selected-token float32 bits, top-8 float32 bits, token identities,
or top-8 ordering. No tolerance or NLL-within-SE criterion is used.

This addresses item 1 of [05yuki's review](https://github.com/05yuki/deepseek-v4.1-flash-1017-ktransformers/pull/5#issuecomment-6378935104).

## Tested revisions and configuration

- Main: `e8608a224c108c06c8dbe636bf0bfef295e82161`.
- Exact PR5 implementation: `b930ba3199e4ce565a28c1a1d693e97374726e14`.
- SGLang: `8b9f72417fa85a04de6193b0a52f4cec63370d77`.
- Ktransformers: `95009ea6856c0799e517e93cb12be5e8494bc7ce`; native binary rebuilt from main.
- One experimental 32 GiB RTX 4080 SUPER (SM89), TP1, eight NUMA pools and 56 expert workers.
- Main plus PR5 only: no canonical-math/KV-policy, retained-session, PR7, PR8, poll-hint or spin-time changes.
- Four request rows; decode graph batches 1/2/4; radix cache disabled; streamed-prefill threshold 512.
- All 211 main runtime requirements and the real attention imports were checked. Missing/new main dependencies were supplied privately: `sgl-flash-mla==1.0.0`, `sgl-deep-gemm==0.2.3.post2`, `tml-renderers==0.1.0`.

## Hot slots were held fixed

Every arm used the same expert-placement file (SHA256
`f0db7c97f5a7ef50487e5e77a3b2ebe4b115118d5f4793f31a4391704c89e9e6`),
`KT_KV_HOT_EXTRA=2`, `KT_KV_HOT_SLOTS_MAX=80`,
`KT_KV_HOT_DYNAMIC_SLOTS=0`, and `KT_KV_HOT_DYNAMIC_RESIDENT=0`.
The saved live-process environment matches these settings.

Each runtime log records exactly one initial allocation:
`80 slots (2-2 a layer of 2)`. No slot release, dynamic startup, or dynamic
round/remapping was observed. The tested static route uses the fixed placement's
`[N, N+k)` physical expert range; the dynamic route table and swapping path are
disabled. The 480-token prompts and 64-token batches stay below the 512-token
streaming threshold, avoiding prefill-lend transitions during these comparisons.
See [the per-arm witnesses](fixed_hot_receipts.json).

## Results

| Shared after submit | Handoff pack | Cases / requests | Exact versus both off | Repeat exact |
| --- | --- | --- | --- | --- |
| 0 | 0 | 28 / 32 | Reference | Yes |
| 1 | 0 | 28 / 32 | Yes | Yes |
| 0 | 1 | 28 / 32 | Yes | Yes |
| 1 | 1 | 28 / 32 | Yes | Yes |

Each arm scored 24 pinned multilingual pieces of 480 tokens, batches of 1/2/4
64-token prompts, and a repeat of piece 00. Every request generated eight tokens
with selected-token and top-8 input/output logprobs. Each arm covers 12,416 scored
input positions and 256 output positions: 50,688 selected scored positions across
112 cases and 128 requests. Every response reported zero cached tokens.

The larger prompts cover the packer's row-cap fallback; small prefills and decode
exercise the enabled packing path. Packing did not permanently disable itself.
Separate GPU fixtures also passed batch sizes 1/2/4/256 with int32/int64 IDs,
three graph replays, row-cap refusal at 257, and unpinned fallback.

All model runs used a 440 GiB/no-swap guard and recorded zero OOM or swap events.
The graceful-stop deadline expired during teardown; the existing fallback killed
the owned scope and verified its removal. Final GPU and port checks were idle,
and the shared installed native binary remained unchanged.

## Evidence and offline verification

Run with Python's standard library:

```sh
python docs/benchmarks/pr5-isolated-main-20261025/verify.py
```

[observations.json.gz](observations.json.gz) contains the complete recorded
numerical arrays for all four arms: float32 values encoded as eight-digit hex,
token IDs, and top-8 ordering. The verifier checks its digest, every original
per-case numerical digest, coverage, finiteness, repeats, fixed-hot witnesses,
and all six pairwise comparisons. It verifies saved observations, not a new model
execution. Full raw-response/source/log archives were independently hash-checked
locally; their identities and the guarded-run receipts are recorded in
[FINAL_REVIEW.json](FINAL_REVIEW.json). [MATRIX_RESULT.json](MATRIX_RESULT.json)
records the complete comparison verdict.

These are isolated correctness observations on this pinned SM89 configuration.
They make no new throughput claim and do not validate TP2/SM120, dynamic-hot,
broader KV/session equivalence, or PR6. The test-results commit changes only this
documentation and evidence; the tested implementation remains `b930ba3`.
