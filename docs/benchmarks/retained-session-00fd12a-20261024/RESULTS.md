# Retained-session admission: reviewed A5 evidence

**Reviewed experimental evidence — required collection and independent review complete.** The default-off admission/ownership fix was functionally validated on the exact A5 profile. A5 numerical/source/resource review and the fixed-history reference review are complete. Material numerical differences remain disclosed; numerical equivalence, general readiness and isolated speedup are not established. This is evidence for experimental source review, not a production-readiness certification.

When idle streaming sessions hold every request-pool row, an ordinary request can remain queued despite no active generation. The optional admission change can reclaim the oldest idle session row, protects in-flight sessions, and preserves an evicted session's token history for re-prefill. It also stops an overlapping-turn rejection from clearing ownership belonging to the session's active turn. Upstream already provides concurrent scheduling, streaming sessions, queues and cache ownership; this patch fixes those specific admission and ownership cases.

Setup gate `SESSION_CONCURRENCY_PATCH` and runtime gate `KT_STREAMING_SESSION_ADMISSION` both default off. The publication-candidate implementation is `b01f440cf93f5952962ba6a4b2b0ea0d614db9d8`, rebased on recipe `e8608a2` / SGLang `8b9f724`; its source patch SHA256 is `9f88a710ef5857cf8f9b53d004b6290e06dcbda7db00cac94bc99e56793f8a09` and reviewed standalone profile SHA256 is `c6780ef0ed01654e09746adf37f781c32a6f1e28bea9f681650708ab06d658ac`. The admission patch is unchanged from the measured older-pin source. These bindings do not transfer the model results to the newer pin.

## Exact measured configuration

Recipe `00fd12a0b3e59cbf7dcc62b7341415fcdd3fc9a6`, SGLang `9c0f6d6a610531bd16ad3513572e55226450b244`, kt-kernel `95009ea6856c0799e517e93cb12be5e8494bc7ce`, DeepSeek-V4.1-Flash-1017 MXFP4 checkpoint, one **experimental 32 GiB NVIDIA GeForce RTX 4080 SUPER (reported 32,760 MiB)** on **dual AMD EPYC 7543**. The owner uses all 64 CPUs with interleave across NUMA nodes 0–7; native experts use 56 threads/eight pools with the recorded strict-bind worker policy. Model scope cap 440 GiB, swap disabled; TP=PP=1.

The session profile enables admission and streaming sessions, four running/prefill requests, four queued requests, FCFS, stream interval 1, decode graph batches `[1,2,4]`, 1,048,576 configured context/total-token budget and chunked prefill 36,864. It disables general radix caching, HiCache and cache TTL. SWA lending is 0 and hot spare floor 3072 MiB. The staged source physically includes the KV-policy patch with its TTL/HiCache policies disabled, alongside the session patch; this is the A arm without B's hot/glue treatment. It does not establish that four 1M-token sessions fit, or that this capacity fits another machine.

Exact profile/launch/preparation, native/build identities and source hashes are bound in [SUMMARY.json](SUMMARY.json). A5 reports the accelerator model; CPU model/card capacity are corroborated by the same-host canonical hardware record, with matching A5/canonical boot identity. Their provenance is recorded in the summary; no fresh hardware query was made. Runtime conclusions remain tied to 00fd12a/9c0f6d6; newer-main source/application proofs do not transfer this model result to another pin.

## Functional results and observed wall time

N1/N2/N4 each completed three retained-session turns with inherited cache counts. With four idle held rows, an ordinary five-token request completed, exactly one session row was evicted, and all four sessions resumed with their committed histories: one re-prefilled 551 tokens; the other three reported 538 cached of 551 prompt tokens. Both overlapping turns returned HTTP 400 while the active turn retained ownership. Active abort recovered from committed history; queued abort generated zero tokens and its retry succeeded. Queue checks observed four running/four queued, HTTP 503 on overflow, first-token order `Q0,Q1,Q2,Q3`, cancellation and a drained exit. These check the existing upstream queue under the patched ownership behavior.

All cohort requests below generate exactly **one reply token per session**. Prompt/cache counts are per request; cohort wall includes client launch/join overhead. Turn 0 submits 512 IDs; turns 1/2 append 12 IDs after the preceding reply. Values are seconds from this correctness run, with exact values and all 44 request summaries in [PROTOCOL.json](PROTOCOL.json).

| Sessions | Turn | Full prompt / cached tokens per request | Reply tokens per request | Cohort wall s |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 512 / 0 | 1 | 7.933296 |
| 1 | 1 | 525 / 512 | 1 | 6.997620 |
| 1 | 2 | 538 / 525 | 1 | 2.640331 |
| 2 | 0 | 512 / 0 | 1 | 15.206022 |
| 2 | 1 | 525 / 512 | 1 | 3.080692 |
| 2 | 2 | 538 / 525 | 1 | 0.523293 |
| 4 | 0 | 512 / 0 | 1 | 23.786591 |
| 4 | 1 | 525 / 512 | 1 | 3.730035 |
| 4 | 2 | 538 / 525 | 1 | 1.238532 |

The fifth ordinary request was 5 prompt / 1 reply / 0 cached, wall 0.606482 s. After eviction, the four resumed requests were 551 prompt / 1 reply each, cached `[0,538,538,538]`, individual walls `[9.643727,9.641418,9.645624,9.010748]`s. No throughput or speedup is inferred: canonical A used different request/graph/memory settings, and no matched performance control was run.

## Numerical observations remain material

All seven retained/cold replay comparisons reconstructed identical full input IDs and matched common request options within each pair. They cover six distinct histories. **One of seven selected tokens changed**, and unchanged winners do not imply matching distributions. The largest probability differences below compare tokens present in both reported top fives; they are not a full-vocabulary distance. Full top-five probabilities/logprobs and exact input hashes are preserved in [REPLAY.json](REPLAY.json).

| Scenario | Prompt / retained cached tokens | Retained → cold winner | Largest observed common-token change, percentage points |
| --- | ---: | ---: | ---: |
| N1/s0 | 538 / 525 | 1 → 1 | 28.7073 |
| N2/s0 | 538 / 525 | 1 → 1 | 24.5723 |
| N2/s1 | 538 / 525 | 1162 → 1162 | 6.2428 |
| N4/s0, evicted | 551 / 0 | 369 → 369 | 1.5493 |
| N4/s1 | 551 / 538 | 1 → 42893 | 40.1387 |
| N4/s2 | 551 / 538 | 1162 → 1162 | 0.2325 |
| N4/s3 | 551 / 538 | 1162 → 1162 | 0.1414 |

For N4/s1, P(1) changes 0.429725→0.131662 and P(42893) changes 0.058157→0.459544. N1 keeps winner 1 while P(1) changes 0.614273→0.327200. A separate short 18-token abort-recovery/cold pair also keeps winner 1, with P(1) changing 0.227625→0.731864 (50.4239 pp), despite both requests reporting zero cached tokens. That short witness is separate from the seven cohort rows and remains visible.

Paired NLL uses the same 24 exact 480-token pieces, scoring 11,496 positions; WikiText uses 12 exact 1024-token chunks, scoring 12,276 positions. Values are nats/token. Positive delta means A5 has higher NLL. SE is the sample standard deviation of piece deltas divided by sqrt(piece count); it is descriptive, with no acceptance threshold selected from these results. [NUMERICAL.json](NUMERICAL.json) retains every piece delta, alignment/score receipt hashes and the original analyzer identity.

| Comparison | A5 mean NLL | A5 minus comparator | Piece SE |
| --- | ---: | ---: | ---: |
| Checkpoint reference → A5, 24 pieces | 0.7240040158 | +0.0098927643 | 0.0050181971 |
| Own canonical A → A5, 24 pieces | 0.7240040158 | +0.0089076990 | 0.0048845357 |
| Own canonical A WikiText → A5, 12 pieces | 0.9012321734 | +0.0027865522 | 0.0021175282 |

Source/control review found inherited numerical differences among CPU, resident, hot/streamed expert and independent checkpoint-reference paths, including different rounding/accumulation boundaries. The short witness also changes hot-expert placement after a flush; zero attention-cache reuse does not prove identical expert execution. These findings motivate focused controls, but establish neither the cause of the observed differences nor which result is correct. The completed exact-history full-prefill reference is reported below. No concrete new admission or committed-history-reset defect was identified in the reviewed source/functional paths; this does not prove absence of every cache/state/asynchronous defect. Numerical identity, general quality equivalence and production readiness are not established.

## Completed fixed-history reference

The authenticated independent reference completed all 40 layers and 43,560 fallback projection reads. Its 14-member terminal archive, exact source/storage identities, actual inputs, full scores, arithmetic and empty/removed cleanup passed root and independent review. The source package/reference main were unchanged. The 64 GiB/Swap0 scope had no high/OOM/OOM-kill/swap; 9,101 memory.max events were loading/file-cache reclaim under the existing loading policy, not a performance result.

[FIXED_HISTORY_REFERENCE.json](FIXED_HISTORY_REFERENCE.json) contains the exact 538-ID H, candidates `[1,455]`, input hashes, all 1,076 scores, serving top fives, provenance and confounds. H matches the reconstructed A5/B4 N1 and N2/s0 histories, compact hash `7a7466ea9a980bbd514717127443e694a6902f25ac678cc72fea32cae97eb685`. Two pieces H+[1], H+[455] each return 538 scores. Their 537 common-prefix target scores are identical. The final score predicts appended candidate x[538] from h[537], giving logP(candidate|H).

| Same exact history | Cached tokens | P(1) | P(455) | Log odds 1/455 | Odds 1/455 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full-prefill checkpoint reference | — | 0.30569335 | 0.03321938 | 2.21944904 | 9.202259 |
| A5 N1 retained | 525 | 0.61427283 | 0.01197641 | 3.93750003 | 51.290217 |
| A5 N1 cold | 0 | 0.32720004 | 0.12037015 | 1.00000012 | 2.718282 |
| B4 N1 retained | 525 | 0.46968248 | 0.06356461 | 2.00000006 | 7.389057 |
| B4 N1 cold | 0 | 0.05824785 | 0.43039663 | -2.00000000 | 0.135335 |
| B4 ordinary radix hot | 256 | 0.33496918 | 0.13963593 | 0.87500000 | 2.398875 |

The reference **favors 1 over 455**: logP(1)=-1.1851727962, logP(455)=-3.4046218395. This agrees with A5's pair-preference direction and B4 retained/hot, while B4 cold reverses it. It does not measure a full-vocabulary reference argmax. A5 cold P(1) is 2.1507 percentage points above reference, but P(455) is 8.7151 points above and its odds are 2.718 versus 9.202; it must not be described as reference-equivalent. Repeated N1/N2 cold top fives are identical within each arm. A5 N2 retained 455 is absent from its top five and remains unknown; no ordinary-radix control was recorded for A5.

This is a full-prefill diagnostic with the disclosed inherited reference/serving arithmetic differences. It narrows this witness without proving cause or establishing incremental-session equivalence. It does not resolve every replay or the separate 18-token uncached recovery variation. Those limits are part of this completed review, not an implicit acceptance threshold.

## Prior failure, resource and evidence limits

A4 is preserved as a **failed** profile: after three N1 retained turns and a successful close, flush failed while upstream SWA clear tried to remap pages before hot VMM loans were released. A5 changes the session hardware policy from SWA lend 1 / spare 1024 MiB to SWA lend 0 / spare 3072 MiB. The admission/lender source and flush/cold controls were not changed to erase the failure. The profile workaround is not an upstream ownership fix.

A5's combined numerical/features client window lasted 446.496 s. Sampled scope charge peaked 424.880 GiB; swap was zero, memory event counters did not increase, and PSI increased 391 us some / 390 us full. The 440 GiB lifetime peak and 3727 max events include loading; they are not a client-only RSS measurement. Cleanup preserves the initial 15-second graceful-stop timeout, followed by successful owned KILL/drain and an empty, removed scope. One earlier raw module-observer receipt was superseded; its embedded inventory remains archive-bound; 36/37 earlier raw receipts and 5/5 terminal raw receipts match. No source change is inferred from that recording limitation.

The complete selected evidence remains preserved outside these compact projections: archive SHA256 `cd039258edeaf64def53fb6aeebe861676cef259f257d378961f11bb0e051b02`, manifest `22fff81601d97ae4231f2a0edbde527a3b61962e628a203c2fc68481234f08cc`, RUN `73dc92a954b2f727c23d32f913087665ce12c6582448e78095d728a4cec08a91`. This directory carries compact numeric/protocol projections rather than the 35 MB report. Single-run short fixtures, observed probability shifts, bounded reference scope and absence of a production-readiness claim remain explicit limits.
