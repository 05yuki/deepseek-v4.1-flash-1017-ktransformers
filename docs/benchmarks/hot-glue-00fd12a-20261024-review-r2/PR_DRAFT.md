# Observed A/B results: optional hot slots and CPU/GPU MoE overlap

The owner accepted this exact completed workload for an observed-comparison draft and bounded feature trials. This is not a numerical-equivalence or sustained-speedup finding. The public `REVIEW.json` binds that decision to the original generated summary and source receipts.

A is upstream recipe `00fd12a` plus SM89 enablement and common host/storage adapters. B adds the optional GPU-glue/hot-slot bundle, restricts service threads to CPUs `7,15,23,31,39,47,55,63`, and changes queue spin from 300 to 2,000 microseconds. Both retain upstream automatic hot lending and strict placement of 56 expert workers on eight NUMA nodes.

One experimental 32 GiB RTX 4080 SUPER, two EPYC 7543 CPUs, TP=1, context/pool cap 1,048,576, max-running=1, radix off. The card is not the stock 16 GiB configuration.

| Decode workload | A tok/s | B tok/s | Rate change | A/B output tokens |
|---|---:|---:|---:|---:|
| p0 | 31.6 | 33.5 | +6.0% | 1024/1024 |
| p1 | 31.1 | 34.9 | +12.2% | 1024/1024 |
| p2 | 29.3 | 35.1 | +19.8% | 1024/1024 |
| p3 | 30.8 | 35.3 | +14.6% | 1024/1024 |
| Chinese prose | 27.2 | 32.2 | +18.4% | 1024/1024 |
| English prose | 28.1 | 32.4 | +15.3% | 1024/1024 |
| English technical prose | 24.7 | 30.6 | +23.9% | 1024/1024 |
| math | 27.2 | 32.5 | +19.5% | 1024/1024 |
| JSON | 26.5 | 32.3 | +21.9% | 1024/1024 |
| Python code | 25.6 | 31.0 | +21.1% | 1024/1024 |

Observed default p0-p3 median: 31.0 to 35.0 tok/s (+12.9%).

| 35K prefill | A seconds | B seconds | Latency change |
|---|---:|---:|---:|
| first | 41.63 | 42.08 | +1.1% |
| second | 29.65 | 29.77 | +0.4% |

| Numerical comparison | NLL delta, nats/token | Piece/chunk SE |
|---|---:|---:|
| A minus checkpoint | +0.000985065 | 0.004105693 |
| B minus checkpoint | +0.003621167 | 0.003984346 |
| B minus A, 24 reference pieces | +0.002636102 | 0.001694099 |
| B minus A, 12 WikiText chunks | +0.000788344 | 0.001037606 |

All 24 paired piece deltas and 12 chunk deltas are retained in `summary.json`; raw reference and served arrays are included. No tolerance or equivalence verdict is inferred.

| Diagnostic | A | B |
|---|---|---|
| Exact upstream check.sh | PASS | PASS |
| Component groups | 10 | 10 |
| Tool calls | 5/5 | 5/5 |
| WikiText NLL | 0.898445621 | 0.899233965 |
| Retrieval | PASS | PASS |
| Client memory event deltas | {"high": 0, "low": 0, "max": 0, "oom": 0, "oom_group_kill": 0, "oom_kill": 0, "sock_throttled": 0} | {"high": 0, "low": 0, "max": 0, "oom": 0, "oom_group_kill": 0, "oom_kill": 0, "sock_throttled": 0} |
| Client memory PSI some/full, seconds | 0.000000 / 0.000000 | 1.624029 / 1.623210 |
| Client memory PSI some/full, percent of window | 0.000000% / 0.000000% | 0.216662% / 0.216553% |
| Client window seconds | 802.783677 | 749.566983 |
| Maximum observed client swap bytes | 0 | 0 |

Build gate `HOT_GLUE_PATCH=1` and runtime gate `KT_GPU_GLUE=1` are opt-in. The separate hardware profile adds the full measured hot-slot and affinity treatment. The 512 MiB value limits replacement copying per rebalance round; the spare floor is 1,024 MiB.

## Limitations

- One ordered A then B campaign, one request at a time. No repeated-run uncertainty or universal speedup claim.
- A is recipe 00fd12a plus SM89 enablement and common storage/host adapters; it already has upstream automatic hot lending.
- B changes GPU glue, hot-slot copying/publication policy, service affinity and spin together. No individual component contribution is isolated.
- The 512 MiB hot setting is the global replacement-copy budget per rebalance round, not resident GPU capacity or spare floor. Initial/refill jobs are outside that cap.
- Timing rates come from rounded upstream stdout; decode includes prompt prefill and stochastic output. Both 35K rows are retained, including first-run setup.
- Piece/chunk SE describes variation over this fixed corpus, not run-to-run error or a numerical equivalence test.
- The independent 24-piece reference uses unchanged checkpoint arithmetic with a documented lossless storage bridge. Its earlier execution is reused; both served score sets are fresh.
- Reference pieces are 480 tokens and do not exercise competitive sparse index selection beyond top-k 512. WikiText is an A/B consistency screen without independent checkpoint scores.
- Retrieval covers approximately 4K/16K/35K. No 1M retrieval, concurrent sessions, prefix-cache policy or host-resident paging acceptance is implied.
- Host-KV default enabled does not prove host paging was activated. Pool geometry and conditional allocation remain separate runtime evidence.
- Live-top-k and fused-merge flags have runtime scope gates; an enabled flag is not proof of execution on every request.
- B-v9 classifies flat server-info telemetry separately from configuration and replays A planner inputs after independent B calibration. Both arms must have matching effective settings/capacities.
- The selected native-source reconstruction includes the common physical-NUMA adapter used in both arms. See `NATIVE_PROVENANCE.md` for the exact source and wheel identities; no byte-identical fresh-toolchain build is claimed.
- Before-client cgroup file-cache trimming and loading reclaim are distinct from client-window memory events. Raw NUMA snapshots do not alone establish every allocation policy.
- The generated `summary.json` remains review-data provenance. The separate owner review accepts only this observed workload and bounded follow-up testing; it does not establish numerical equivalence or repeatability.

All positive sampled B memory-PSI intervals were during the initial tool-call checks (21:09:07–13 UTC). No later increments appeared in NLL, either benchmark, WikiText or retrieval. The stall cause is not established. Zero memory-cap/OOM event deltas do not imply zero memory pressure.
