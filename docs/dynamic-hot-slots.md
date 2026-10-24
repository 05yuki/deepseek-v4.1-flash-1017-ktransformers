# Dynamic hot-slot refill and publication

This extends tier-1 `hot-lend` and its existing `KT_KV_HOT_DYNAMIC_SLOTS=1`
mechanism. There is no `KT_KV_HOT_PORT` master or second slot manager.
The original dynamic-hot policy defaults stay intact. Optional controls:

| Variable (prefix `KT_KV_HOT_DYN_`) | Default | Effect |
|---|---:|---|
| `DEFER` | 0 | publish copied experts after their CUDA event completes |
| `ASYNC` | 0 | enqueue copies on a worker; implies DEFER |
| `REFILL_FAST` | 0 | refill released slots from saved scores; implies DEFER |
| `REFILL_RETRY` / `REFILL_TRIES` | 4 / 64 | fast refill retry interval / count |
| `BATCH_LOAD` | 0 | batch whole-part scale conversion and down scatter |
| `PIN_ARENAS` | 0 | register arenas before the first swap round |
| `ROUTE_FOLD` | 0 | emit resident masked routes in the routing kernel |
| `DWELL` | 0 | minimum rounds before replacing a new occupant |
| `BYTES_MB` | 0 | copy-traffic cap per round (0: unlimited) |

The existing `EVERY`, `SWAPS`, `HALF_LIFE`, and `HYST` controls remain in use.
Deferred publication (including ASYNC and REFILL_FAST) is TP1-only and is
rejected before model loading; ordinary multi-GPU dynamic hot slots remain
available with those three switches off. Copy jobs drain before slot memory
is released or resident weights are restored.

05yuki measured about 2% over the glue at TP1 with the earlier bundle's values.
That is a maintainer measurement of the original profile, not this split's
default policy. Moving experts between CPU and GPU can change numerical
results, as in the existing dynamic-hot mechanism. No fresh benchmark or
numerical-identity claim is included here.

RubenTorney's Strata PR #1904 supplied the idea of asynchronously copying
arena-backed experts and publishing only completed copies. This Python/Triton
implementation was written locally for SGLang/KT on 2026-10-22; it does not
copy that PR's C++ implementation. The former local “Strata” workspace label
was ambiguous and has been removed from implementation-origin claims.
