# Native source and measured wheel provenance

Both measured arms use the common recipe's native patch and a small
physical-NUMA mapping adapter. B additionally uses the optional
`kt-kernel-gpu-glue-only.patch`: its native changes are the poll-hint logic in
`cpuinfer.h` and the packed-handoff hook in `experts_base.py`. Queue spin is an
existing runtime control. The hot-slot patch adds no native source delta.

The measured B wheel SHA256 is
`44b58099fd4df210c76901030fc85d5e7bf032b0a419d317199d280f08954080`.
A uses
`436156b6cf447b38903b89fa9cc8329a0e92872fcb757dc58f2896389b1bfe30`.
The post-run receipts verify each installed native extension and Python module
against its wheel, plus the selected native source identities. B is the same
original strict-policy wheel whose selected source identity was recorded at
build time; the temporary preferred-policy experiment is not this measurement.

| Selected native file | A source SHA256 | B source SHA256 |
|---|---|---|
| `cpu_backend/cpuinfer.h` | `565c84e1466de281e90c1bad61ce7eafb97b257dd55fc24d0f4899c7ac162719` | `d9faa3a8e20aaefffbd6f29abdf60250721c270bf38d90e831265fc3bc46eac8` |
| `cpu_backend/worker_pool.cpp` | `8675cd3f0c163c3c6c4d04526a7cc624aa49809989839ec53028339e01551060` | `8675cd3f0c163c3c6c4d04526a7cc624aa49809989839ec53028339e01551060` |
| `python/experts_base.py` | `8fe111688e6f858943c1e985a4e9e211cce706e0635668de5bc293dd471fe590` | `3366dab25b36f116703010a7e83a00dc54d90cd71c54dd13a8a1a0f6657e9eea` |

The common strict-binding header `cpu_backend/worker_pool.h` has SHA256
`319aaf8fb18688e1465a592cee6e84914e3de12d54c10dafe347db0cbbaa9f99`.
The common adapter does not change its `HWLOC_MEMBIND_BIND`, thread-local or
strict flags. A bounded runtime probe verified strict binding and a faulted
page's node on every node before the measured campaigns.

## What differs from a clean optional-bundle setup

The optional patch intentionally excludes the already-common host adapter.
Consequently, the draft's three-file reconstruction proof starts from the
adapted A tree. It is not a claim that the current setup selector applies that
adapter. The exact additional source is provided as
[`common-host-numa-adapter.patch`](common-host-numa-adapter.patch), SHA256
`8b9eac865042e9cd30eb3367cbb5048b9d595705b25b895a4d28fd0c87637f58`.

It makes three changes in `worker_pool.cpp`:

- Include `unordered_map`, replacing the two dense vectors of per-node thread
  counts with maps keyed by physical NUMA ID.
- Preserve zero-initialized count/update behavior for present IDs.
- Map the distributor's subpool index through `numa_ids.at(index)` before
  setting its memory policy, rather than treating that index as a physical ID.

Applying the recipe's worker-pool hunks to the exact KTransformers
`95009ea6856c0799e517e93cb12be5e8494bc7ce` source (Git blob
`05d70660b32b33a30be3f1fa83375857570b0806`) produces unadapted worker-pool
SHA256 `aeb433b0fc65f7a769e55ca6a02ac79f2d824a99bf07447539ad610778b90345`.
The host adapter produces the shared measured `8675cd…` source above. This
source replay was checked locally with exact hunk contexts; no new native
binary was needed or built to establish the distinction.

Actual A and B startup logs both report eight subpools in the order
`[0:7] [1:7] [2:7] [3:7] [4:7] [5:7] [6:7] [7:7]`. For this observed identity
mapping, `numa_ids[i] == i`; both the original vector and the map counters start
at zero and update the same keys by seven. Thus the adapter preserves expert
pool target nodes and the strict memory-policy calls on this host. The C++
container's small bookkeeping allocations differ; byte-identical execution is
not claimed. Sparse/reordered physical IDs are where the mapping correction
matters, including avoiding a dense-vector index outside the subpool count.

For exact source reproduction in a separate build tree, apply the common
adapter after the recipe and optional B native patches, before compiling the
wheel. Its paths are relative to `kt-kernel/cpu_backend`; for example use
`git apply --directory=kt-kernel/cpu_backend` with the adapter's absolute path
from the KTransformers checkout. Then run the thirteen-file source verifier.
Do not patch an installed runtime without rebuilding/reverifying its native
artifact. A fresh toolchain may produce a different wheel binary; that is
distinct from the measured wheel/source provenance recorded here.

The comparison's measured native treatment is the B-only poll/handoff delta
on top of the same common adapter, not a performance claim for the adapter.
