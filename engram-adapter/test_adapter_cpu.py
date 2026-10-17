"""CPU-only check of the engram adapter against the real tables.

1. sitecustomize must have wrapped sglang.srt.layers.engram so that
   EngramEmbedding.__init__ is the adapter's.
2. row_store lookups (nvme mode, O_DIRECT) must return exactly the bytes a
   plain pread of the flat file returns, for random rows, twice (miss then hit).
3. Out-of-range ids must zero the row and bump the error counter, not crash.
"""
import ctypes as C, json, os, random, sys, time
from pathlib import Path

import engram_backend as eb
from sglang.srt.layers import engram

assert engram.EngramEmbedding.__init__.__qualname__.startswith("install."), engram.EngramEmbedding.__init__.__qualname__
print("hook installed:", engram.EngramEmbedding.__init__.__qualname__)

d = Path(os.environ["DSV41_ENGRAM_DIR"])
man = json.loads((d / "engram-manifest.json").read_text())
lib = eb._lib
lib.row_store_lookup.argtypes = [C.c_void_p]
U = C.c_uint64
for layer, meta in man["layers"].items():
    # rows live in the checkpoint shard named by base_dir (or DSV41_ENGRAM_BASE)
    base = Path(os.environ.get("DSV41_ENGRAM_BASE") or man.get("base_dir") or d)
    path = base / meta["file"]
    rows, woff, soff = meta["rows"], 0, meta["weight_bytes"]
    st = lib.row_store_open(str(path).encode(), rows, woff, soff, 256 << 20, 0)
    assert st, "open failed"
    lib.row_store_range(st, 0, rows)
    n = 64
    ids = [random.randrange(rows) for _ in range(n - 1)] + [rows + 5]  # last one out of range
    id_arr = (C.c_int64 * n)(*ids)
    w = (C.c_uint8 * (n * 256))(); s = (C.c_uint8 * (n * 8))()
    work = eb.Work(st, C.addressof(id_arr), C.addressof(w), C.addressof(s), n)
    with path.open("rb") as f:
        for rep in ("miss", "hit"):
            t0 = time.perf_counter()
            lib.row_store_lookup(C.addressof(work))
            dt = (time.perf_counter() - t0) * 1e3
            for i, rid in enumerate(ids[:-1]):
                f.seek(woff + rid * 256); ew = f.read(256)
                f.seek(soff + rid * 8); es = f.read(8)
                assert bytes(w[i*256:(i+1)*256]) == ew, (layer, rid, "weight")
                assert bytes(s[i*8:(i+1)*8]) == es, (layer, rid, "scale")
            assert bytes(w[(n-1)*256:]) == b"\0" * 256
            print(f"layer {layer} {rep}: {n-1} rows exact, {dt:.2f} ms  stats={eb._stats(st)}")
    assert eb._stats(st)["errors"] == 2, eb._stats(st)  # one bad id per pass
    lib.row_store_close(st)
print("ADAPTER CPU TEST OK")
