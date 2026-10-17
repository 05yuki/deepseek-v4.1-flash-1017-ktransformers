"""Prove a shard manifest returns the same rows as the extracted flat tables.

Run while both still exist: for random row ids it opens a row store on each
manifest and compares the 256 weight bytes and 8 scale bytes byte for byte.
Only after this passes is it safe to delete the extracted tables.
"""

from __future__ import annotations

import ctypes as C
import json
import os
import random
import sys
from pathlib import Path

import engram_backend as eb

U = C.c_uint64
_lib = eb._lib
_lib.row_store_lookup.argtypes = [C.c_void_p]


def open_store(manifest_path: Path, layer: str):
    man = json.loads(manifest_path.read_text())
    meta = man["layers"][layer]
    base = Path(man.get("base_dir") or manifest_path.parent)
    path = base / meta["file"]
    woff = int(meta.get("weight_offset", 0))
    soff = int(meta.get("scale_offset", meta["weight_bytes"]))
    store = _lib.row_store_open(str(path).encode(), meta["rows"], woff, soff, 0, 0)
    if not store:
        raise SystemExit(f"row_store_open failed: {path}")
    _lib.row_store_range(store, 0, meta["rows"])
    return store, meta, path, woff, soff


def fetch(store, ids):
    n = len(ids)
    id_arr = (C.c_int64 * n)(*ids)
    w = (C.c_uint8 * (n * 256))()
    s = (C.c_uint8 * (n * 8))()
    work = eb.Work(store, C.addressof(id_arr), C.addressof(w), C.addressof(s), n)
    _lib.row_store_lookup(C.addressof(work))
    return bytes(w), bytes(s)


def main() -> None:
    flat = Path(sys.argv[1])
    shard = Path(sys.argv[2])
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 256
    layers = sorted(json.loads(flat.read_text())["layers"])
    rng = random.Random(20260910)
    for layer in layers:
        fs, fmeta, fpath, fw, fsoff = open_store(flat, layer)
        ss, smeta, spath, sw, ssoff = open_store(shard, layer)
        if fmeta["rows"] != smeta["rows"]:
            raise SystemExit(f"layer {layer}: row count differs {fmeta['rows']} vs {smeta['rows']}")
        rows = fmeta["rows"]
        ids = [0, 1, rows - 2, rows - 1] + [rng.randrange(rows) for _ in range(n - 4)]
        fwb, fsb = fetch(fs, ids)
        swb, ssb = fetch(ss, ids)
        if fwb != swb or fsb != ssb:
            bad = next(i for i in range(len(ids))
                       if fwb[i*256:(i+1)*256] != swb[i*256:(i+1)*256]
                       or fsb[i*8:(i+1)*8] != ssb[i*8:(i+1)*8])
            raise SystemExit(f"layer {layer}: MISMATCH at row id {ids[bad]}")
        st = eb._stats(ss)
        print(f"layer {layer}: {len(ids)} rows identical  "
              f"flat={fpath.name} shard={spath.name}@{sw}/{ssoff}  errors={st['errors']}")
        if st["errors"] or eb._stats(fs)["errors"]:
            raise SystemExit(f"layer {layer}: row store reported read errors")
        _lib.row_store_close(fs)
        _lib.row_store_close(ss)
    print("SHARD MANIFEST VERIFIED")


if __name__ == "__main__":
    main()
