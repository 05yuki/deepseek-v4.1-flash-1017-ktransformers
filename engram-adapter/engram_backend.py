"""Bounded exact file-backed replacement for EngramEmbedding's owned-row gather.

Derived from adapter/engram_backend.py in
https://github.com/0xSero/deepseek-v4.1-flash-4x-rtx-pro-6000
Copyright (c) 2026 0xSero, MIT License. Modifications for the KTransformers
dual-EPYC host:
  - the manifest names the file and the byte offsets of the weight and scale
    tensors, so the rows can come either from the flat tables written by
    tools/extract_engram_tables.py or straight out of the checkpoint shards
    (tools/make_engram_shard_manifest.py), which costs no extra disk;
  - libcudart is located through torch, not hard-coded site-packages globs;
  - callback errors are counted, not aborted; check_errors() raises after sync;
  - per-table cache budget comes from DSV41_CACHE_GIB split by layers and
    TP ranks, with the resident (RAM) mode selected by DSV41_ENGRAM_MODE=ram.

The original hash, gating, projections, TP all-reduce and model remain
unchanged. The rows each rank owns are looked up by a native callback that runs
inside CUDA graphs without calling the CUDA API, then handed to the upstream
`engram_gather` kernel with sequential ids, so FP8 decode and E8M0 scaling stay
exactly the upstream arithmetic.
"""

import ctypes as C
import json
import logging
import os
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

P, U = C.c_void_p, C.c_uint64
_HERE = Path(__file__).resolve().parent
_lib = C.CDLL(str(_HERE / "librow_store.so"))
_lib.row_store_open.argtypes = [C.c_char_p, U, U, U, U, C.c_int]
_lib.row_store_open.restype = P
_lib.row_store_range.argtypes = [P, U, U]
_lib.row_store_stats.argtypes = [P, C.POINTER(U)]
_lib.row_store_close.argtypes = [P]


class Work(C.Structure):
    _fields_ = [("store", P), ("ids", P), ("weights", P), ("scales", P), ("count", U)]


def _find_cudart() -> C.CDLL:
    candidates = []
    try:
        import nvidia.cuda_runtime  # type: ignore

        candidates += [
            str(p) for p in Path(nvidia.cuda_runtime.__file__).parent.glob("lib/libcudart.so*")
        ]
    except Exception:
        pass
    cuda_home = os.environ.get("CUDA_HOME") or "/usr/local/cuda"
    candidates += [str(p) for p in Path(cuda_home).glob("targets/x86_64-linux/lib/libcudart.so*")]
    candidates += [str(p) for p in Path(cuda_home).glob("lib64/libcudart.so*")]
    for c in candidates:
        try:
            return C.CDLL(c)
        except OSError:
            continue
    raise RuntimeError("libcudart.so not found for cudaLaunchHostFunc")


_cuda = _find_cudart()
_cuda.cudaLaunchHostFunc.argtypes = [P, P, P]
_cuda.cudaLaunchHostFunc.restype = C.c_int

DIM, SCALES, BLOCK = 256, 8, 32

# Lookups of at least this many rows (24 a token: n-grams 2..4 x 8 heads) are
# prefill-sized and share one staging set per device that is rebuilt larger
# when needed. Below it each power-of-two capacity keeps its own set, because
# the decode CUDA graphs captured those pointers. Before 10-06 every capacity
# kept its own set, so prefills of new lengths piled up device and pinned host
# buffers (170 MiB a rank after a 38K prompt) that were never used again.
EAGER_ROWS = int(os.environ.get("DSV41_ENGRAM_EAGER_ROWS", str(1 << 14)))


def _lend_prefill() -> bool:
    """True inside a KT prefill-lend window, when the lend scratch is resident."""
    try:
        from sglang.srt.layers.moe import kt_lend
    except ImportError:
        return False
    return kt_lend.ENABLED and kt_lend.STATE.get("phase") == "prefill"


def _stats(store) -> dict:
    buf = (U * 9)()
    _lib.row_store_stats(store, buf)
    return {
        "hits": buf[0], "misses": buf[1], "reads": buf[2],
        "cache_bytes": buf[3], "errors": buf[4], "slots": buf[5],
        "calls": buf[6], "lookup_ns": buf[7], "miss_ns": buf[8],
    }


def _start_timing_log(store, layer_id):
    # DSV41_ENGRAM_TIMING=<seconds>: log the row store's lookup time and miss
    # rate as deltas every interval, so the engram share of a decode step can
    # be read from server.log without a profiler.
    import threading, time, sys
    every = float(os.environ.get("DSV41_ENGRAM_TIMING", "0") or 0)
    if every <= 0:
        return
    def run():
        prev = _stats(store); t0 = time.time()
        while True:
            time.sleep(every)
            cur = _stats(store); t1 = time.time()
            d = {k: cur[k] - prev[k] for k in cur}
            calls = d["calls"]; looks = d["hits"] + d["misses"]
            if calls:
                print(f"[engram-timing layer {layer_id}] {t1-t0:.0f}s: calls={calls} rows={looks} "
                      f"miss={d['misses']} ({100.0*d['misses']/max(1,looks):.1f}%) "
                      f"lookup={d['lookup_ns']/1e6:.1f}ms ({d['lookup_ns']/1e3/calls:.0f}us/call) "
                      f"miss_io={d['miss_ns']/1e6:.1f}ms", file=sys.stderr, flush=True)
            prev, t0 = cur, t1
    threading.Thread(target=run, daemon=True, name=f"engram-timing-{layer_id}").start()


def install(module):
    cls = module.EngramEmbedding

    def init(self, num_embeddings, dim, layer_id):
        torch.nn.Module.__init__(self)
        assert dim == DIM, dim
        par = module.get_parallel()
        self.dim, self.tp_size = dim, par.tp_size
        rank = par.tp_rank
        self.row_start = num_embeddings * rank // self.tp_size
        end = num_embeddings * (rank + 1) // self.tp_size
        self.rows, self.host_table = end - self.row_start, None

        table_dir = Path(os.environ["DSV41_ENGRAM_DIR"])
        name = os.environ.get("DSV41_ENGRAM_MANIFEST", "engram-manifest.json")
        manifest = json.loads((table_dir / name).read_text())
        meta = manifest["layers"][str(layer_id)]
        # Rows may live in a flat extracted table next to the manifest, or in a
        # checkpoint shard somewhere else; base_dir (or DSV41_ENGRAM_BASE, which
        # wins so the checkpoint can move without editing the manifest) says
        # which. Offsets default to the flat layout: weights first, then scales.
        base = Path(os.environ.get("DSV41_ENGRAM_BASE") or manifest.get("base_dir") or table_dir)
        path = base / meta["file"]
        woff = int(meta.get("weight_offset", 0))
        soff = int(meta.get("scale_offset", meta["weight_bytes"]))
        assert meta["rows"] == num_embeddings, (meta["rows"], num_embeddings)
        assert meta["dim"] == DIM and meta["block_size"] == BLOCK, meta
        assert meta["weight_dtype"] == "F8_E4M3" and meta["scale_dtype"] == "F8_E8M0", meta
        n_layers = len(manifest["layers"])

        mode = os.environ.get("DSV41_ENGRAM_MODE", "nvme")
        budget_gib = float(os.environ.get("DSV41_CACHE_GIB", "128"))
        budget = int(budget_gib * 2**30) // (n_layers * self.tp_size)
        self._store = _lib.row_store_open(
            str(path).encode(), num_embeddings, woff, soff, budget, int(mode == "ram")
        )
        if not self._store:
            raise RuntimeError(f"row_store_open failed for {path}")
        _lib.row_store_range(self._store, self.row_start, end)
        self._staging, self._works = {}, {}

        # The loader must see the parameter names but never allocate the tables.
        self.weight = torch.nn.Parameter(torch.empty(0, dtype=torch.float8_e4m3fn), requires_grad=False)
        self.scale = torch.nn.Parameter(torch.empty(0, dtype=torch.float8_e8m0fnu), requires_grad=False)

        def validate(param, source):
            want = [num_embeddings, DIM] if param is self.weight else [num_embeddings, DIM // BLOCK]
            if list(source.shape) != want:
                raise ValueError(f"engram checkpoint shape {list(source.shape)} != {want}")

        self.weight.weight_loader = validate
        self.scale.weight_loader = validate
        logger.info(
            "engram adapter: mode=%s layer=%s rank=%s rows=[%s,%s) cache=%.1f GiB slots=%s "
            "file=%s woff=%s soff=%s",
            mode, layer_id, rank, self.row_start, end, budget / 2**30,
            _stats(self._store)["slots"], path.name, woff, soff,
        )
        _start_timing_log(self._store, layer_id)

    def owned(self, indices):
        from sglang.kernels.ops.embeddings.engram_gather import engram_gather

        count = indices.numel()
        if not count:
            return self._empty(indices)
        capacity = 1 << (count - 1).bit_length()
        dev = indices.device.index
        if capacity >= EAGER_ROWS and not torch.cuda.is_current_stream_capturing():
            # inside a lend window the set lives in the lend scratch (no VRAM
            # between prefills); outside one it is a plain allocation. The two
            # never mix: the scratch set is unmapped outside the window.
            lent = _lend_prefill()
            key = (dev, "eager-lend" if lent else "eager")
            have = self._staging.get(key)
            if have is None or have[0].numel() < capacity:
                if have is not None:
                    # host callbacks still queued hold the old Work and pinned rows
                    torch.cuda.current_stream().synchronize()
                    old = have[0].data_ptr()
                    self._works = {k: v for k, v in self._works.items() if v.ids != old}
                    del self._staging[key], have
                from contextlib import nullcontext

                region = nullcontext()
                if lent:
                    from sglang.srt.layers.moe import kt_lend

                    region = kt_lend.scratch_region()
                ids = torch.empty(capacity, dtype=torch.int64, pin_memory=True)
                w = torch.empty((capacity, DIM), dtype=torch.uint8, pin_memory=True)
                s = torch.empty((capacity, SCALES), dtype=torch.uint8, pin_memory=True)
                # dw / ds are rewritten by every lookup before the gather reads
                # them, so the scratch may lose their contents between windows;
                # seq is written once and must not live there
                with region:
                    dw = torch.empty((capacity, DIM), dtype=torch.uint8, device=indices.device)
                    ds = torch.empty((capacity, SCALES), dtype=torch.uint8, device=indices.device)
                seq = torch.arange(capacity, dtype=torch.int64, device=indices.device)
                self._staging[key] = (ids, w, s, dw, ds, seq)
        else:
            key = (dev, capacity)
            if key not in self._staging:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(f"engram staging {key} must be warmed before graph capture")
                ids = torch.empty(capacity, dtype=torch.int64, pin_memory=True)
                w = torch.empty((capacity, DIM), dtype=torch.uint8, pin_memory=True)
                s = torch.empty((capacity, SCALES), dtype=torch.uint8, pin_memory=True)
                dw, ds = w.to(indices.device), s.to(indices.device)
                seq = torch.arange(capacity, dtype=torch.int64, device=indices.device)
                self._staging[key] = (ids, w, s, dw, ds, seq)
        ids, w, s, dw, ds, seq = self._staging[key]
        wk = (dev, ids.data_ptr(), count)
        if wk not in self._works and len(self._works) >= 256:
            # every prompt's last chunk is a new count; drop the prefill-sized
            # ones once their callbacks have run (the decode ones stay)
            torch.cuda.current_stream().synchronize()
            self._works = {k: v for k, v in self._works.items() if k[2] < EAGER_ROWS}
        if wk not in self._works:
            self._works[wk] = Work(self._store, ids.data_ptr(), w.data_ptr(), s.data_ptr(), count)
        work = self._works[wk]
        ids[:count].copy_(indices.reshape(-1), non_blocking=True)
        err = _cuda.cudaLaunchHostFunc(
            torch.cuda.current_stream().cuda_stream, C.cast(_lib.row_store_lookup, P), C.addressof(work)
        )
        if err:
            raise RuntimeError(f"cudaLaunchHostFunc failed: {err}")
        dw[:count].copy_(w[:count], non_blocking=True)
        ds[:count].copy_(s[:count], non_blocking=True)
        out = self._empty(indices)
        engram_gather(dw.data_ptr(), ds.data_ptr(), seq[:count], out.view(-1, DIM), DIM, BLOCK)
        return out

    def check_errors(self):
        st = _stats(self._store)
        if st["errors"]:
            raise RuntimeError(f"engram row store reported {st['errors']} failed lookups: {st}")
        return st

    cls.__init__ = init
    cls._owned_rows = owned
    cls.engram_stats = check_errors
