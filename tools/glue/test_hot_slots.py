"""CPU-only checks for the rebased hot-slot port. No GPU, no server.

usage: CUDA_VISIBLE_DEVICES="" TRITON_INTERPRET=1 python test_hot_slots.py <orig dir> <new dir> [--sim]

<orig dir> holds the current base sources; <new dir> holds the candidate sources.
The modules are loaded by path with a
stub sglang package; Triton kernels run in Triton's interpreter on CPU tensors.

1. the swap planner (_plan_layer/_plan_round) at HYST 1.0, DWELL 0, no byte cap chooses
   exactly what the round loop before the port chose (random scores and holdings);
2. --sim: a synthetic routing model, swap rate and hit share by hysteresis/dwell/byte cap
   on a stationary text and after a change of text;
3. the batched loader matches the upstream _arena_load_parts for 8 small whole parts
   and full six/eight-part 2304x5120 geometry, both row orders, including negative
   scales; malformed layouts fail;
4. the route kernel: with ROWS_ZERO off it writes what the tree's kernel writes; with it on
   its resident (ids, weights) equal the wrapper's KT_GPU_MASK_ZERO chain;
5. master-off defaults, candidate knobs, wrapper glue preservation, dwell and byte caps;
6. _refill_choose; the deferred/async job state machine with stub events.
The old optional SLOT_ACC arithmetic test is excluded with that feature.
"""
import importlib.util
import ast
import copy
import os
import sys
import threading
import types
from pathlib import Path

import numpy as np
import torch

ORIG, NEW = sys.argv[1], sys.argv[2]
SIM = "--sim" in sys.argv
FAILS = []


def check(cond, what):
    print(("ok   " if cond else "FAIL ") + what, flush=True)
    if not cond:
        FAILS.append(what)


def _stub_pkg():
    for name in ("sglang", "sglang.srt", "sglang.srt.layers", "sglang.srt.layers.moe"):
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = []
            sys.modules[name] = m
    vmm = types.ModuleType("sglang.srt.layers.moe.kt_kv_vmm")
    vmm.ENABLED = False
    sys.modules["sglang.srt.layers.moe.kt_kv_vmm"] = vmm
    sys.modules["sglang.srt.layers.moe"].kt_kv_vmm = vmm


def load(path, name, env=None):
    _stub_pkg()
    old = {}
    for k, v in (env or {}).items():
        old[k] = os.environ.get(k)
        os.environ[k] = v
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return mod


DYN_ENV = {"KT_KV_HOT_LEND": "1", "KT_KV_HOT_DYNAMIC_SLOTS": "1", "KT_KV_HOT_DYNAMIC_RESIDENT": "1",
           "KT_GPU_MASK_ZERO": "1"}
old_mod = load(os.path.join(ORIG, "kt_kv_hot_lend.py"), "hl_orig", DYN_ENV)
new_mod = load(os.path.join(NEW, "kt_kv_hot_lend.py"), "hl_new", dict(DYN_ENV, KT_KV_HOT_PORT="1", KT_KV_HOT_ROUTE_FOLD="1"))


# -- 1. planner equivalence -----------------------------------------------------------

def legacy_round(LAYERS, scores, budget, hyst, res_on):
    """The round loop of the tree's _dyn_round, verbatim but for its inputs."""
    swaps = []
    for L in LAYERS:
        k = L.k
        if not k:
            continue
        sc, held = scores[L.idx], L.held
        res = res_on and L.res_t is not None
        places = [(0, j) for j in range(k)] + ([(1, r) for r in range(L.n)] if res else [])

        def has(p):
            return held[p[1]] if p[0] == 0 else L.res_held[p[1]]

        hs, changed, n = {has(p) for p in places}, set(), 0
        for e in np.argsort(-sc, kind="stable").tolist():
            if n >= budget or sc[e] <= 0:
                break
            if (e < L.n and not res) or e in hs:
                continue
            opts = [p for p in places if p not in changed]
            if not opts:
                break
            empty = [p for p in opts if has(p) < 0]
            if empty:
                p = empty[0]
            else:
                p = min(opts, key=lambda q: (sc[has(q)], q))
                if not sc[e] > sc[has(p)] * hyst:
                    break
            old = has(p)
            swaps.append((L, p[0], p[1], old, e))
            hs.discard(old)
            hs.add(e)
            if p[0] == 0:
                held[p[1]] = e
            else:
                L.res_held[p[1]] = e
            changed.add(p)
            n += 1
    return swaps


class FakeLayer:
    def __init__(self, idx, n, k, E, rng):
        self.idx, self.n, self.k, self.x, self.b = idx, n, k, k, 2
        perm = rng.permutation(E).tolist()
        self.held = [e if rng.random() > 0.1 else -1 for e in perm[:k]]
        self.res_held = perm[k:k + n]
        self.res_now = list(self.res_held)
        self.res_t = {} if n else None
        self.since = {}
        self.names = [(0, "w13_weight"), (0, "w2_weight"), (0, "w13_weight_scale_inv"), (0, "w2_weight_scale_inv")]
        self.sz = {"w13_weight": 11796480, "w2_weight": 5898240, "w13_weight_scale_inv": 737280,
                   "w2_weight_scale_inv": 368640}


def planner_equivalence(cases=400):
    rng = np.random.default_rng(1)
    m = new_mod
    m.DYN_HYST, m.DYN_DWELL, m.DYN_BYTES = 1.0, 0, 0
    bad = 0
    for c in range(cases):
        E, nl = 64, 4
        res_on = bool(c % 2)
        mk = lambda: [FakeLayer(i, 1 + (c % 3), 4 + (c % 7), E, np.random.default_rng(c * 10 + i)) for i in range(nl)]
        La, Lb = mk(), mk()
        scores = rng.exponential(1.0, (nl, E)) * (rng.random((nl, E)) > 0.3)
        if c % 5 == 0:
            scores = np.round(scores * 2) / 2  # ties
        budget = 1 + c % 9
        want = legacy_round(La, scores, budget, 1.0, res_on)
        m.LAYERS[:] = Lb
        m.DYN.update(scores=scores, rounds=c, res=res_on)
        m.DYN_SWAPS, m.DYN_EVERY = budget, 1
        got = m._plan_round()
        a = [(L.idx, w, i, o, e) for L, w, i, o, e in want]
        b = [(L.idx, w, i, o, e) for L, w, i, o, e in got]
        if a != b or [L.held for L in La] != [L.held for L in Lb] or [L.res_held for L in La] != [L.res_held for L in Lb]:
            bad += 1
    m.LAYERS[:] = []
    check(bad == 0, f"planner at HYST 1.0 / DWELL 0 / no byte cap = the tree's round loop ({cases} random rounds, "
                    f"{bad} differ)")


# -- 2. a routing model ------------------------------------------------------------------

def simulate(hyst, dwell, bytes_mb, every=16, swaps=1, forwards=5120, layers=8, E=384, k=23, n=1, seed=0,
             half_life=256.0):
    rounds = forwards // every
    m = new_mod
    m.DYN_HYST, m.DYN_DWELL, m.DYN_BYTES = hyst, dwell, bytes_mb << 20
    m.DYN_SWAPS, m.DYN_EVERY = swaps, every
    rng = np.random.default_rng(seed)
    base = 1.0 / (np.arange(E) + 6.0) ** 1.1
    base /= base.sum()

    def dists(s):
        r = np.random.default_rng(s)
        return [base[r.permutation(E)] for _ in range(layers)]

    phases = [dists(seed + 100), dists(seed + 200)]
    Ls = []
    for i in range(layers):
        L = FakeLayer(i, n, k, E, np.random.default_rng(seed + i))
        L.held = list(range(n, n + k))
        L.res_held = list(range(n))
        L.res_now = list(range(n))
        Ls.append(L)
    m.LAYERS[:] = Ls
    scores = np.zeros((layers, E))
    m.DYN.update(scores=scores, rounds=0, res=True)
    out = {"swaps": [], "hit": []}
    for rnd in range(2 * rounds):
        p = phases[rnd // rounds]
        seen = np.stack([rng.multinomial(every * 6, p[i]) for i in range(layers)]).astype(np.float64)
        hit = sum(seen[L.idx, [e for e in L.held if e >= 0] + L.res_held].sum() for L in Ls) / seen.sum()
        scores *= 0.5 ** (every / half_life)
        scores += seen
        m.DYN["rounds"] = rnd + 1
        sw = m._plan_round()
        out["swaps"].append(len(sw))
        out["hit"].append(hit)
    m.LAYERS[:] = []
    s, h = np.array(out["swaps"]), np.array(out["hit"])
    steady = slice(rounds * 2 // 3, rounds)
    final = h[2 * rounds - rounds // 6:].mean()
    rec = next((i for i in range(rounds, 2 * rounds) if h[i] >= final - 0.02), 2 * rounds) - rounds
    return {"swaps/forward/layer": s[steady].mean() / every / layers, "hit": h[steady].mean(),
            "first round": int(s[0]), "max/round": int(s.max()),
            "forwards to recover": rec * every, "hit after": final}


# -- 3. batched part loader ------------------------------------------------------------------

def loader_equivalence(P=8, mm=64, K=64, E=5):
    m = new_mod
    n = P * mm
    kb, kg = K // 2, K // 32
    ws, ss = mm * K // 2, 4 * mm * K // 32
    g = torch.Generator().manual_seed(3)
    parts = []
    for p in range(P):
        ar = []
        for _ in range(3):
            ar.append((torch.randint(0, 256, (E * ws,), dtype=torch.uint8, generator=g), ws, None))
        for _ in range(3):
            ex = torch.randint(100, 150, (E * ss // 4,), generator=g).to(torch.float32)
            sign = torch.where(torch.rand(E * ss // 4, generator=g) < 0.2, -1.0, 1.0)
            ar.append(((sign * torch.exp2(ex - 127)).view(torch.uint8), ss, None))
        parts.append(ar)
    st = types.SimpleNamespace(_segs=[(p, 0, mm, p * mm, mm) for p in range(P)], _arena_parts={7: parts},
                               _arenas={7: parts[0]}, _seg_whole=False, _register_arenas=lambda i: None)

    class L:
        layer_idx = 7

    m.DYN["stage"] = torch.empty(max(n * kg + 1, K * n // 32), dtype=torch.float32)
    old_mod.DYN["stage"] = torch.empty_like(m.DYN["stage"])
    ok = True
    for order in (0, 1):
        for e in range(E):
            outs = []
            for fn in (old_mod._arena_load_parts, m._arena_load_parts_batched):
                w13 = torch.zeros(2 * n, kb, dtype=torch.uint8)
                s13 = torch.zeros(2 * n, kg, dtype=torch.uint8)
                w2 = torch.zeros(K, n // 2, dtype=torch.uint8)
                s2 = torch.zeros(K, n // 32, dtype=torch.uint8)
                fn(st, L, e, w13, s13, w2, s2, order)
                outs.append((w13, s13, w2, s2))
            ok = ok and all(torch.equal(a, b) for a, b in zip(*outs))
    check(ok, f"batched part loader writes the same bytes as _arena_load_parts ({P} parts, both row orders, "
              f"{E} experts, negative scales included)")
    check(m._parts_whole(st), f"{P} whole parts take the batched path")
    st2 = types.SimpleNamespace(_segs=[(0, 32, 32, 0, 64), (1, 0, 32, 32, 64)])
    m.DYN.pop("whole_key", None)
    check(not m._parts_whole(st2), "a rank's half of two parts (two GPUs) keeps the per-part path")
    m.DYN.pop("whole_key", None)
    # Check actual entry-point validation before writes, including 2304x5120.
    keys = (("w13", False), ("w13", True), ("w2", False), ("w2", True))
    dst = dict(zip(keys, outs[1]))
    m._validate_arena_layout(st, L, E - 1, dst)
    check(True, f"{P} x {mm}-row parts / K={K}: arena bounds and group geometry accepted")
    sg = types.ModuleType("sglang.srt.layers.moe.kt_stream_prefill")
    sg._instance = st
    sys.modules[sg.__name__] = sg
    sys.modules["sglang.srt.layers.moe"].kt_stream_prefill = sg
    m.BATCH_LOAD = True
    m._arena_load(L, E - 1, dst, 1)
    check(all(torch.equal(dst[key], want) for key, want in zip(keys, outs[0])),
          f"{P}-part complete arena entry point equals upstream per-part loader")
    invalid = []
    partial = copy.copy(st)
    partial._segs = [(0, 16, mm - 16, 0, mm)] + st._segs[1:]
    invalid.append((partial, E - 1, "unaligned partial part"))
    missing = copy.copy(st)
    missing._arena_parts = {7: parts[:-1]}
    invalid.append((missing, E - 1, "missing part"))
    wrong_whole = copy.copy(st)
    wrong_whole._seg_whole = True
    invalid.append((wrong_whole, E - 1, "inconsistent whole-part metadata"))
    gap = copy.copy(st)
    gap._segs = list(st._segs)
    gap._segs[-1] = (P - 1, 0, mm, (P - 1) * mm + 32, mm)
    invalid.append((gap, E - 1, "gap/overlap in destination"))
    invalid.extend(((st, -1, "negative expert"), (st, E, "expert outside arena bounds")))
    unequal = copy.copy(st)
    unequal._arena_parts = {7: [list(ar) for ar in parts]}
    tensor, stride, mmapped = unequal._arena_parts[7][0][1]
    unequal._arena_parts[7][0][1] = (torch.cat((tensor, torch.zeros(64, dtype=torch.uint8))), stride + 32, mmapped)
    invalid.append((unequal, 0, "unequal projection strides"))
    multi = copy.copy(st)
    multi.world = 2
    invalid.append((multi, E - 1, "TP2 deferred publication"))
    m.DYN_DEFER = True
    rejected = []
    for candidate, expert, why in invalid:
        sg._instance = candidate
        saved = {key: t.clone() for key, t in dst.items()}
        try:
            m._arena_load(L, expert, dst, 1)
        except RuntimeError:
            rejected.append(all(torch.equal(dst[key], saved[key]) for key in dst))
        else:
            rejected.append(False)
    check(all(rejected), f"{P}-part loader refuses all {len(invalid)} unsupported layouts before slot writes")
    # A valid aligned partial layout retains the legacy loader (no truncation).
    m.DYN_DEFER = False
    p_st = types.SimpleNamespace(_segs=[(0, 32, 32, 0, mm), (1, 0, 32, 32, mm)],
                                _arena_parts={7: parts[:2]}, world=2)
    p_dst = {keys[0]: torch.empty((128, K // 2), dtype=torch.uint8),
             keys[1]: torch.empty((128, K // 32), dtype=torch.uint8),
             keys[2]: torch.empty((K, 32), dtype=torch.uint8),
             keys[3]: torch.empty((K, 2), dtype=torch.uint8)}
    m._validate_arena_layout(p_st, L, 0, p_dst)
    check(not m._parts_whole(p_st), "aligned partial partitions validate but never use whole-part batch path")
    m.DYN_DEFER = True


def eight_part_geometry():
    """Use the actual streamer's segmentation function without loading its runtime."""
    tree = ast.parse(Path(ORIG, "kt_stream_prefill.py").read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_rank_segments")
    namespace = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "_rank_segments", "exec"), namespace)
    segs = namespace["_rank_segments"](0, 1, 8, 2304)
    check(segs == [(p, 0, 288, p * 288, 288) for p in range(8)],
          "actual streamer gives 8 whole 288-row parts covering all 2304 rows at GPU TP1")
    check(all(v % 32 == 0 for seg in segs for v in seg[1:])
          and (288 * 5120 // 2) % 64 == 0 and (4 * 288 * 5120 // 32) % 64 == 0,
          "8-part native group32 geometry and 64-byte expert weight/scale strides align")
    # The cache must not retain a result when an existing list is changed in place.
    st = types.SimpleNamespace(_segs=segs)
    whole = new_mod._parts_whole(st)
    segs[-1] = (7, 32, 256, 7 * 288, 288)
    check(whole and not new_mod._parts_whole(st),
          "8-part geometry cache notices in-place segment changes")


# -- 4. route kernel -----------------------------------------------------------------------

def route_kernel():
    if new_mod._route_kernel is None:
        check(False, "triton missing")
        return
    E, nl, M, topk = 64, 3, 4, 6
    g = torch.Generator().manual_seed(5)
    tbl = torch.full((nl, E), -1, dtype=torch.int64)
    rtbl = torch.full((nl, E), -1, dtype=torch.int64)
    for i in range(nl):
        perm = torch.randperm(E, generator=g)
        tbl[i, perm[:10]] = torch.arange(10)
        rtbl[i, perm[10:12]] = torch.arange(2)
    ids = torch.randint(0, E, (M, topk), generator=g, dtype=torch.int32)
    ids[0, 0] = -1
    w = torch.rand(M, topk, generator=g, dtype=torch.float32)
    outs = {}
    for name, m in (("orig", old_mod), ("new", new_mod)):
        m.STATE["tbl"], m.STATE["rtbl"] = tbl.clone(), rtbl.clone()
        m.STATE["cnt"] = torch.zeros((nl, E), dtype=torch.float32)
        L = types.SimpleNamespace(idx=1, n=2)
        outs[name] = (m._route_fused(L, ids, w), m.STATE["cnt"].clone())
    (o, co), (nw, cn) = outs["orig"], outs["new"]
    same = all(torch.equal(a, b) for a, b in zip(o[:5], nw[:5])) and torch.equal(co, cn)
    check(same, "route kernel: CPU ids/weights, slot ids/weights, rows and counts as the tree's kernel")
    rows = nw[4]
    msk = rows < 0
    want_ids = torch.where(msk, torch.zeros_like(rows), rows).to(torch.int64)
    want_w = torch.where(msk, torch.zeros_like(w), w)
    rz = nw[5]
    check(rz is not None and torch.equal(rz[0], want_ids) and rz[0].dtype == torch.int64
          and torch.equal(rz[1], want_w) and torch.equal(rz[1].view(torch.int32), want_w.view(torch.int32)),
          "route fold: the resident (ids, weights) equal the wrapper's mask-zero chain, bit for bit")


# -- 5. top-k sum with the add -----------------------------------------------------------------



# -- 6. refill choice and the job state machine ----------------------------------------------

def refill_plan():
    m = new_mod
    layer = types.SimpleNamespace(idx=0, n=1, b=2, x=3, res_t={}, res_now=[0],
                                  res_held=[1], held=[3, 2, 0], since={})
    m.DYN.update(res=True, rounds=7, scores=np.array([[0., 2., 3., 10.]]))
    job, pending = {"items": [], "pub": {0: [], 1: []}}, set()
    m._refill(layer, 2, job, pending)
    check([(w, slot, expert, src) for _, w, slot, expert, src in job["items"]]
          == [(1, 0, 1, "arena"), (0, 0, 3, "arena"), (0, 1, 2, "bank")],
          "refill plan includes resident restore, eight-part arena slot and placement bank slot")
    check(job["pub"] == {0: [(0, 3, 0), (0, 2, 1)], 1: [(0, 1, 0)]}
          and pending == {(0, 1, 0), (0, 0, 0), (0, 0, 1)},
          "refill publication and pending sets exactly cover all planned copies")
    check(layer.held == [3, 2, -1] and layer.res_now == [1]
          and layer.since == {(1, 0): 7, (0, 0): 7, (0, 1): 7},
          "refill drops unmapped tail slots and starts dwell for every replaced place")


def refill_and_jobs():
    m = new_mod
    sc = np.zeros(20)
    sc[[5, 9, 3, 12, 0]] = [9, 8, 7, 6, 5]
    got = m._refill_choose(sc, [3, 4, 9, -1], 4, 1, 2, {0})
    check(got == [3, 5, 9, 12], f"refill: the 4 best scores but the resident row's, 3 and 9 keep their slots ({got})")
    got = m._refill_choose(None, [2, -1, 4], 3, 1, 2, {0})
    check(got == [2, -1, 4], f"refill without scores keeps what was held ({got})")
    got = m._refill_choose(sc, [3, 4, 9, 7, 8], 2, 1, 2, {0})
    check(got == [5, 9], f"refill to fewer slots takes the best two ({got})")

    # the job state machine, stub CUDA objects
    class Ev:
        def __init__(self):
            self.done, self.recorded = False, False

        def record(self, s=None):
            self.recorded = True

        def query(self):
            return self.done or not self.recorded

    class Stream:
        def wait_event(self, ev):
            pass

    cuda = types.SimpleNamespace(Event=Ev, current_stream=lambda dev=None: Stream(), stream=lambda s: _Null(),
                                 device=lambda d: _Null(), Stream=Stream)

    class _Null:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    real = m.torch.cuda
    writes = []
    m.torch.cuda = cuda
    m._tbl_write = lambda table, rows, dev: writes.extend(rows) if table is not None and rows else None
    copied = []
    m._copy_item = lambda L, w, i, e, src: copied.append((L.idx, w, i, e, src))
    m.STATE["tbl"], m.STATE["rtbl"] = "tbl", "rtbl"
    m.DYN["stream"] = Stream()
    try:
        for asyn in (False, True):
            m.DYN_ASYNC = asyn
            writes.clear()
            copied.clear()
            job = m._job_new("round")
            L = types.SimpleNamespace(idx=2)
            job["items"].append((L, 0, 1, 33, "arena"))
            job["pub"][0].append((2, 33, 1))
            m._job_start(job, "dev")
            m._job_join()
            check(copied == [(2, 0, 1, 33, "arena")] and job["enq"], f"job enqueued (worker thread {asyn})")
            check(not m._job_publish("dev") and not writes and m.DYN["job"] is job,
                  "copies not done: nothing enters the table, the job stays")
            job["ev"].done = True
            check(m._job_publish("dev") and writes == [(2, 33, 1)] and m.DYN["job"] is None,
                  "copies done: the entry goes in at the next forward")
        # a worker that raises: the scheduler thread sees it
        m.DYN_ASYNC = True

        def boom(*a):
            raise ValueError("x")

        m._copy_item = boom
        job = m._job_new("round")
        job["items"].append((types.SimpleNamespace(idx=0), 0, 0, 1, "arena"))
        m._job_start(job, "dev")
        try:
            m._job_join()
            check(False, "a failed enqueue raises on the scheduler thread")
        except RuntimeError:
            check(True, "a failed enqueue raises on the scheduler thread")
        m.DYN["job"] = None
    finally:
        m.torch.cuda = real


def off_and_scope():
    env = dict(DYN_ENV, KT_KV_HOT_PORT="0")
    for knob in ("DYN_ASYNC", "DYN_DEFER", "REFILL_FAST", "BATCH_LOAD", "PIN_ARENAS", "ROUTE_FOLD"):
        env["KT_KV_HOT_" + knob] = "1"
    off = load(os.path.join(NEW, "kt_kv_hot_lend.py"), "hl_off", env)
    settings = ("ENABLED", "EXTRA", "EXTRA_AUTO", "DYNAMIC", "DYN_RES", "DYN_EVERY", "DYN_SWAPS", "DYN_HALF_LIFE", "DYN_HYST", "RETRY", "MIN_MB", "MARGIN_MB", "SPARE_FLOOR_MB")
    check(all(getattr(off, k) == getattr(old_mod, k) for k in settings), "master off preserves upstream settings/defaults including enabled dynamic slots")
    check(not any(getattr(off, k) for k in ("DYN_ASYNC", "DYN_DEFER", "REFILL_FAST", "BATCH_LOAD", "PIN_ARENAS", "ROUTE_FOLD", "DYN_DWELL", "DYN_BYTES")), "master off overrides every inherited candidate feature switch")
    malformed = dict(env)
    for knob in ("REFILL_RETRY", "REFILL_TRIES", "DYN_DWELL", "DYN_BYTES_MB"):
        malformed["KT_KV_HOT_" + knob] = "stale-invalid-value"
    stale = load(os.path.join(NEW, "kt_kv_hot_lend.py"), "hl_off_stale", malformed)
    check(all(getattr(stale, k) == getattr(old_mod, k) for k in settings) and stale.REFILL_RETRY == 4 and stale.REFILL_TRIES == 0,
          "master off ignores malformed candidate-only refill/dwell/byte-cap environment values")
    on = load(os.path.join(NEW, "kt_kv_hot_lend.py"), "hl_on", dict(DYN_ENV, KT_KV_HOT_PORT="1"))
    check(on.DYN_HYST == 1.25 and on.DYN_DWELL == 2 and on.DYN_BYTES == 512 << 20 and on.DYN_SWAPS == 1 and all(getattr(on, k) for k in ("DYN_ASYNC", "DYN_DEFER", "REFILL_FAST", "BATCH_LOAD", "PIN_ARENAS", "ROUTE_FOLD")), "master on selects intended candidate knobs (1.25/2/512 MiB and feature set)")
    check(not hasattr(on, "ZERO_RES") and not hasattr(on, "SLOT_ACC") and on.load_count(0) == old_mod.load_count(0), "zero-resident and slot-accumulator experiments excluded")
    a = ast.parse(Path(ORIG, "kt_ep_wrapper.py").read_text())
    b = ast.parse(Path(NEW, "kt_ep_wrapper.py").read_text())
    ca = next(n for n in a.body if isinstance(n, ast.ClassDef) and n.name == "KTEPWrapperMethod")
    cb = next(n for n in b.body if isinstance(n, ast.ClassDef) and n.name == "KTEPWrapperMethod")
    old_methods = {n.name: ast.dump(n) for n in ca.body if isinstance(n, ast.FunctionDef) and n.name != "apply"}
    new_methods = {n.name: ast.dump(n) for n in cb.body if isinstance(n, ast.FunctionDef) and n.name != "apply"}
    check(old_methods == new_methods, "wrapper glue submit/sync/create methods remain AST-identical")
    text = Path(NEW, "kt_kv_hot_lend.py").read_text()
    check('min_bytes = max(1, min_bytes - max(0, spare_now))' in text and 'm.decode_lend(runner, RESERVE, min_bytes)' in text, "latest upstream free-page plus spare-memory lend floor retained")


def preflight_before_lend():
    m = load(os.path.join(NEW, "kt_kv_hot_lend.py"), "hl_preflight", dict(DYN_ENV, KT_KV_HOT_PORT="1"))
    pkg = sys.modules["sglang.srt.layers.moe"]
    names = ("kt_stream_prefill", "kt_kv_prefill_lend")
    saved = {name: (getattr(pkg, name, None), sys.modules.get(pkg.__name__ + "." + name)) for name in names}
    st = types.SimpleNamespace(world=1, _arenas={7: []})
    stream = types.ModuleType(pkg.__name__ + ".kt_stream_prefill")
    stream.ZEROCOPY, stream._instance = True, st
    lend = types.ModuleType(pkg.__name__ + ".kt_kv_prefill_lend")
    calls = []
    lend.STATE, lend.PRE_RESTORE = {"lent": None}, []
    lend.decode_lend = lambda *args: calls.append(args) or 0
    for name, module in zip(names, (stream, lend)):
        setattr(pkg, name, module)
        sys.modules[module.__name__] = module
    m.ENABLED, m.SPARE_FLOOR_MB = True, -1
    m.LAYERS[:] = [types.SimpleNamespace(layer_idx=7, names=[("param", "w")], slots={"w": types.SimpleNamespace(device="cpu")}, k=0)]
    runner = types.SimpleNamespace(is_draft_worker=False)
    cases = (
        ("GPU TP2", lambda: setattr(st, "world", 2)),
        ("disabled dynamic slots", lambda: setattr(m, "DYNAMIC", False)),
        ("disabled zero-copy", lambda: setattr(stream, "ZEROCOPY", False)),
        ("missing streamer", lambda: setattr(stream, "_instance", None)),
        ("missing layer arenas", lambda: setattr(st, "_arenas", {})),
    )
    initial = dict(m.STATE)
    try:
        for why, mutate in cases:
            st.world, st._arenas = 1, {7: []}
            stream.ZEROCOPY, stream._instance, m.DYNAMIC = True, st, True
            mutate()
            try:
                m.tick(runner)
            except RuntimeError:
                check(not calls and not lend.PRE_RESTORE and m.STATE == initial,
                      f"{why} rejected before decode_lend, hook registration and state mutation")
            else:
                check(False, f"{why} must fail before lending")
        st.world, st._arenas = 1, {7: []}
        stream.ZEROCOPY, stream._instance, m.DYNAMIC = True, st, True
        m.tick(runner)
        check(len(calls) == 1 and len(lend.PRE_RESTORE) == 1,
              "valid TP1 candidate reaches decode_lend after preflight")
        calls.clear()
        m.HOT_SLOTS, m.DYNAMIC, stream.ZEROCOPY, stream._instance = False, False, False, None
        m.STATE["want"] = True
        m.tick(runner)
        check(len(calls) == 1, "master off bypasses candidate preflight and preserves upstream lend entry")
    finally:
        for name, (attr, module) in saved.items():
            if attr is None:
                delattr(pkg, name)
            else:
                setattr(pkg, name, attr)
            if module is None:
                sys.modules.pop(pkg.__name__ + "." + name, None)
            else:
                sys.modules[pkg.__name__ + "." + name] = module


def dwell_and_budget():
    m = new_mod
    sc = np.array([1.0, 1.1, 1.3, 2.0])
    p = [(0, 0)]
    check(m._plan_layer(sc, p, {p[0]: 0}, 0, False, 1, 1.25)[0][2] == 3,
          "hysteresis selects the strongest gain")
    check(m._plan_layer(sc[:2], p, {p[0]: 0}, 0, False, 1, 1.25) == [],
          "hysteresis refuses subthreshold replacement")
    L = FakeLayer(0, 0, 1, 4, np.random.default_rng(77))
    L.held, L.since = [0], {p[0]: 10}
    m.LAYERS[:] = [L]
    m.DYN.update(scores=sc[None, :], rounds=12, res=False)
    m.DYN_DWELL, m.DYN_HYST, m.DYN_BYTES = 2, 1.25, 0
    check(m._plan_round() == [], "dwell holds a place for the next two rounds")
    m.DYN["rounds"] = 13
    check(len(m._plan_round()) == 1, "dwell expires on the third round")
    plans = [("a", [(None, 0, 1, 10), (None, 1, 2, 2)]), ("b", [(None, 0, 1, 9), (None, 1, 2, 8)])]
    got = m._byte_cut(plans, 512 << 20, lambda L: 22 << 20)
    check(sum(len(p) for _, p in got) == 4 and m.DYN["round_bytes"] <= 512 << 20, "round bytes account for all planned copies")
    many = [("a", [(None, i, i + 1, 100 - i) for i in range(50)])]
    got = m._byte_cut(many, 512 << 20, lambda L: 22 << 20)
    check(len(got[0][1]) == 23 and m.DYN["round_bytes"] == 506 << 20, "512 MiB round cap keeps a plan prefix without exceeding budget")
    m.LAYERS[:] = []


if __name__ == "__main__":
    torch.set_num_threads(2)
    off_and_scope()
    preflight_before_lend()
    planner_equivalence()
    dwell_and_budget()
    loader_equivalence()
    loader_equivalence(P=6, mm=384, K=5120, E=2)
    eight_part_geometry()
    loader_equivalence(P=8, mm=288, K=5120, E=2)
    route_kernel()
    refill_plan()
    refill_and_jobs()
    if SIM:
        print("\nrouting model: 8 layers x 384 experts, 1 resident row + 23 slots, Zipf(1.1) routes, 6 a token;"
              " 5120 forwards of one text, then 5120 of another; 1 swap a forward and layer")
        for every, hl, hyst, dwell, mb in ((16, 256, 1.0, 0, 0), (16, 256, 1.25, 0, 0), (16, 256, 1.25, 2, 0),
                                           (16, 256, 1.25, 2, 256), (16, 256, 1.5, 2, 256), (8, 256, 1.25, 2, 256),
                                           (8, 128, 1.25, 2, 256), (8, 128, 1.25, 2, 128), (8, 128, 1.0, 0, 0)):
            r = simulate(hyst, dwell, mb, every=every, half_life=hl)
            print(f"  every {every:>2} half-life {hl} hyst {hyst:<4} dwell {dwell} cap {mb:>3} MiB: " +
                  ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in r.items()), flush=True)
    print(f"\n{len(FAILS)} failed" if FAILS else "\nall passed")
    sys.exit(1 if FAILS else 0)
