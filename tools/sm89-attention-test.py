"""GPU regression for the SM89 dual-scope fallback; see docs/sm89.md.

Constructs V4 pages with padding, ragged lengths, sinks and masked indices;
compares against a dense FP64 oracle and checks CUDA graph replay.
Run on an idle SM89 card after setup.sh: .venv/bin/python tools/sm89-attention-test.py --gpu --no-timing
"""

import argparse
import sys

import torch
from sglang.kernels.ops.attention import sm89_dual_scope as DS
from sglang.kernels.ops.attention.flash_mla_sm120_triton import (
    flash_mla_sparse_decode_triton,
)

INTERP = False

H, D, PS = 64, 512, 64
TOPK_SWA, TOPK_EXTRA = 128, 512
FAILS = []


def page_bytes(ps):
    return -(-ps * 584 // 576) * 576  # KVLayout.V4.page_bytes


def make_cache(n_tok, gen, amp, device, ps=PS):
    """V4 paged cache with n_tok slots and its exact dequantized rows (fp64)."""
    n_pages = -(-n_tok // ps)
    n = n_pages * ps
    x = torch.randn(n, D, generator=gen) * amp
    x = x * torch.exp(
        torch.randn(n, 1, generator=gen) * 0.3
    )  # per-token magnitude spread
    nope = x[:, :448].reshape(n, 7, 64)
    e = torch.ceil(torch.log2(nope.abs().amax(-1).clamp(min=1e-12) / 448.0)).clamp(
        -60, 60
    )
    fp8 = (nope / torch.exp2(e)[..., None]).to(torch.float8_e4m3fn)
    rope = x[:, 448:].to(torch.bfloat16)
    data = torch.cat(
        [fp8.view(torch.uint8).reshape(n, 448), rope.view(torch.uint8).reshape(n, 128)],
        1,
    )
    scales = torch.cat(
        [(e + 127).to(torch.uint8), torch.zeros(n, 1, dtype=torch.uint8)], 1
    )
    pb = page_bytes(ps)
    raw = torch.zeros(n_pages, pb, dtype=torch.uint8)
    raw[:, : ps * 576] = data.reshape(n_pages, ps * 576)
    raw[:, ps * 576 : ps * 584] = scales.reshape(n_pages, ps * 8)
    raw = raw.to(device)
    cache = raw[:, : ps * 584].view(n_pages, ps, 1, 584).view(torch.float8_e4m3fn)
    deq = torch.cat(
        [
            (fp8.double() * torch.exp2(e).double()[..., None]).reshape(n, 448),
            rope.double(),
        ],
        1,
    )
    return cache, deq.to(device)


def make_indices(rows, topk, n_slots, gen, lo, device, holes=True):
    idx = torch.full((rows, topk), -1, dtype=torch.int32)
    lens = torch.randint(lo, topk + 1, (rows,), generator=gen, dtype=torch.int32)
    for r in range(rows):
        k = int(lens[r])
        idx[r, :k] = torch.randperm(n_slots, generator=gen)[:k].to(torch.int32)
        if holes and k > 4 and r % 3 == 1:
            idx[r, 1] = -1  # a hole inside the valid range (both kernels mask it)
    return idx.to(device), lens.to(device)


def make_case(rows, seed, device, score_std=3.0, swa_only=False):
    g = torch.Generator().manual_seed(seed)
    amp_k = 1.0
    swa, swa_deq = make_cache(4 * PS, g, amp_k, device)
    q = (
        (torch.randn(rows, 1, H, D, generator=g) * score_std)
        .to(torch.bfloat16)
        .to(device)
    )
    si, sl = make_indices(rows, TOPK_SWA, swa_deq.shape[0], g, 1, device)
    case = dict(
        q=q,
        k_cache=swa,
        indices=si,
        topk_length=sl,
        deq=swa_deq,
        attn_sink=(torch.randn(H, generator=g) * 2).to(device),
        softmax_scale=D**-0.5,
    )
    if not swa_only:
        ex, ex_deq = make_cache(16 * PS, g, amp_k, device)
        ei, el = make_indices(rows, TOPK_EXTRA, ex_deq.shape[0], g, 0, device)
        if rows >= 3:
            ei[0] = -1  # a row with no compressed candidates
            el[0] = 0
        case.update(
            extra_k_cache=ex, extra_indices=ei, extra_topk_length=el, extra_deq=ex_deq
        )
    return case


def reference(c):
    """fp64: one softmax over both candidate lists, sink in the denominator."""
    q = c["q"][:, 0].double()
    out = torch.zeros(q.shape, dtype=torch.float64, device=q.device)
    sink = c["attn_sink"].double()
    for r in range(q.shape[0]):
        ks = []
        for idx_k, len_k, deq_k in (
            ("indices", "topk_length", "deq"),
            ("extra_indices", "extra_topk_length", "extra_deq"),
        ):
            if c.get(idx_k) is None:
                continue
            idx = c[idx_k][r].long()
            n = (
                idx.shape[0]
                if c.get(len_k) is None
                else min(int(c[len_k][r]), idx.shape[0])
            )
            idx = idx[:n]
            idx = idx[idx >= 0]
            ks.append(c[deq_k][idx])
        k = torch.cat(ks, 0)
        if k.shape[0] == 0:
            continue
        s = q[r] @ k.T * c["softmax_scale"]  # [H, n]
        m = torch.maximum(s.amax(1), sink)
        w = torch.exp(s - m[:, None])
        out[r] = (w @ k) / (w.sum(1) + torch.exp(sink - m))[:, None]
    return out


def call_today(c):
    return flash_mla_sparse_decode_triton(
        q=c["q"],
        k_cache=c["k_cache"],
        indices=c["indices"].unsqueeze(1),
        topk_length=c["topk_length"],
        attn_sink=c["attn_sink"],
        head_dim_v=D,
        softmax_scale=c["softmax_scale"],
        extra_k_cache=c.get("extra_k_cache"),
        extra_indices=None
        if c.get("extra_indices") is None
        else c["extra_indices"].unsqueeze(1),
        extra_topk_length=c.get("extra_topk_length"),
    )[0][:, 0]


def call_dual(c, split="auto"):
    DS._SPLITK_MODE = split
    assert DS.supported(c["q"], c["k_cache"], c.get("extra_k_cache"), D, c["attn_sink"])
    return DS.sm89_dual_scope_decode(
        q=c["q"],
        k_cache=c["k_cache"],
        indices=c["indices"].unsqueeze(1),
        topk_length=c["topk_length"],
        attn_sink=c["attn_sink"],
        head_dim_v=D,
        softmax_scale=c["softmax_scale"],
        extra_k_cache=c.get("extra_k_cache"),
        extra_indices=None
        if c.get("extra_indices") is None
        else c["extra_indices"].unsqueeze(1),
        extra_topk_length=c.get("extra_topk_length"),
    )[:, 0]


def ordered(x):
    i = x.contiguous().view(torch.int16).to(torch.int32)
    mag = i & 0x7FFF
    return torch.where(i < 0, -mag, mag)


def ulp_of(ref):
    a = ref.abs().clamp(min=2.0**-126)
    return torch.exp2(torch.floor(torch.log2(a)) - 7)


def err_stats(o, ref):
    o = o.double()
    e = (o - ref).abs()
    rms = ref.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-30)
    eu = (e / ulp_of(ref)).flatten()
    q = torch.quantile(
        eu[: 1 << 24].float(), torch.tensor([0.5, 0.99], device=eu.device)
    ).tolist()
    return dict(
        max_abs=e.max().item(),
        mean_abs=e.mean().item(),
        ulp_p50=q[0],
        ulp_p99=q[1],
        ulp_max=eu.max().item(),
        max_rel_rms=(e / rms).max().item(),
    )


def diff_stats(a, b):
    d = (ordered(a) - ordered(b)).abs().flatten()
    edges = [0, 1, 2, 4, 8, 16]
    hist, prev = [], -1
    for e in edges:
        hist.append(((d > prev) & (d <= e)).float().mean().item())
        prev = e
    hist.append((d > edges[-1]).float().mean().item())
    rms = b.double().pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-30)
    ad = (a.double() - b.double()).abs()
    # ordered-int distance explodes across zero; the tail that matters is on elements that
    # are not tiny against their row
    big = (b.double().abs() >= rms / 16).flatten()
    return dict(
        hist=hist,
        max_ulp=int(d.max()),
        max_ulp_big=int(d[big].max()) if big.any() else 0,
        max_abs=ad.max().item(),
        max_rel_rms=(ad / rms).max().item(),
    )


def fmt_err(s):
    return (
        f"max|e| {s['max_abs']:.3e} mean|e| {s['mean_abs']:.3e} ulp p50/p99/max "
        f"{s['ulp_p50']:.2f}/{s['ulp_p99']:.2f}/{s['ulp_max']:.0f} max|e|/rms {s['max_rel_rms']:.2e}"
    )


def fmt_diff(s):
    h = " ".join(f"{x * 100:.2f}" for x in s["hist"])
    return (
        f"ulp-dist % [0|1|2|3-4|5-8|9-16|>16] {h}; max {s['max_ulp']} ulp "
        f"({s['max_ulp_big']} on |x| >= rms/16), "
        f"max|d| {s['max_abs']:.3e}, max|d|/rms {s['max_rel_rms']:.2e}"
    )


def check(name, ok, detail=""):
    if not ok:
        FAILS.append(f"{name}: {detail}")
    print(f"  [{'ok' if ok else 'FAIL'}] {name} {detail}")


def compare_case(label, c, splits, rel_tol):
    ref = reference(c)
    print(f"{label}: rows={c['q'].shape[0]} extra={'extra_k_cache' in c}")
    today = None
    try:
        today = call_today(c)
        st = err_stats(today, ref)
        print(f"  today       vs fp64: {fmt_err(st)}")
    except Exception as ex:  # the interpreter may not run today's autotuned kernel
        check("portable reference executes", False, f"{type(ex).__name__}: {ex}")
    for sp in splits:
        o = call_dual(c, sp)
        sd = err_stats(o, ref)
        print(f"  dual[{sp:>4s}] vs fp64: {fmt_err(sd)}")
        check(f"{label} dual[{sp}] finite", bool(torch.isfinite(o.float()).all()))
        check(
            f"{label} dual[{sp}] max|e|/rms <= {rel_tol}",
            sd["max_rel_rms"] <= rel_tol,
            f"{sd['max_rel_rms']:.2e}",
        )
        if today is not None:
            print(f"  dual[{sp:>4s}] vs today: {fmt_diff(diff_stats(o, today))}")
            st = err_stats(today, ref)
            check(
                f"{label} dual[{sp}] mean|e| <= 3x today",
                sd["mean_abs"] <= 3 * st["mean_abs"] + 1e-12,
                f"{sd['mean_abs']:.3e} vs {st['mean_abs']:.3e}",
            )
        if sp in ("auto", "0"):
            o2 = call_dual(c, sp)
            check(f"{label} dual[{sp}] deterministic", torch.equal(o, o2))


def empty_rows_case(device):
    c = make_case(4, 99, device)
    c["indices"][2] = -1
    c["extra_indices"][2] = -1
    for sp in ("0", "8"):
        o = call_dual(c, sp)
        check(
            f"empty row -> 0 (split {sp})",
            bool((o[2] == 0).all()) and bool(torch.isfinite(o.float()).all()),
        )


def batch_invariance(device):
    c = make_case(8, 7, device)
    per_row = ("q", "indices", "topk_length", "extra_indices", "extra_topk_length")
    one = {
        k: (v[1:2] if k in per_row else v) for k, v in c.items()
    }  # row 1 has both lists
    for sp in ("auto", "0"):
        a, b = call_dual(one, sp)[0], call_dual(c, sp)[1]
        same = torch.equal(a, b)
        print(
            f"  row 1 alone vs in an 8-row batch, split {sp}: "
            f"{'bit-identical' if same else 'differs (' + fmt_diff(diff_stats(a, b)) + ')'}"
        )
        if sp == "0":
            check("split 0 is batch-invariant", same)


def graph_check(device):
    for rows in (1, 2, 4, 5, 8, 9, 64):
        c = make_case(rows, 300 + rows, device)
        eager = call_dual(c, "auto").clone()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = call_dual(c, "auto")
        g.replay()
        torch.cuda.synchronize()
        check(f"CUDA graph replay == eager (rows {rows})", torch.equal(out, eager))
        c["q"].copy_(
            (torch.randn_like(c["q"].float()) * 3).to(torch.bfloat16)
        )  # new q, same buffers
        g.replay()
        torch.cuda.synchronize()
        check(
            f"CUDA graph follows new q (rows {rows})",
            torch.equal(out, call_dual(c, "auto")),
        )


def concurrent_graph_check(device):
    # Capture on one stream, replay on separate streams: capture-stream-keyed
    # global scratch would alias the buffers of these independent graphs.
    cases = [make_case(4, seed, device) for seed in (701, 702)]
    expected = [call_dual(c, "auto").clone() for c in cases]
    torch.cuda.synchronize()
    capture_stream = torch.cuda.Stream()
    graphs, outputs = [], []
    for case in cases:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            output = call_dual(case, "auto")
        graphs.append(graph)
        outputs.append(output)
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for _ in range(20):
        for stream, graph in zip(streams, graphs):
            with torch.cuda.stream(stream):
                graph.replay()
    torch.cuda.synchronize()
    for i, (out, ref) in enumerate(zip(outputs, expected)):
        check(f"independent graph scratch, graph {i}", torch.equal(out, ref))


def timing(device):
    print(
        "timing: per call, 40 calls (layers) captured in one CUDA graph, replayed 20x"
    )
    cfgs = ["16,64,8", "16,64,4", "16,32,4"]

    def graph_us(fn):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(40):
                fn()
        g.replay()
        torch.cuda.synchronize()
        s, e = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        s.record()
        for _ in range(20):
            g.replay()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) * 1000 / (20 * 40)

    for swa_only in (False, True):
        for rows in (1, 2, 4, 8, 16, 64, 256):
            c = make_case(rows, 500 + rows, device, swa_only=swa_only)
            line = [
                f"  {'swa-only' if swa_only else 'swa+c'} rows {rows:4d}: today {graph_us(lambda: call_today(c)):7.1f} us"
            ]
            for cfg in cfgs:
                DS._BLOCK_H, DS._BLOCK_N, DS._NUM_WARPS = (
                    int(x) for x in cfg.split(",")
                )
                for sp in ["auto", "0", "2", "4", "8"] if rows <= 8 else ["auto"]:
                    line.append(
                        f"dual[{cfg} s{sp}] {graph_us(lambda: call_dual(c, sp)):6.1f}"
                    )
            DS._BLOCK_H, DS._BLOCK_N, DS._NUM_WARPS = 16, 64, 8
            print(" | ".join(line))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--no-timing", action="store_true")
    a = ap.parse_args()
    if not a.gpu:
        ap.error(
            "pass --gpu on an idle SM89 GPU; use test_sm89_dispatch.py for CPU checks"
        )
    device = "cuda"
    if a.gpu:
        assert torch.cuda.get_device_capability() == (8, 9), "run on the SM89 card"
        torch.backends.cuda.matmul.allow_tf32 = False
        splits, tol, sizes = (
            ["auto", "file", "0", "2", "4", "8"],
            0.02,
            (1, 2, 4, 8, 37, 300),
        )
    else:
        # the interpreter truncates every bf16 cast (no RNE): looser bound than on the GPU
        splits, tol, sizes = ["file", "0", "2", "4", "8"], 0.05, (1, 3)
    print(
        f"device {device}, interpreter {INTERP}, portable reference, "
        f"cfg {DS._BLOCK_H},{DS._BLOCK_N},{DS._NUM_WARPS}"
    )
    for rows in sizes:
        for std in (1.0, 3.0, 6.0) if a.gpu else (3.0,):
            compare_case(
                f"random score_std={std}",
                make_case(rows, 10 * rows + int(std), device, std),
                splits,
                tol,
            )
    compare_case(
        "swa-only layer", make_case(sizes[-1], 5, device, swa_only=True), splits, tol
    )
    empty_rows_case(device)
    batch_invariance(device)
    if a.gpu:
        graph_check(device)
        concurrent_graph_check(device)
        if not a.no_timing:
            timing(device)
    print("RESULT:", "PASS" if not FAILS else f"FAIL ({len(FAILS)})")
    for f in FAILS:
        print("  ", f)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
