"""The GPU MoE kernels of SGLANG_MXFP4_MOE_FP32 (sglang/srt/layers/moe/mxfp4_fp32_moe.py in
the tree patch) against an fp64 reference of the same MXFP4 arithmetic, at
DeepSeek-V4.1-Flash-1017's sizes for one of two tensor-parallel ranks (H 5120, I 1152, top-6;
MXFP4_TEST_I=2304 for one GPU), with random E2M1 codes and E8M0 scales. Beside it Marlin on
the same weights, timed in CUDA graphs. Routes at weight 0 on expert 0 stand in for the
CPU's routes. Then the same calls 100 times: the outputs must repeat bit for bit.

usage: PYTHONPATH=third_party/sglang/python .venv/bin/python tools/mxfp4-fp32-moe-test.py
"""
import importlib.util
import os
import sys

import torch

if len(sys.argv) > 1:
    spec = importlib.util.spec_from_file_location("mxfp4_fp32_moe", sys.argv[1])
    mm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mm)
else:
    from sglang.srt.layers.moe import mxfp4_fp32_moe as mm

torch.manual_seed(0)
dev = "cuda"
H, E, TOPK, LIMIT = 5120, 8, 6, 10.0
# I: one TP2 rank's intermediate (1152, not a multiple of the GEMV's 256-value step) by default
I = int(os.environ.get("MXFP4_TEST_I", "1152"))


def rand_mxfp4(n, k):
    q = torch.randint(0, 256, (E, n, k // 2), dtype=torch.uint8, device=dev)
    s = torch.randint(118, 123, (E, n, k // 32), dtype=torch.uint8, device=dev)
    return q, s


w13, s13 = rand_mxfp4(2 * I, H)
w2, s2 = rand_mxfp4(H, I)


def routes(M, masked=True):
    ids = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(M)]).to(torch.int64)
    w = torch.rand(M, TOPK, device=dev)
    w = w / w.sum(1, keepdim=True)
    if masked:
        ids[:, -2:] = 0  # masked routes: expert 0 at weight 0
        w[:, -2:] = 0
    return ids, w


def rel(a, b):
    return float((a.double() - b).norm() / b.norm())


def graph_ms(fn, reps=20):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(reps):
        g.replay()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / reps


def eager_ms(fn, reps=10):
    fn(); torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(reps):
        fn()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / reps


marlin = None
try:
    from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import fused_marlin_moe
    from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_mxfp4_layer_for_marlin

    layer = torch.nn.Module()
    layer.w13_weight = torch.nn.Parameter(w13.clone().view(torch.int8), requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(w2.clone().view(torch.int8), requires_grad=False)
    layer.w13_weight_scale_inv = torch.nn.Parameter(s13.clone().view(torch.float8_e8m0fnu), requires_grad=False)
    layer.w2_weight_scale_inv = torch.nn.Parameter(s2.clone().view(torch.float8_e8m0fnu), requires_grad=False)
    layer.num_experts = E
    layer.num_local_experts = E
    layer.hidden_size = H
    layer.intermediate_size_per_partition = I
    layer.params_dtype = torch.bfloat16
    prepare_moe_mxfp4_layer_for_marlin(layer)

    def marlin(x, w, ids):
        return fused_marlin_moe(hidden_states=x, w1=layer.w13_weight, w2=layer.w2_weight,
                                w1_scale=layer.w13_weight_scale, w2_scale=layer.w2_weight_scale,
                                gating_output=torch.empty((x.shape[0], E), device=dev, dtype=torch.float32),
                                topk_weights=w, topk_ids=ids, global_num_experts=E,
                                num_bits=4, is_k_full=True, clamp_limit=LIMIT)
except Exception as exc:  # the comparison is optional
    print(f"(Marlin comparison unavailable: {type(exc).__name__}: {exc})")

ok = True
for M in (1, 2, 4, 8, 16, 64, 512):
    x = (torch.randn(M, H, device=dev) * 0.5).to(torch.bfloat16)
    ids, w = routes(M)
    ref = mm.mxfp4_moe_reference(x, w13, s13, w2, s2, w, ids, LIMIT)
    out = mm.mxfp4_fp32_moe(x, w13, s13, w2, s2, w, ids, LIMIT)
    once = rel(ref.to(torch.bfloat16), ref)
    e = rel(out, ref)
    line = f"M {M:4d}: fp32 kernels {e:.2e} (one rounding {once:.2e})"
    ok &= e <= once * 1.05 + 1e-6 and bool(torch.isfinite(out).all())
    fn = lambda: mm.mxfp4_fp32_moe(x, w13, s13, w2, s2, w, ids, LIMIT)
    t_new = graph_ms(fn) if M <= mm.GEMV_MAX_TOKENS else eager_ms(fn)
    if M > mm.GEMV_MAX_TOKENS:  # the capture fallback (GEMV) is checked for correctness too
        g_out = torch.empty_like(out)
        s_ = torch.cuda.Stream(); s_.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s_):
            mm.mxfp4_fp32_moe(x, w13, s13, w2, s2, w, ids, LIMIT, out=g_out)
        torch.cuda.current_stream().wait_stream(s_)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            mm.mxfp4_fp32_moe(x, w13, s13, w2, s2, w, ids, LIMIT, out=g_out)
        g_out.zero_(); g.replay(); torch.cuda.synchronize()
        eg = rel(g_out, ref)
        ok &= eg <= once * 1.05 + 1e-6
        line_g = f" (in a graph {eg:.2e})"
    else:
        line_g = ""
    line += f", {t_new:.3f} ms{line_g}"
    if marlin is not None:
        try:
            mo = marlin(x, w, ids)
            t_m = (graph_ms if M <= mm.GEMV_MAX_TOKENS else eager_ms)(lambda: marlin(x, w, ids))
            line += f" | Marlin {rel(mo, ref):.2e}, {t_m:.3f} ms"
        except Exception as exc:
            line += f" | Marlin failed: {type(exc).__name__}: {str(exc)[:120]}"
    print(line, flush=True)
for M in (1, 2, 16, 64):
    x = (torch.randn(M, H, device=dev) * 0.5).to(torch.bfloat16)
    ids, w = routes(M)
    first = mm.mxfp4_fp32_moe(x, w13, s13, w2, s2, w, ids, LIMIT).clone()
    same = sum(torch.equal(mm.mxfp4_fp32_moe(x, w13, s13, w2, s2, w, ids, LIMIT), first) for _ in range(100))
    print(f"M {M:4d}: {same} of 100 repeats bit-identical", flush=True)
    ok &= same == 100
print("PASS" if ok else "FAIL")
