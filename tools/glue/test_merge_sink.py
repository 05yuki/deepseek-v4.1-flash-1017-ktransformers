"""Fused merge+sink vs the torch chain (_merge_partial_attn then _apply_attn_sink).

CPU structure check (Triton interpreter; libdevice maps to numpy there, so this is
a closeness check, not a bit check), from a tree where apply_bundle.py ran:
    cat <tree>/third_party/sglang/python/sglang/kernels/ops/attention/flash_mla_sm120_triton.py \
        tests/test_merge_sink.py | CUDA_VISIBLE_DEVICES= TRITON_INTERPRET=1 \
        KT_SM89_FUSED_MERGE=1 python -
GPU bit check (patched tree's environment, idle GPU):  python tests/test_merge_sink.py --gpu
"""
import sys

import torch

if "_fused_merge_sink" not in globals():
    from sglang.kernels.ops.attention.flash_mla_sm120_triton import (  # noqa: F401
        _apply_attn_sink,
        _fused_merge_sink,
        _merge_partial_attn,
    )


if __import__("os").environ.get("TRITON_INTERPRET") == "1":
    # the interpreter has no libdevice: numpy-backed stand-ins (closeness check only)
    import triton.language as _tl
    from triton.language.extra import libdevice as _ld

    _ld.exp = lambda x: _tl.exp(x)
    _ld.log = lambda x: _tl.log(x)
    _ld.log1p = lambda x: _tl.log(1.0 + x)


def _bf16_trunc(x):
    # the Triton interpreter converts fp32 -> bf16 by truncation (no rounding mode)
    return (x.float().contiguous().view(torch.int32) & -65536).view(torch.float32).to(torch.bfloat16)


def _ref_chain_trunc(o1, l1, o2, l2, sink):
    """_merge_partial_attn + _apply_attn_sink with truncating bf16 conversions."""
    out, lse = o1, l1
    if o2 is not None:
        max_lse = torch.maximum(l1, l2)
        w1 = torch.where(l1 > -1e20, torch.exp(l1 - max_lse), torch.zeros_like(l1))
        w2 = torch.where(l2 > -1e20, torch.exp(l2 - max_lse), torch.zeros_like(l2))
        total = (w1 + w2).clamp(min=1e-20)
        merged = (w1.unsqueeze(-1) * o1.float() + w2.unsqueeze(-1) * o2.float()) / total.unsqueeze(-1)
        out, lse = _bf16_trunc(merged), max_lse + torch.log(total)
    if sink is not None:
        comb = torch.logaddexp(lse, sink.view(1, 1, -1).expand_as(lse))
        w = torch.where(lse > -1e20, torch.exp(lse - comb), torch.zeros_like(lse))
        out, lse = _bf16_trunc(out.float() * w.unsqueeze(-1)), comb
    return out, lse


def case(B, H, D, dev, seed):
    g = torch.Generator().manual_seed(seed)
    o1 = (torch.randn(B, 1, H, D, generator=g) * 3).to(torch.bfloat16)
    o2 = (torch.randn(B, 1, H, D, generator=g) * 3).to(torch.bfloat16)
    l1 = torch.randn(B, 1, H, generator=g) * 4
    l2 = torch.randn(B, 1, H, generator=g) * 4
    # empty partials (-inf lse), a far-apart pair, equal lses
    l1[0, 0, 0] = float("-inf")
    l2[0, 0, 1] = float("-inf")
    l1[0, 0, 2] = l2[0, 0, 2] = float("-inf")
    l1[-1, 0, 3] = 90.0
    l2[-1, 0, 3] = -60.0
    l2[-1, 0, 4] = l1[-1, 0, 4]
    sink = torch.randn(H, generator=g)
    return [t.to(dev) for t in (o1, l1, o2, l2, sink)]


def run(dev):
    worst = 0
    bad = 0
    for seed, (B, H) in enumerate([(1, 64), (3, 64), (37, 64)]):
        o1, l1, o2, l2, sink = case(B, H, 512, dev, seed)
        for extra, use_sink in [(True, True), (True, False), (False, True)]:
            ro, rl = (o1, l1)
            if dev == "cpu":
                ro, rl = _ref_chain_trunc(o1, l1, o2 if extra else None, l2 if extra else None,
                                          sink if use_sink else None)
            else:
                if extra:
                    ro, rl = _merge_partial_attn(ro, rl, o2, l2)
                if use_sink:
                    ro, rl = _apply_attn_sink(ro, rl, sink)
            fo, fl = _fused_merge_sink(
                o1, l1, o2 if extra else None, l2 if extra else None, sink if use_sink else None
            )
            if dev == "cuda":
                same = torch.equal(fo.view(torch.int16), ro.view(torch.int16)) and torch.equal(
                    fl.view(torch.int32), rl.contiguous().view(torch.int32)
                )
                bad += int(not same)
                print(f"B={B} extra={extra} sink={use_sink}: bit-identical={same}")
            else:
                diff = (fo.float() - ro.float()).abs()
                # bf16 ulps of the reference value
                ulp = torch.where(ro.float() == 0, torch.ones_like(diff),
                                  2.0 ** (torch.floor(torch.log2(ro.float().abs().clamp_min(1e-30))) - 7))
                nulps = (diff / ulp).max().item()
                frac = (diff > 0).float().mean().item()
                d = diff.max().item()
                fin = torch.isfinite(rl)
                dl = (fl[fin] - rl[fin]).abs().max().item() if fin.any() else 0.0
                inf_ok = torch.equal(torch.isfinite(fl), fin)
                worst = max(worst, d)
                bad += int(nulps > 2 or frac > 1e-3 or dl > 1e-5 or not inf_ok)
                print(f"B={B} extra={extra} sink={use_sink}: max|dout|={d:.3g} ({nulps:.0f} bf16 ulp, "
                      f"{frac:.2e} of elements differ) max|dlse|={dl:.3g} inf_ok={inf_ok}")
    return bad


if __name__ == "__main__" or "_fused_merge_sink" in globals():
    dev = "cuda" if "--gpu" in sys.argv else "cpu"
    nbad = run(dev)
    print("merge_sink:", "PASS" if nbad == 0 else f"FAIL ({nbad})")
    raise SystemExit(1 if nbad else 0)
