"""The fp32 e4m3 rounding of patches/dsv4_o_fp8_qdq.py (the path for GPUs without
Triton's e4m3 type) against torch's float8_e4m3fn cast, on the CPU: every bf16 value
times every power of two that keeps it within +-448 (what x / scale can be), plus the
clamp edges and signed zeros, plus random fp32.

usage: python e4m3-emulation-test.py
"""
import importlib.util
from pathlib import Path

import torch

torch.set_num_threads(1)
here = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("qdq", here / "patches/dsv4_o_fp8_qdq.py")
qdq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qdq)

allbf16 = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).float()
allbf16 = allbf16[torch.isfinite(allbf16)]
bad = n = 0
for k in range(-140, 141):
    v = (allbf16 * 2.0 ** k).clamp(-448.0, 448.0)
    v = v[torch.isfinite(v)]
    want = v.to(torch.float8_e4m3fn).float()
    got = qdq.e4m3_round_emulated(v)
    bad += int((got.view(torch.int32) != want.view(torch.int32)).sum())
    n += v.numel()
g = torch.Generator().manual_seed(0)
for _ in range(20):
    v = ((torch.rand(5_000_000, generator=g) * 2 - 1) * 448.0 * 2.0 ** torch.randint(-30, 1, (5_000_000,), generator=g).float())
    want = v.to(torch.float8_e4m3fn).float()
    got = qdq.e4m3_round_emulated(v)
    bad += int((got.view(torch.int32) != want.view(torch.int32)).sum())
    n += v.numel()
edge = torch.tensor([0.0, -0.0, 448.0, -448.0, 432.0, 440.0, 2.0 ** -9, 2.0 ** -10, -(2.0 ** -10), 3 * 2.0 ** -11, 2.0 ** -6, 15 * 2.0 ** -10])
bad += int((qdq.e4m3_round_emulated(edge).view(torch.int32) != edge.to(torch.float8_e4m3fn).float().view(torch.int32)).sum())
n += edge.numel()
print(f"{n} values, {bad} differ (bitwise, signed zero included)")
print("PASS" if bad == 0 else "FAIL")
