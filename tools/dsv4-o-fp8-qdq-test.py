"""patches/dsv4_o_fp8_qdq.py against its torch reference and, when tilelang imports, the
checkpoint's own act_quant (inference/kernel.py): bit equality on attention-output-shaped
bf16 tensors, including tiny and huge blocks and an all-zero block.

usage: python dsv4-o-fp8-qdq-test.py [model dir with inference/kernel.py]
"""
import importlib.util
import os
import sys
from pathlib import Path

import torch

here = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("qdq", here / "patches/dsv4_o_fp8_qdq.py")
qdq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qdq)

ref_act_quant = None
if len(sys.argv) > 1:
    try:
        sys.dont_write_bytecode = True  # leave the model dir as downloaded
        sys.path.insert(0, str(Path(sys.argv[1]) / "inference"))
        from kernel import act_quant as ref_act_quant  # noqa: E402
    except Exception as e:  # tilelang missing
        print(f"reference act_quant not loaded: {e!r}")

bad = 0
# the e4m3 cast in hardware (sm89 and newer) and the fp32 emulation older GPUs get
paths = ["emulated"] + (["native"] if torch.cuda.get_device_capability() >= (8, 9) else [])
for path in paths:
    os.environ["O_FP8_QDQ_EMULATE"] = "1" if path == "emulated" else "0"
    qdq._NATIVE.clear()
    torch.manual_seed(0)
    for T in (1, 3, 37, 4096):
        o = torch.randn(T, 2, 4096, device="cuda", dtype=torch.bfloat16)
        o[0, 0, :32] = 0  # amax floor
        o[0, 1, :32] = -0.0
        o[-1, 1, 64:96] *= 1e4
        o[-1, 1, 96:128] *= 1e-6
        want = qdq.o_fp8_qdq_reference(o, 32)
        got = qdq.o_fp8_qdq_(o.clone(), 32)
        n = int((got.view(torch.int16) != want.view(torch.int16)).sum())
        msg = f"{path} T {T}: triton vs torch {n} bf16 differ"
        bad += n
        if ref_act_quant is not None:
            r = o.clone()
            ref_act_quant(r, 32, "ue8m0", torch.float8_e8m0fnu, True)
            m = int((got.view(torch.int16) != r.view(torch.int16)).sum())
            msg += f", vs the checkpoint's act_quant {m}"
            bad += m
        rel = ((got.float() - o.float()).norm() / o.float().norm()).item()
        print(msg + f"; relative change {rel:.4f}")
print("PASS" if bad == 0 else f"FAIL ({bad})")
