"""kt_handoff_pack's pack kernel vs the copies it replaces.

CPU (interpreter: host addresses are plain CPU pointers), from the bundle directory:
    cat files/kt_handoff_pack.py tests/test_handoff_pack.py | \
      CUDA_VISIBLE_DEVICES= TRITON_INTERPRET=1 python -
GPU (patched tree's environment, idle GPU): python tests/test_handoff_pack.py --gpu
  bytes through pinned buffers, a captured CUDA graph replayed with new inputs, the
  row cap and the unpinned fallback. The ordering against kt-kernel's stream write is
  left to the full-model runs (it needs the KT_STREAM_MEMOPS poller).
"""
import sys

import torch
import triton

if "pack_to_host" not in globals():
    import sglang.srt.layers.moe.kt_handoff_pack as _mod
    from sglang.srt.layers.moe.kt_handoff_pack import _pack_kernel, pack_to_host  # noqa: F401
else:
    _mod = sys.modules[__name__]


def cpu_check():
    bad = 0
    for bs, hidden, k in [(1, 5120, 6), (4, 5120, 6), (3, 4096, 8)]:
        x = (torch.randn(bs, hidden) * 5).to(torch.bfloat16)
        w = torch.rand(bs, k)
        ids = torch.randint(-1, 384, (bs, k), dtype=torch.int32)
        hx = torch.empty_like(x)
        hw = torch.empty_like(w)
        hid = torch.empty((bs, k), dtype=torch.int64)
        nx = x.numel()
        _pack_kernel[(triton.cdiv(nx, 2048),)](
            x, w, ids, hx, hw, hid, nx, ids.numel(),
            BLOCK=2048, KBLOCK=max(16, triton.next_power_of_2(ids.numel())),
        )
        ok = torch.equal(hx.view(torch.int16), x.view(torch.int16)) and torch.equal(hw, w) and torch.equal(
            hid, ids.to(torch.long)
        )
        print(f"bs={bs} hidden={hidden} k={k}: {'ok' if ok else 'MISMATCH'}")
        bad += int(not ok)
    return bad


def _pinned(bs, hidden=5120, k=6):
    return (
        torch.empty((bs, hidden), dtype=torch.bfloat16, pin_memory=True),
        torch.empty((bs, k), dtype=torch.float32, pin_memory=True),
        torch.empty((bs, k), dtype=torch.int64, pin_memory=True),
    )


def _inputs(bs, hidden=5120, k=6, id_dtype=torch.int32):
    x = (torch.randn(bs, hidden, device="cuda") * 5).to(torch.bfloat16)
    w = torch.rand(bs, k, device="cuda")
    ids = torch.randint(-1, 384, (bs, k), dtype=id_dtype, device="cuda")
    return x, w, ids


def _same(x, w, ids, hx, hw, hid):
    return torch.equal(hx, x.cpu()) and torch.equal(hw, w.cpu()) and torch.equal(hid, ids.long().cpu())


def gpu_check():
    bad = 0
    for bs in (1, 2, 4, 256):
        for id_dtype in (torch.int32, torch.int64):
            x, w, ids = _inputs(bs, id_dtype=id_dtype)
            hx, hw, hid = _pinned(bs)
            assert pack_to_host(x, w, ids, hx, hw, hid), "pack fast path refused"
            torch.cuda.synchronize()
            ok = _same(x, w, ids, hx, hw, hid)
            print(f"gpu bs={bs} ids={id_dtype}: {'ok' if ok else 'MISMATCH'}")
            bad += int(not ok)
    return bad


def gpu_graph_check():
    bad = 0
    bs = 2
    x, w, ids = _inputs(bs)
    hx, hw, hid = _pinned(bs)
    assert pack_to_host(x, w, ids, hx, hw, hid)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        assert pack_to_host(x, w, ids, hx, hw, hid)
    for trial in range(3):
        nx, nw, nids = _inputs(bs)
        x.copy_(nx)
        w.copy_(nw)
        ids.copy_(nids)
        g.replay()
        torch.cuda.synchronize()
        ok = _same(x, w, ids, hx, hw, hid)
        print(f"gpu graph replay {trial}: {'ok' if ok else 'MISMATCH'}")
        bad += int(not ok)
    return bad


def gpu_fallback_check():
    bad = 0
    # above the row cap: refused, nothing issued, the switch stays on
    x, w, ids = _inputs(257)
    hx, hw, hid = _pinned(257)
    refused = not pack_to_host(x, w, ids, hx, hw, hid) and not _mod._disabled
    print(f"gpu row cap refuses 257 rows: {'ok' if refused else 'MISMATCH'}")
    bad += int(not refused)
    # unpinned host buffers: refused, and the pack turns itself off for the process
    x, w, ids = _inputs(2)
    hx = torch.empty((2, 5120), dtype=torch.bfloat16)
    hw = torch.empty((2, 6), dtype=torch.float32)
    hid = torch.empty((2, 6), dtype=torch.int64)
    refused = not pack_to_host(x, w, ids, hx, hw, hid) and _mod._disabled
    print(f"gpu unpinned buffers refused: {'ok' if refused else 'MISMATCH'}")
    bad += int(not refused)
    return bad


if __name__ == "__main__" or "pack_to_host" in globals():
    if "--gpu" in sys.argv:
        nbad = gpu_check() + gpu_graph_check() + gpu_fallback_check()  # fallback last: it disables
    else:
        nbad = cpu_check()
    print("handoff_pack:", "PASS" if nbad == 0 else f"FAIL ({nbad})")
    raise SystemExit(1 if nbad else 0)
