#!/usr/bin/env python
"""kt-kernel's AVX2 MXFP4 MoE at decode (one token) on one socket, at DeepSeek-V4.1-Flash-1017's
TP2-rank shapes (hidden 5120, intermediate 1152 a rank, 384 experts, scale codes on): the time
of a call with C routed experts whose weights come from DRAM, and the read rate it reaches.
Compare with tools/membw-probe at the same thread count on the same node.

Synthetic weights (random E2M1 bytes, E8M0 scales), one subpool on NODE. Each call routes the
token to C experts drawn at random from all 384, so the 3.4 GB set is far larger than L3.
KT_DECODE_PHASES=<n> prints kt-kernel's phases.

usage: numactl --cpunodebind=N --membind=N python kt-mxfp4-decode-bw.py [--threads 32,24,16] [--c 4]
"""
import argparse
import hashlib
import os
import time

import torch

os.environ.setdefault("KT_MXFP4_SCALE_CODES", "1")
from kt_kernel import kt_kernel_ext  # noqa: E402


def make(e, n, k, gs, seed):
    g = torch.Generator().manual_seed(seed)
    w = torch.randint(0, 256, (e, n, k // 2), dtype=torch.uint8, generator=g)
    ex = torch.randint(-12, -2, (e, n, k // gs), generator=g).to(torch.float32)
    s = torch.pow(2.0, ex).to(torch.bfloat16)
    return [w[i].contiguous() for i in range(e)], [s[i].contiguous() for i in range(e)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=384)
    ap.add_argument("--hidden", type=int, default=5120)
    ap.add_argument("--inter", type=int, default=1152)
    ap.add_argument("--c", type=int, default=4, help="routed experts a call on the CPU")
    ap.add_argument("--threads", default="32", help="comma-separated thread counts to time")
    ap.add_argument("--node", type=int, default=0)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--m", type=int, default=1, help="tokens a call (rows)")
    ap.add_argument("--same", action="store_true",
                    help="every row routed to the same C experts (a verify step's overlap), not C each")
    ap.add_argument("--check", type=int, default=0,
                    help="compare N calls of M rows with the same rows computed one at a time, byte for byte")
    args = ap.parse_args()
    gs = 32

    t0 = time.time()
    gw, gsc = make(args.experts, args.inter, args.hidden, gs, 1)
    uw, usc = make(args.experts, args.inter, args.hidden, gs, 2)
    dw, dsc = make(args.experts, args.hidden, args.inter, gs, 3)
    print(f"weights built in {time.time() - t0:.0f} s", flush=True)

    wpe = 3 * args.hidden * args.inter
    torch.manual_seed(0)
    M = args.m
    if args.same:
        sets = [torch.randperm(args.experts)[: args.c].repeat(M, 1).to(torch.int64).contiguous() for _ in range(997)]
    else:
        sets = [torch.stack([torch.randperm(args.experts)[: args.c] for _ in range(M)]).to(torch.int64).contiguous() for _ in range(997)]
    # bytes a call: the distinct experts it reads
    nbytes = sum(len(set(s.flatten().tolist())) for s in sets[:50]) / 50 * wpe * (0.5 + 1 / gs)
    rw = torch.rand(M, args.c, dtype=torch.float32).contiguous()
    x = (torch.randn(M, args.hidden) / 10).to(torch.bfloat16).contiguous()
    y = torch.empty(M, args.hidden, dtype=torch.bfloat16)
    bsz = torch.tensor([M], dtype=torch.int32)

    for nt in [int(v) for v in args.threads.split(",")]:
        wc = kt_kernel_ext.WorkerPoolConfig()
        wc.subpool_count = 1
        wc.subpool_numa_map = [args.node]
        wc.subpool_thread_count = [nt]
        cpu = kt_kernel_ext.CPUInfer(wc)
        cfg = kt_kernel_ext.moe.MOEConfig(args.experts, args.c, args.hidden, args.inter, 0)
        cfg.max_len = M
        cfg.pool = cpu.backend_
        cfg.quant_config.bits = 4
        cfg.quant_config.group_size = gs
        cfg.quant_config.zero_point = False
        cfg.gate_projs = [[t.data_ptr() for t in gw]]
        cfg.up_projs = [[t.data_ptr() for t in uw]]
        cfg.down_projs = [[t.data_ptr() for t in dw]]
        cfg.gate_scales = [[t.data_ptr() for t in gsc]]
        cfg.up_scales = [[t.data_ptr() for t in usc]]
        cfg.down_scales = [[t.data_ptr() for t in dsc]]
        moe = kt_kernel_ext.moe.AVX2MXFP4_MOE(cfg)
        p2l = torch.arange(args.experts, dtype=torch.int64).contiguous()
        cpu.submit(moe.load_weights_task(p2l.data_ptr()))
        cpu.sync()

        def call(ids):
            cpu.submit(moe.forward_task(bsz.data_ptr(), args.c, ids.data_ptr(), rw.data_ptr(),
                                        x.data_ptr(), y.data_ptr(), False))
            cpu.sync()

        j = 0
        for _ in range(50):
            call(sets[j % 997]); j += 1
        times = []
        for _ in range(args.rounds):
            t = time.perf_counter()
            for _ in range(args.iters):
                call(sets[j % 997]); j += 1
            times.append((time.perf_counter() - t) / args.iters * 1e6)
        med = sorted(times)[len(times) // 2]
        h = hashlib.sha256()
        for i in range(64):
            call(sets[i])
            h.update(y.view(torch.int16).numpy().tobytes())
        if args.check:
            y1 = torch.empty(1, args.hidden, dtype=torch.bfloat16)
            one = torch.tensor([1], dtype=torch.int32)
            same = 0
            for i in range(args.check):
                call(sets[i])
                for r in range(M):
                    ids_r = sets[i][r:r + 1].contiguous()
                    rw_r = rw[r:r + 1].contiguous()
                    x_r = x[r:r + 1].contiguous()
                    cpu.submit(moe.forward_task(one.data_ptr(), args.c, ids_r.data_ptr(), rw_r.data_ptr(),
                                                x_r.data_ptr(), y1.data_ptr(), False))
                    cpu.sync()
                    same += int(torch.equal(y1[0].view(torch.int16), y[r].view(torch.int16)))
            print(f"check: {same} of {args.check * M} rows byte-identical to the row alone", flush=True)
        print(f"threads {nt:3d}: {med:7.1f} us a call ({', '.join(f'{u:.0f}' for u in times)}), "
              f"{nbytes / 1e6:.1f} MB -> {nbytes / med / 1e3:.1f} GB/s, outputs {h.hexdigest()[:16]}", flush=True)
        del moe, cpu


if __name__ == "__main__":
    main()
