"""Exactness test of live_topk.py against the torch decode path's selection rule.

CPU (Triton interpreter, no GPU), from the bundle directory:
    cat files/live_topk.py tests/test_live_topk.py | \
      CUDA_VISIBLE_DEVICES= TRITON_INTERPRET=1 python -
GPU (patched tree's environment, idle GPU): python tests/test_live_topk.py --gpu
  which also compares against the real torch.topk + sort and topk_among_blocks calls
  on the device, and replays one captured CUDA graph with new device-side lengths.

The CPU reference reproduces torch's CUDA gatherTopK rule (strictly-greater keys,
then keys equal to the k-th in ascending scan order) over the FULL padded width, the
way the decode path calls it; the kernels see only the live columns.
"""
import sys

import torch

if "live_topk_full" not in globals():  # run as a file: import the installed module
    from sglang.kernels.ops.attention.dsv4.live_topk import (  # noqa: F401
        live_decode_slots,
        live_topk_among_blocks,
        live_topk_full,
    )


def ord_key(v: torch.Tensor) -> torch.Tensor:
    x = v.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    k = torch.where(x >= 0x80000000, x ^ 0xFFFFFFFF, x | 0x80000000)
    return torch.where(v == v, k, torch.full_like(k, 0xFFFFFFFF))


def cuda_topk_rule(keys: torch.Tensor, k: int) -> torch.Tensor:
    """Scan indices picked by torch's CUDA top-k (unsorted), for one row."""
    t = torch.sort(keys, descending=True).values[k - 1]
    gt = (keys > t).nonzero().flatten()
    eq = (keys == t).nonzero().flatten()[: k - gt.numel()]
    return torch.cat([gt, eq])


def ref_full(scores, slots, lens, k):
    bs, L = scores.shape
    kk = min(k, L)
    out = torch.full((bs, k), -1, dtype=torch.int32)
    for b in range(bs):
        idx = torch.sort(cuda_topk_rule(ord_key(scores[b]), kk)).values
        reach = idx < lens[b]
        out[b, :kk] = torch.where(reach, slots[b, idx.clamp_max(L - 1)], -1).to(torch.int32)
    return out


def ref_blocks(scores, slots, lens, blocks, k, bsz):
    bs, L = scores.shape
    nb = blocks.shape[1]
    kk = min(k, L)
    out = torch.full((bs, k), -1, dtype=torch.int32)
    for b in range(bs):
        f = torch.arange(nb * bsz)
        blk = blocks[b, f // bsz].to(torch.int64)
        pos = blk * bsz + f % bsz
        ok = (blk >= 0) & (pos < lens[b]) & (pos < L)
        vals = torch.where(ok, scores[b, pos.clamp(0, L - 1)], torch.tensor(float("-inf")))
        pick = cuda_topk_rule(ord_key(vals), min(kk, f.numel()))
        picked = torch.where(vals[pick] == float("-inf"), -1, pos[pick])
        idx = torch.sort(picked.masked_fill(picked < 0, L)).values
        if idx.numel() < kk:
            idx = torch.cat([idx, torch.full((kk - idx.numel(),), L)])
        reach = idx < lens[b]
        out[b, :kk] = torch.where(reach, slots[b, idx.clamp_max(L - 1)], -1).to(torch.int32)
    return out


def make_case(bs, L, lens, ratio, seed, nan_row=None):
    g = torch.Generator().manual_seed(seed)
    # few distinct bf16 values -> many ties at the threshold
    base = torch.randint(-40, 40, (bs, L), generator=g).float() / 8.0
    scores = base.to(torch.bfloat16).float()
    lens_t = torch.tensor(lens, dtype=torch.int64)
    j = torch.arange(L)
    scores = scores.masked_fill(j[None, :] >= lens_t[:, None], float("-inf"))
    if nan_row is not None and lens[nan_row] > 3:
        scores[nan_row, 2] = float("nan")
    r2t = torch.randint(0, 1 << 20, (bs + 2, L * ratio + 8), generator=g, dtype=torch.int32)
    req = torch.randperm(bs + 2, generator=g)[:bs].to(torch.int64)
    slots_ref = r2t[req[:, None], (j * ratio)[None, :]].to(torch.int64) // ratio
    slots_ref = slots_ref.masked_fill(j[None, :] >= lens_t[:, None], 0)
    return scores, lens_t, r2t, req, slots_ref


def check(dev, k=512, L=3000, ratio=1):
    bad = 0
    lens = [0, 1, k - 1, k, k + 1, 1500, L, L + 7]
    bs = len(lens)
    scores, lens_t, r2t, req, slots_ref = make_case(bs, L, lens, ratio, 1, nan_row=5)
    S, LN, R2T, RQ = scores.to(dev), lens_t.to(dev), r2t.to(dev), req.to(dev)
    slots = live_decode_slots(R2T, RQ, LN, ratio, L)
    for b in range(bs):
        n = min(lens[b], L)
        if not torch.equal(slots[b, :n].cpu(), slots_ref[b, :n]):
            bad += 1
            print("slots mismatch row", b)
    out = torch.full((bs, k), -1, dtype=torch.int32, device=dev)
    live_topk_full(S, slots, LN, min(k, L), out)
    ref = ref_full(scores, slots_ref, lens_t, k)
    for b in range(bs):
        if not torch.equal(out[b].cpu(), ref[b]):
            bad += 1
            print("full mismatch row", b, lens[b])
    # published blocks: finite (live) blocks first in a random rank order, -1 after
    for nb in (L // 8 + 1, 96):
        bsz = 8
        blocks = torch.full((bs, nb), -1, dtype=torch.int32)
        g = torch.Generator().manual_seed(7 + nb)
        for b in range(bs):
            nlive = (min(lens[b], L) + bsz - 1) // bsz
            perm = torch.randperm(nlive, generator=g)[:nb].to(torch.int32)
            blocks[b, : perm.numel()] = perm
        out = torch.full((bs, k), -1, dtype=torch.int32, device=dev)
        live_topk_among_blocks(S, slots, LN, blocks.to(dev), min(k, L), bsz, out)
        ref = ref_blocks(scores, slots_ref, lens_t, blocks, k, bsz)
        for b in range(bs):
            if not torch.equal(out[b].cpu(), ref[b]):
                bad += 1
                print("blocks mismatch nb", nb, "row", b, lens[b])
    return bad


def _torch_refs(S, LN, sref, blocks, kk, k, L):
    """The production torch selections: scoring.select_decode (topk + sort + gather)
    and dense_blocks.consume_decode (topk_among_blocks + write_decode)."""
    from sglang.kernels.ops.attention.dsv4.candidate_blocks import topk_among_blocks

    bs = S.shape[0]
    idx = S.topk(kk, dim=-1, sorted=False).indices.sort(dim=-1).values
    reach = idx < LN[:, None]
    ref = torch.full((bs, k), -1, dtype=torch.int32, device=S.device)
    ref[:, :kk] = torch.where(reach, sref.gather(1, idx.clamp_max(L - 1)), -1).to(torch.int32)
    pick = topk_among_blocks(S, LN, blocks, kk, block_size=8)
    pick = pick.masked_fill(pick < 0, L).sort(dim=-1).values
    reach = pick < LN[:, None]
    ref2 = torch.full((bs, k), -1, dtype=torch.int32, device=S.device)
    ref2[:, :kk] = torch.where(reach, sref.gather(1, pick.clamp_max(L - 1)), -1).to(torch.int32)
    return ref, ref2


def check_against_torch_gpu(k=512, L=16385, ratio=1):
    """Device-side check of the exact production calls (torch.topk + sort + gather)."""
    from sglang.kernels.ops.attention.dsv4.candidate_blocks import (
        select_candidate_block_ids,
        topk_among_blocks,
    )

    dev = "cuda"
    bad = 0
    cases = [([37, 512, 513, 4000, 16384, 16385], None),
             ([0, 1, 9000, 12000, 700, 511], None),
             ([37, 512, 513, 4000, 16384, 16385], 0),
             ([0, 1, 9000, 12000, 700, 511], 5)]
    if L >= 65536:
        cases += [([65536, 65535, 32768, 32769, 512, 0], None)]
    for seed, (lens, nan_row) in enumerate(cases):
        bs = len(lens)
        scores, lens_t, r2t, req, slots_ref = make_case(bs, L, lens, ratio, 100 + seed, nan_row=nan_row)
        S, LN, R2T, RQ = scores.cuda(), lens_t.cuda(), r2t.cuda(), req.cuda()
        slots = live_decode_slots(R2T, RQ, LN, ratio, L)
        kk = min(k, L)
        # full path (scoring.select_decode -> write_decode)
        idx = S.topk(kk, dim=-1, sorted=False).indices.sort(dim=-1).values
        reach = idx < LN[:, None]
        sref = slots_ref.cuda()
        ref = torch.full((bs, k), -1, dtype=torch.int32, device=dev)
        ref[:, :kk] = torch.where(reach, sref.gather(1, idx.clamp_max(L - 1)), -1).to(torch.int32)
        out = torch.full((bs, k), -1, dtype=torch.int32, device=dev)
        live_topk_full(S, slots, LN, kk, out)
        bad += int(not torch.equal(out, ref))
        # candidate path (dense_blocks.consume_decode)
        blocks = select_candidate_block_ids(S, LN[:, None], topk_blocks=2048, block_size=8)
        pick = topk_among_blocks(S, LN, blocks, kk, block_size=8)
        pick = pick.masked_fill(pick < 0, L).sort(dim=-1).values
        reach = pick < LN[:, None]
        ref2 = torch.full((bs, k), -1, dtype=torch.int32, device=dev)
        ref2[:, :kk] = torch.where(reach, sref.gather(1, pick.clamp_max(L - 1)), -1).to(torch.int32)
        out2 = torch.full((bs, k), -1, dtype=torch.int32, device=dev)
        live_topk_among_blocks(S, slots, LN, blocks, kk, 8, out2)
        bad += int(not torch.equal(out2, ref2))
        print("gpu seed", seed, "full", torch.equal(out, ref), "blocks", torch.equal(out2, ref2))
    return bad


def check_graph_replay(k=512, L=16385, ratio=1):
    """Capture slots + both selections once at one set of lengths, then replay the
    graph with other device-side lengths, scores and blocks copied into the static
    inputs; every replay must equal the eager torch selections."""
    from sglang.kernels.ops.attention.dsv4.candidate_blocks import select_candidate_block_ids

    cases = [
        [5, 600, 4000, 16385],
        [16385, 3, 513, 9000],
        [0, 512, 12000, 1],
        [16384, 16385, 511, 7000],
    ]
    bs, kk = len(cases[0]), min(k, L)
    scores, lens_t, r2t, req, _ = make_case(bs, L, cases[0], ratio, 300)
    S, LN, R2T, RQ = scores.cuda(), lens_t.cuda(), r2t.cuda(), req.cuda()
    BLK = select_candidate_block_ids(S, LN[:, None], topk_blocks=2048, block_size=8).contiguous()
    OUT1 = torch.full((bs, k), -1, dtype=torch.int32, device="cuda")
    OUT2 = torch.full((bs, k), -1, dtype=torch.int32, device="cuda")

    def body():
        slots = live_decode_slots(R2T, RQ, LN, ratio, L)
        live_topk_full(S, slots, LN, kk, OUT1)
        live_topk_among_blocks(S, slots, LN, BLK, kk, 8, OUT2)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        body()  # compile and warm up outside the capture
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        body()
    bad = 0
    for seed, lens in enumerate(cases[1:] + cases[:1]):
        scores, lens_t, r2t, req, slots_ref = make_case(bs, L, lens, ratio, 400 + seed)
        S.copy_(scores)
        LN.copy_(lens_t)
        R2T.copy_(r2t)
        RQ.copy_(req)
        BLK.copy_(select_candidate_block_ids(S, LN[:, None], topk_blocks=2048, block_size=8))
        OUT1.fill_(7)
        OUT2.fill_(7)
        g.replay()
        torch.cuda.synchronize()
        ref, ref2 = _torch_refs(S, LN, slots_ref.cuda(), BLK, kk, k, L)
        same = torch.equal(OUT1, ref) and torch.equal(OUT2, ref2)
        bad += int(not same)
        print("graph replay lens", lens, "identical", same)
    return bad


if __name__ == "__main__" or "live_topk_full" in globals():
    gpu = "--gpu" in sys.argv
    dev = "cuda" if gpu else "cpu"
    nbad = check(dev)
    if gpu:
        nbad += check_against_torch_gpu()
        nbad += check_graph_replay()
        nbad += check_against_torch_gpu(L=65536, ratio=2)
        nbad += check_graph_replay(L=65536, ratio=2)
    print("live_topk:", "PASS" if nbad == 0 else f"FAIL ({nbad})")
    raise SystemExit(1 if nbad else 0)
