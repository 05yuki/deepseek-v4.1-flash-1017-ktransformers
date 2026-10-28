"""flash_mla_sparse_decode_triton with the fused merge+sink on vs off (GPU only).

Exercises the patched entry point itself: the two sparse decodes into torch.empty
buffers, then the fused kernel, against the unmodified torch chain on zero-filled
buffers. Random V4 pages (584 B per token: 448 fp8 nope, 64 bf16 rope, 8 scale bytes)
with -1 indices and per-row lengths. Eager and CUDA-graph replay must both be
bit-identical to the off path. Run in the patched tree's environment, idle GPU:
    python tests/test_fused_merge_wiring.py
The switch is forced through the module's gate, so this runs on any CUDA GPU.
"""
import torch

import sglang.kernels.ops.attention.flash_mla_sm120_triton as fm


def make_cache(num_pages, page_size, g):
    raw = torch.randint(0, 0x7F, (num_pages, page_size * 584), generator=g, dtype=torch.uint8)
    # UE8M0 scales around 2^0
    scales = raw[:, page_size * 576 :].view(num_pages, page_size, 8)
    scales.copy_(torch.randint(120, 128, scales.shape, generator=g, dtype=torch.uint8))
    data = raw[:, : page_size * 576].view(num_pages, page_size, 576)
    rope = data[..., 448:576].contiguous().view(torch.bfloat16)
    rope.copy_((torch.randn(rope.shape, generator=g) * 0.5).to(torch.bfloat16))
    data[..., 448:576] = rope.view(torch.uint8)
    return raw.view(num_pages, page_size, 1, 584).cuda()


def make_inputs(B, H, topk, n_tokens, g):
    q = (torch.randn(B, 1, H, 512, generator=g) * 0.5).to(torch.bfloat16)
    idx = torch.randint(0, n_tokens, (B, 1, topk), generator=g, dtype=torch.int32)
    idx[:, :, -3:] = -1
    lens = torch.randint(0, topk + 1, (B,), generator=g, dtype=torch.int32)
    lens[0] = 0  # an empty scope row
    return q.cuda(), idx.cuda(), lens.cuda()


def call(on, args):
    fm._fused_merge_on = (lambda dev: True) if on else (lambda dev: False)
    return fm.flash_mla_sparse_decode_triton(**args)


def same(a, b):
    (oa, la), (ob, lb) = a, b
    return torch.equal(oa.view(torch.int16), ob.view(torch.int16)) and torch.equal(
        la.contiguous().view(torch.int32), lb.contiguous().view(torch.int32)
    )


def run():
    gate = fm._fused_merge_on
    g = torch.Generator().manual_seed(11)
    page_size, pages = 64, 32
    swa = make_cache(pages, page_size, g)
    extra = make_cache(pages, page_size, g)
    bad = 0
    try:
        for B, H in [(1, 64), (4, 64), (13, 64)]:
            q, i1, t1 = make_inputs(B, H, 128, pages * page_size, g)
            _, i2, t2 = make_inputs(B, H, 512, pages * page_size, g)
            sink = torch.randn(H, generator=g).cuda()
            for use_extra, use_sink in [(True, True), (True, False), (False, True), (False, False)]:
                args = dict(
                    q=q, k_cache=swa, indices=i1, topk_length=t1,
                    attn_sink=sink if use_sink else None, head_dim_v=512, softmax_scale=None,
                    extra_k_cache=extra if use_extra else None,
                    extra_indices=i2 if use_extra else None,
                    extra_topk_length=t2 if use_extra else None,
                )
                ref = call(False, args)
                got = call(True, args)
                ok = same(ref, got)
                bad += int(not ok)
                print(f"eager B={B} extra={use_extra} sink={use_sink}: bit-identical={ok}")
            # graph replay with new query values, fused path on
            args = dict(q=q, k_cache=swa, indices=i1, topk_length=t1, attn_sink=sink,
                        head_dim_v=512, softmax_scale=None, extra_k_cache=extra,
                        extra_indices=i2, extra_topk_length=t2)
            call(True, args)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out_g, lse_g = call(True, args)
            for _ in range(2):
                q.copy_((torch.randn(q.shape, generator=g) * 0.5).to(torch.bfloat16))
                graph.replay()
                torch.cuda.synchronize()
                ok = same(call(False, args), (out_g, lse_g))
                bad += int(not ok)
                print(f"graph B={B}: bit-identical={ok}")
    finally:
        fm._fused_merge_on = gate
    return bad


if __name__ == "__main__":
    nbad = run()
    print("fused_merge_wiring:", "PASS" if nbad == 0 else f"FAIL ({nbad})")
    raise SystemExit(1 if nbad else 0)
