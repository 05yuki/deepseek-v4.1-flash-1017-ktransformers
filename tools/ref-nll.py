"""Per-token log-probabilities from DeepSeek-V4.1-Flash-1017's own reference implementation
(the checkpoint's inference/model.py), streamed one layer at a time through one 16 GB GPU.

  python tools/ref-nll.py <model dir> <out.json>

It scores tools/nll_pieces.py's 24 pieces in one batch, so every layer's weights are read once
(about 4 minutes on an RTX 5070 Ti, peak 13.3 GiB). The output has tools/nll.py's format:

  python tools/nll.py compare ref.json served.json

What differs from running model.py directly, and why:
  - sparse_attn: model.py's TileLang kernel wants 141 KB of shared memory, more than a consumer
    Blackwell GPU has (99 KB). The same arithmetic runs in fp32 torch instead: a softmax over the
    selected positions plus the sink, with -1 slots masked.
  - The engram tables (about 95 GB each) stay on disk. Rows are gathered from a memory map of the
    safetensors file and dequantized exactly as ParallelEngramEmbedding does.
  - wo_a is dequantized to bf16 at load, as the checkpoint's convert.py does. The routed experts are
    viewed as fp4, as convert.py --expert-dtype fp4 does.
Everything else is model.py unchanged, at world size 1. kernel.py's other TileLang kernels (the FP8 and
FP4 GEMMs, the quantizers) compile at first use, so CUDA_HOME has to point at a CUDA toolkit.
"""

import json
import os
import struct
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

MODEL_DIR = sys.argv[1]
OUT = sys.argv[2]
sys.path.insert(0, os.path.join(MODEL_DIR, "inference"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M  # noqa: E402
import nll_pieces  # noqa: E402

ATTN_CHUNK = 64


def sparse_attn_torch(q, kv, attn_sink, topk_idxs, softmax_scale):
    b, s, h, d = q.shape
    o = torch.empty_like(q)
    for bi in range(b):
        for c0 in range(0, s, ATTN_CHUNK):
            idx = topk_idxs[bi, c0:c0 + ATTN_CHUNK].long()  # [c, k]
            valid = idx >= 0
            kvg = kv[bi][idx.clamp_min(0)].float()  # [c, k, d]
            sc = torch.einsum("chd,ckd->chk", q[bi, c0:c0 + ATTN_CHUNK].float(), kvg) * softmax_scale
            sc.masked_fill_(~valid[:, None, :], -torch.inf)
            sink = attn_sink.float().view(1, h, 1).expand(sc.size(0), h, 1)
            p = torch.cat([sc, sink], dim=-1).softmax(dim=-1)[..., :-1]
            o[bi, c0:c0 + ATTN_CHUNK] = torch.einsum("chk,ckd->chd", p, kvg).to(q.dtype)
    return o


M.sparse_attn = sparse_attn_torch


class Checkpoint:
    def __init__(self, path):
        self.path = path
        self.wm = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
        self.handles = {}

    def get(self, name):
        from safetensors import safe_open

        f = self.wm[name]
        if f not in self.handles:
            self.handles[f] = safe_open(os.path.join(self.path, f), framework="pt", device="cpu")
        return self.handles[f].get_tensor(name)

    def memmap(self, name):
        path = os.path.join(self.path, self.wm[name])
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            meta = json.loads(fh.read(n))[name]
        start, end = meta["data_offsets"]
        rows = meta["shape"][0]
        return np.memmap(path, dtype=np.uint8, mode="r", offset=8 + n + start, shape=(rows, (end - start) // rows))


CK = Checkpoint(MODEL_DIR)


class MemmapEngramEmbedding(torch.nn.Module):
    """ParallelEngramEmbedding (world size 1) with the table left in the checkpoint file."""

    current = None  # layer id, set before each Engram is built

    def __init__(self, num_embeddings, dim):
        super().__init__()
        layer = MemmapEngramEmbedding.current
        self.block_size = M.fp8_block_size
        self.weight = CK.memmap(f"layers.{layer}.engram.embed.weight")
        self.scale = CK.memmap(f"layers.{layer}.engram.embed.scale")
        assert self.weight.shape == (num_embeddings, dim), (self.weight.shape, num_embeddings, dim)

    def forward(self, indices):
        flat = indices.flatten().cpu().numpy()
        order = np.argsort(flat, kind="stable")  # sorted reads, then put back
        w = np.empty((flat.size, self.weight.shape[1]), np.uint8)
        s = np.empty((flat.size, self.scale.shape[1]), np.uint8)
        w[order] = self.weight[flat[order]]
        s[order] = self.scale[flat[order]]
        dev = indices.device
        values = torch.from_numpy(w).to(dev).view(torch.float8_e4m3fn).view(*indices.shape, -1)
        scales = torch.from_numpy(s).to(dev).view(torch.float8_e8m0fnu).view(*indices.shape, -1)
        values = values.float().unflatten(-1, (-1, self.block_size)) * scales.float().unsqueeze(-1)
        return values.flatten(-2).to(torch.bfloat16)


M.ParallelEngramEmbedding = MemmapEngramEmbedding


def load_module(mod, prefix):
    for name, p in list(mod.named_parameters()):
        t = CK.get(prefix + name)
        if name.endswith("wo_a.weight"):  # convert.py: dequantize to bf16
            s = CK.get(prefix + name.replace("weight", "scale"))
            ob, ib = t.size(0) // s.size(0), t.size(1) // s.size(1)
            t = (t.unflatten(0, (-1, ob)).unflatten(-1, (-1, ib)).float() * s[:, None, :, None].float())
            t = t.flatten(2, 3).flatten(0, 1).bfloat16()
        elif p.dtype == torch.float4_e2m1fn_x2:
            assert t.dtype == torch.int8, (name, t.dtype)
            t = t.view(torch.float4_e2m1fn_x2)
        assert tuple(t.shape) == tuple(p.shape), (prefix + name, t.shape, p.shape)
        p.data = t.to("cuda") if t.dtype == p.dtype else t.to("cuda").to(p.dtype)


@torch.inference_mode()
def main():
    t0 = time.time()
    tok = nll_pieces.tokenizer(MODEL_DIR)
    ids = nll_pieces.pieces(tok)
    n_pieces, piece = len(ids), len(ids[0])
    cfg = json.load(open(os.path.join(MODEL_DIR, "inference", "config.json")))
    cfg.update(vision_n_layers=0, max_batch_size=n_pieces, max_seq_len=piece)
    args = M.ModelArgs(**cfg)
    M.world_size, M.rank = 1, 0
    M.default_dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")

    layout = M.EngramLayout.from_args(args)
    hasher = M.NgramHashState(args, layout, tok)
    x = torch.tensor(ids, dtype=torch.long)
    engram_hashes = hasher(x, 0)
    h = F.embedding(x.cpu(), CK.get("embed.weight")).to("cuda")
    h = h.unsqueeze(2).repeat(1, 1, args.hc_mult, 1)
    pre_mix = M.make_identity_pre_mix(h, args.hc_mult)

    for i in range(args.n_layers):
        t1 = time.time()
        if i in layout.layer_ids:
            MemmapEngramEmbedding.current = i
            eng = M.Engram(args, i, layout)
            load_module(eng, f"layers.{i}.engram.")
            h = eng(h, engram_hashes[:, :, eng.layer_hash_index, :], None)
            del eng
        blk = M.Block(i, args, None)
        load_module(blk, f"layers.{i}.")
        t2 = time.time()
        h, pre_mix = blk(h, 0, pre_mix, None)
        torch.cuda.synchronize()
        print(f"layer {i}: load {t2 - t1:.1f} s, run {time.time() - t2:.1f} s, "
              f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
        del blk
        torch.cuda.empty_cache()

    h = M.Block.hc_pre(None, h, pre_mix)  # uses no weights
    norm = M.RMSNorm(args.dim, args.norm_eps)
    load_module(norm, "norm.")
    head_w = CK.get("head.weight").to("cuda", torch.float32)
    h = norm(h)
    result = []
    for b in range(n_pieces):
        lp = F.linear(h[b].float(), head_w).log_softmax(-1)  # [piece, vocab]
        tgt = x[b, 1:].to("cuda")
        result.append(lp[:-1].gather(1, tgt[:, None]).squeeze(1).tolist())
    json.dump(result, open(OUT, "w"))
    n = sum(len(r) for r in result)
    print("scored %d tokens in %d pieces, mean NLL %.4f (%.0f s)"
          % (n, len(result), -sum(sum(r) for r in result) / n, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
