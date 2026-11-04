#!/usr/bin/env python3
"""An engram manifest for the V4.1 recipe's adapter from a llama.cpp "deepseek41" GGUF directory, either layout of
the GSQ-RCO pack:
- the earlier one (before 2026-10-27): each blk.<L>.engram_embd table (IQ4_NL, [256, rows] in ggml order) as layer L;
- llama.cpp's (2026-10-27 and later): one per_layer_token_embd table holding the layers' rows one after another, the
  layers and their bucket counts in the first shard's metadata (deepseek41.ple.*).
The rows are read in place from the GGUF file (row_bytes 144). A file seen under two names (the shard-6 link to the
Engram file) counts once.
usage: make-engram-manifest-gguf.py <gguf dir> <out dir>"""
import glob, json, os, re, sys

from gguf import GGUFReader

ROW_BYTES = 144  # IQ4_NL, 256 values a row


def field(r, key):
    f = r.fields.get(key)
    if f is None:
        return None
    vals = [f.parts[i] for i in f.data]
    out = [v.tolist()[0] if hasattr(v, "tolist") else v for v in vals]
    return out if len(out) != 1 or key.endswith(("layers", "sizes", "offsets", "multipliers")) else out[0]


def entry(path, offset, rows):
    return {
        "file": os.path.basename(path), "rows": rows, "dim": 256, "blocks": 8, "block_size": 32,
        "weight_dtype": "IQ4_NL", "row_bytes": ROW_BYTES, "scale_dtype": None,
        "weight_offset": int(offset), "scale_offset": 0, "weight_bytes": rows * ROW_BYTES, "scale_bytes": 0,
    }


gguf_dir, out_dir = sys.argv[1], sys.argv[2]
seen, readers = set(), []
# real files before links, so a table is named by the file that holds it
paths = sorted(glob.glob(os.path.join(gguf_dir, "*.gguf")), key=lambda p: (os.path.islink(p), p))
for path in paths:
    real = os.path.realpath(path)
    if real not in seen:
        seen.add(real)
        readers.append((path, GGUFReader(path)))

layers, table, ple = {}, None, None
for path, r in readers:
    if field(r, "deepseek41.ple.layers") is not None:
        ple = r
    for t in r.tensors:
        if t.name == "per_layer_token_embd.weight":
            table = (path, t)
        m = re.match(r"blk\.(\d+)\.engram_embd\.weight$", t.name)
        if not m:
            continue
        dim, rows = (int(x) for x in t.shape.tolist())  # ggml order: [256, rows]
        assert t.tensor_type.name == "IQ4_NL" and dim == 256, (t.name, t.tensor_type.name, dim)
        assert t.n_bytes == rows * ROW_BYTES, (t.name, t.n_bytes, rows)
        layers[m.group(1)] = entry(path, t.data_offset, rows)

if not layers and table is not None:
    path, t = table
    assert ple is not None, f"{gguf_dir}: per_layer_token_embd without the first shard's deepseek41.ple metadata"
    dim, total = (int(x) for x in t.shape.tolist())
    assert t.tensor_type.name == "IQ4_NL" and dim == 256 and t.n_bytes == total * ROW_BYTES, \
        (t.tensor_type.name, dim, t.n_bytes, total)
    ids = field(ple, "deepseek41.ple.layers")
    sizes = field(ple, "deepseek41.ple.head_vocab_sizes")
    offsets = field(ple, "deepseek41.ple.head_offsets")
    n_cols = (field(ple, "deepseek41.ple.ngram_size") - 1) * field(ple, "deepseek41.ple.heads_per_ngram")
    assert len(sizes) == len(ids) * n_cols and len(offsets) == len(sizes), (len(ids), n_cols, len(sizes))
    start = 0
    for e, layer in enumerate(ids):
        rows = sum(sizes[e * n_cols:(e + 1) * n_cols])
        assert offsets[e * n_cols] == start, (layer, offsets[e * n_cols], start)  # the layer's first row
        layers[str(layer)] = entry(path, t.data_offset + start * ROW_BYTES, rows)
        start += rows
    assert start == total, (start, total)

assert layers, f"no Engram tables (blk.N.engram_embd or per_layer_token_embd) in {gguf_dir}"
os.makedirs(out_dir, exist_ok=True)
manifest = {"source": os.path.abspath(gguf_dir), "base_dir": os.path.abspath(gguf_dir), "layers": layers}
with open(os.path.join(out_dir, "engram-manifest.json"), "w") as f:
    json.dump(manifest, f, indent=1)
print(f"{out_dir}/engram-manifest.json: layers {sorted(layers, key=int)} from {gguf_dir}")
