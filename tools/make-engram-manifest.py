"""Write the engram manifest the NVMe adapter (engram-adapter/) reads: for every layer
with an engram table, the checkpoint shard holding it and the byte offsets of its FP8
rows and their E8M0 scales. The adapter then reads rows straight out of the
checkpoint, with no extracted copy.

Only the safetensors headers are read.

usage: python make-engram-manifest.py <model dir> <out dir>
       (writes <out dir>/engram-manifest.json; set DSV41_ENGRAM_DIR to <out dir>)
"""
import json
import re
import struct
import sys
from pathlib import Path

model, out = Path(sys.argv[1]), Path(sys.argv[2])
weight_map = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
layers = {}
for key, shard in sorted(weight_map.items()):
    m = re.fullmatch(r"layers\.(\d+)\.engram\.embed\.weight", key)
    if not m:
        continue
    layer = m.group(1)
    scale_key = f"layers.{layer}.engram.embed.scale"
    if weight_map.get(scale_key) != shard:
        raise SystemExit(f"{scale_key} is not in {shard}")
    with open(model / shard, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    w, s = header[key], header[scale_key]
    rows, dim = w["shape"]
    blocks = s["shape"][1]
    if s["shape"][0] != rows or dim % blocks:
        raise SystemExit(f"layer {layer}: weight {w['shape']} and scale {s['shape']} do not pair")
    base = 8 + n
    layers[layer] = {
        "file": shard,
        "rows": rows,
        "dim": dim,
        "blocks": blocks,
        "block_size": dim // blocks,
        "weight_dtype": w["dtype"],
        "scale_dtype": s["dtype"],
        "weight_offset": base + w["data_offsets"][0],
        "scale_offset": base + s["data_offsets"][0],
        "weight_bytes": w["data_offsets"][1] - w["data_offsets"][0],
        "scale_bytes": s["data_offsets"][1] - s["data_offsets"][0],
    }
if not layers:
    raise SystemExit(f"no engram tables in {model}")
out.mkdir(parents=True, exist_ok=True)
manifest = {"source": str(model.resolve()), "base_dir": str(model.resolve()), "layers": layers}
(out / "engram-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"{out / 'engram-manifest.json'}: layers {', '.join(layers)}")
