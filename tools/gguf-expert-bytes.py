#!/usr/bin/env python3
"""The routed experts' bytes in a GGUF build, by GGUF type, for serve.sh's expert cache size: the raw rows
(blk.<L>.ffn_{gate,up,down}_exps) and what the IQ1_M decode copy (KT_GGUF_CACHE_IQ1M_T, 76 bytes a block of
256 beside the raw 56) adds.
usage: gguf-expert-bytes.py <gguf dir> <out json>"""
import glob
import json
import os
import re
import sys

from gguf import GGUFReader

gguf_dir, out = sys.argv[1], sys.argv[2]
by_type = {}
for path in sorted(glob.glob(os.path.join(gguf_dir, "*.gguf"))):
    for t in GGUFReader(path).tensors:
        if re.match(r"blk\.\d+\.ffn_(gate|up|down)_exps\.weight$", t.name):
            by_type[t.tensor_type.name] = by_type.get(t.tensor_type.name, 0) + int(t.n_bytes)
assert by_type, f"no routed expert tensors in {gguf_dir}"
raw = sum(by_type.values())
iq1m_copy = by_type.get("IQ1_M", 0) // 56 * 76
res = {"source": os.path.abspath(gguf_dir), "bytes_by_type": by_type, "raw_bytes": raw, "iq1m_copy_bytes": iq1m_copy}
with open(out, "w") as f:
    json.dump(res, f, indent=1)
gib = 1 << 30
print(f"{out}: experts {raw / gib:.1f} GiB ("
      + ", ".join(f"{k} {v / gib:.1f}" for k, v in sorted(by_type.items())) + f"), IQ1_M decode copy {iq1m_copy / gib:.1f} GiB")
