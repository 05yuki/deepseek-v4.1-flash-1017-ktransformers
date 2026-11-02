#!/usr/bin/env python3
"""Export a setup's prefill-lend chunk hint as a shipped hint (hints/kt-lend/), so the first launch on
the same GPU model and count skips the chunk calibration (lib/kt-lend-auto.sh seeds it).

  tools/lend-hint-export.py <hint key> [--recipe <tag>] [--note <text>] [--out <file>]

<hint key> is what `KT_LEND_PRINT_KEY=1 ./serve.sh` prints for the setup after it was calibrated and has
served a long prompt (the server rewrites the rank files every prefill window). The rank files come from
KT_LEND_CACHE_DIR (default ~/.cache/kt-lend); the GPU name and memory from nvidia-smi for the GPUs in
CUDA_VISIBLE_DEVICES (all of one model). The file holds no path, host or user name.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("key")
ap.add_argument("--recipe", default="", help="the recipe tag the hint was measured with")
ap.add_argument("--note", default="", help="free text: the host's CPU, anything a reader should know")
ap.add_argument("--out", default="", help="default hints/kt-lend/<gpu slug>-x<gpus>.json under the checkout")
args = ap.parse_args()

cache = os.environ.get("KT_LEND_CACHE_DIR") or os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "kt-lend")
prefix = os.path.join(cache, f"chunk-{args.key}")
files = sorted(glob.glob(prefix + "-rank*.json"))
if not files:
    sys.exit(f"no {prefix}-rank*.json (calibrate and serve a long prompt first)")
ranks = []
for f in files:
    r = json.load(open(f))
    for k in ("chunk", "alloc_per_row", "free_at_prefill_mb", "bytes_per_token", "current_chunk", "full_tokens"):
        if not r.get(k):
            sys.exit(f"{f}: {k} missing or 0; the setup has not served a long prompt since the calibration")
    ranks.append(r)

sel = os.environ.get("CUDA_VISIBLE_DEVICES")
cmd = ["nvidia-smi"] + (["-i", sel] if sel else []) + ["--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]
gpus = [l.split(", ") for l in subprocess.check_output(cmd, text=True).strip().splitlines()]
names = {g[0].strip() for g in gpus}
if len(names) != 1:
    sys.exit(f"the GPUs are not of one model: {sorted(names)}")
if len(gpus) != len(ranks):
    sys.exit(f"{len(gpus)} GPUs selected, {len(ranks)} rank files")
name = names.pop()
vram = min(int(g[1]) for g in gpus)
slug = re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", name.lower())).strip("-")
out = args.out or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hints", "kt-lend",
                                f"{slug}-{round(vram / 1024)}g-x{len(ranks)}.json")
doc = {
    "gpu": name, "vram_mib": vram, "gpus": len(ranks),
    "measured": {"date": time.strftime("%Y-%m-%d"), "recipe": args.recipe, "note": args.note},
    "ranks": ranks,
}
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    json.dump(doc, f, indent=1)
    f.write("\n")
print(f"{out}: {name} x{len(ranks)}, {vram} MiB, chunk {min(r['chunk'] for r in ranks)}, "
      f"free {min(r['free_at_prefill_mb'] for r in ranks)} MiB at prefill")
