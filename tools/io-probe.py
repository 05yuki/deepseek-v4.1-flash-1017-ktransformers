#!/usr/bin/env python3
"""How the drive under the GGUF answers the reads serve.sh sends it.

  tools/io-probe.py <GGUF dir> <out.json> [--piece-kib 512,2048] [--threads 1,2,4,8,16,32] [--seconds 2]

Reads random aligned pieces of the largest shards with O_DIRECT (the page cache is not involved, as in
the expert cache and the streamed prefill), from 1, 2, 4, ... threads at a time, and writes for each
piece size and thread count the rate (GB/s), the mean time of one read (ms) and the CPU time the
reading threads took (CPU seconds per GB). `saturating_threads` is the fewest threads that reach 90%
of the best rate; serve.sh sizes its reader pools from it. Takes about a minute; nothing is written to
the drive.
"""

import argparse
import json
import mmap
import os
import random
import resource
import threading
import time
from pathlib import Path


def block_device(path):
    """The block device a path lives on, its NUMA node and model, from sysfs (None when unknown)."""
    st = os.stat(path)
    dev = f"{os.major(st.st_dev)}:{os.minor(st.st_dev)}"
    sys_dev = Path(f"/sys/dev/block/{dev}").resolve()
    disk = sys_dev.parent if (sys_dev / "partition").exists() else sys_dev
    info = {"device": disk.name, "numa_node": None, "model": None}
    for p in (disk / "device" / "numa_node", disk / "device" / "device" / "numa_node"):
        if p.exists():
            node = int(p.read_text())
            info["numa_node"] = node if node >= 0 else None
            break
    m = disk / "device" / "model"
    if m.exists():
        info["model"] = m.read_text().strip()
    return info


def run(files, piece, threads, seconds):
    fds = [(os.open(f, os.O_RDONLY | os.O_DIRECT), os.path.getsize(f)) for f in files]
    stop = time.monotonic() + seconds
    counts = [0] * threads
    busy = [0.0] * threads

    def work(i):
        buf = mmap.mmap(-1, piece)  # page-aligned, as O_DIRECT needs
        rng = random.Random(i)
        n, t_busy = 0, 0.0
        while time.monotonic() < stop:
            fd, size = fds[rng.randrange(len(fds))]
            off = rng.randrange(0, (size - piece) // piece) * piece
            t = time.monotonic()
            os.preadv(fd, [buf], off)
            t_busy += time.monotonic() - t
            n += 1
        counts[i], busy[i] = n, t_busy
        buf.close()

    r0 = resource.getrusage(resource.RUSAGE_SELF)
    t0 = time.monotonic()
    ts = [threading.Thread(target=work, args=(i,)) for i in range(threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    wall = time.monotonic() - t0
    r1 = resource.getrusage(resource.RUSAGE_SELF)
    for fd, _ in fds:
        os.close(fd)
    n = sum(counts)
    gb = n * piece / 1e9
    cpu = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
    return {
        "threads": threads,
        "gb_per_s": round(gb / wall, 3),
        "read_ms": round(1000 * sum(busy) / max(n, 1), 3),
        "cpu_s_per_gb": round(cpu / max(gb, 1e-9), 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf_dir")
    ap.add_argument("out")
    ap.add_argument("--piece-kib", default="512,2048")
    ap.add_argument("--threads", default="1,2,4,8,16,32")
    ap.add_argument("--seconds", type=float, default=2.0)
    a = ap.parse_args()
    files = sorted(Path(a.gguf_dir).glob("**/*.gguf"), key=lambda p: p.stat().st_size, reverse=True)[:8]
    if not files:
        raise SystemExit(f"no .gguf files under {a.gguf_dir}")
    result = {"path": str(Path(a.gguf_dir).resolve()), **block_device(files[0]), "pieces": {}}
    for kib in (int(k) for k in a.piece_kib.split(",")):
        curve = [run([str(f) for f in files], kib * 1024, int(t), a.seconds) for t in a.threads.split(",")]
        best = max(c["gb_per_s"] for c in curve)
        sat = next(c["threads"] for c in curve if c["gb_per_s"] >= 0.9 * best)
        result["pieces"][str(kib)] = {"curve": curve, "best_gb_per_s": best, "saturating_threads": sat}
        for c in curve:
            print(f"{kib:5d} KiB x {c['threads']:2d} threads: {c['gb_per_s']:6.2f} GB/s, "
                  f"{c['read_ms']:6.2f} ms a read, {c['cpu_s_per_gb']:.3f} CPU s a GB", flush=True)
        print(f"{kib:5d} KiB: best {best:.2f} GB/s, 90% of it from {sat} threads", flush=True)
    Path(a.out).write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
