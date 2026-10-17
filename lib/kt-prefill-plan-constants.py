#!/usr/bin/env python3
"""The prefill plan's constants from a setup's prefill-lend records
(start-qwen38-next-nvidia-v0519.sh, KT_PREFILL_PLAN=1). Prints

  <free MiB at a window's start, at the cap chunk> <KiB a row> <FIXED_MiB,ROW_B,SEQ_B,CARRY_MiB or -> <delta MiB>

or nothing when the setup has no measurement yet.

- KiB a row (the older model): the largest allocated bytes a row in the hint
  files x 1.10 for the allocator's gaps, or the lend's reserved bytes a row
  (bytes_per_token) where that is more.
- The cost model, from the per-forward records (<hint>-rank<r>.windows.jsonl):

    RESV_FACTOR x allocated above a window's start
        = FIXED + ROW * rows + SEQ * sequence end + CARRY * [not the window's first forward]

  by least squares over every forward of every rank, then FIXED raised by the
  KT_PREFILL_PLAN_ENVELOPE quantile of the residuals (0.95). Allocated, not
  reserved: the reserved peak carries whatever torch's cache had grown to
  earlier in the window (1.05-1.65 x allocated on Flash-Next) and a tight
  window gets that back (34,816 rows: 7,192 allocated, 7,356 reserved, 7,780
  free with the loan); the plan's margin (KT_PREFILL_PLAN_MARGIN_MB, 128 in the
  launcher) covers that gap. RESV_FACTOR (1.00) scales the target if needed.
  CARRY is what a later chunk of the same window holds from the earlier ones
  (Flash-Next: ~250 MiB).
  Used only with 24 forwards or more over rows that at least double, a per-row
  cost within 0.7-1.5 x the lend's measured allocated bytes a row, and a
  sequence term under a quarter of the row term; else "-" (the per-row model).
- The free memory is the fallback for the plan's first window only (the plan
  reads it live afterwards): the smallest recorded, less the chunk-sized scratch
  the cap adds over the chunk it was measured at.

  kt-prefill-plan-constants.py <hint prefix> <cap chunk> [--show]
"""
import glob
import json
import os
import sys

prefix, cap = sys.argv[1], int(sys.argv[2])
show = "--show" in sys.argv
RESV_FACTOR = float(os.environ.get("KT_PREFILL_PLAN_RESV_FACTOR", "1.00"))
# FIXED is raised by this quantile of the residuals (1 = the largest)
ENVELOPE = float(os.environ.get("KT_PREFILL_PLAN_ENVELOPE", "0.95"))
recs = [json.load(open(f)) for f in sorted(glob.glob(prefix + "-rank*.json"))]
recs = [r for r in recs if r.get("alloc_per_row") and r.get("current_chunk")]
fws = []
for f in sorted(glob.glob(prefix + "-rank*.windows.jsonl")):
    seen = set()
    for line in open(f):
        try:
            w = json.loads(line)
        except Exception:
            continue
        # a window's forwards are written in order: the first one of each window
        # (launches restart the window count, so key on the free memory too)
        key = (w.get("window"), w.get("free"))
        w["later"] = 1.0 if key in seen else 0.0
        seen.add(key)
        fws.append(w)
fws = [w for w in fws if w.get("rows", 0) >= 1024 and w.get("alloc", 0) > 0]
if not recs and not fws:
    sys.exit(0)

free = None
if recs:
    free = min(r["free_at_prefill_mb"] - r.get("scratch_chunk_mb", 0) * max(0.0, cap / r["current_chunk"] - 1)
               for r in recs)
if fws:
    f2 = min(w["free"] for w in fws) / 2**20
    free = f2 if free is None else min(free, f2)
# allocated x 1.10, or the lend's own reserved bytes a row where that is more:
# without the fitted plane this is all the plan has, and on GLM-5.3 the prefill's
# reserved growth ran 1.33 x its allocated (430 against 322 KB a row), which the
# scratch's own allocator does not hand back (a 2 x 18,432 plan ran out of memory)
row_kb = max(max(r["alloc_per_row"] * 1.10, r.get("bytes_per_token", 0)) for r in recs) / 1024 if recs else 0
# the plane's check against the measure stays on allocated bytes, which it fits
alloc_kb = max(r["alloc_per_row"] for r in recs) * 1.10 / 1024 if recs else 0


def solve(m, v):
    """Gaussian elimination; None when singular."""
    n = len(v)
    a = [row[:] + [x] for row, x in zip(m, v)]
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(a[r][i]))
        if abs(a[p][i]) < 1e-9:
            return None
        a[i], a[p] = a[p], a[i]
        for r in range(n):
            if r != i:
                k = a[r][i] / a[i][i]
                a[r] = [x - k * y for x, y in zip(a[r], a[i])]
    return [a[i][n] / a[i][i] for i in range(n)]


cost = "-"
if len(fws) >= 8 and len({w["rows"] for w in fws}) >= 2 and len({w["seq"] for w in fws}) >= 2:
    has_later = any(w["later"] for w in fws) and not all(w["later"] for w in fws)
    X = [(1.0, float(w["rows"]), float(w["seq"])) + ((w["later"],) if has_later else ()) for w in fws]
    y = [float(w["alloc"]) * RESV_FACTOR for w in fws]
    k = len(X[0])
    xtx = [[sum(x[i] * x[j] for x in X) for j in range(k)] for i in range(k)]
    xty = [sum(x[i] * yy for x, yy in zip(X, y)) for i in range(k)]
    b = solve(xtx, xty)
    if b is not None and b[1] > 0:
        if not has_later:
            b.append(0.0)
        b[2], b[3] = max(0.0, b[2]), max(0.0, b[3])
        res = [yy - (b[0] + b[1] * x[1] + b[2] * x[2] + (b[3] * x[3] if has_later else 0.0)) for x, yy in zip(X, y)]
        srt = sorted(res)
        env = srt[-1] if ENVELOPE >= 1 else srt[min(len(srt) - 1, int(ENVELOPE * len(srt)))]
        b[0] += max(0.0, env)
        # trust the plane only where the records can carry it: enough forwards
        # over a wide range of rows, a per-row cost near the lend's own measure,
        # and a sequence term that stays a correction (a crashed run's few
        # forwards gave 168 KiB a row and 79 KB a sequence token, and the plan
        # put a 35K prompt in one chunk that did not fit, 10-15)
        rows_seen = sorted(w["rows"] for w in fws)
        why = None
        if len(fws) < 24:
            why = f"{len(fws)} forwards"
        elif rows_seen[-1] < 2 * rows_seen[0]:
            why = f"rows only {rows_seen[0]}-{rows_seen[-1]}"
        elif alloc_kb and not 0.7 * alloc_kb * 1024 / 1.10 <= b[1] <= 1.5 * alloc_kb * 1024:
            why = f"{b[1] / 1024:.0f} KiB a row against {alloc_kb / 1.10:.0f} measured"
        elif b[2] * max(w["seq"] for w in fws) > 0.25 * b[1] * rows_seen[-1]:
            why = f"sequence term {b[2]:.0f} B a token too large"
        if why is None:
            cost = f"{b[0] / 2**20:.0f},{b[1]:.0f},{b[2]:.0f},{b[3] / 2**20:.0f}"
        elif show:
            print(f"  plane not used ({why}): the per-row model stands", file=sys.stderr)
        if show:
            for w, r in sorted(zip(fws, res), key=lambda t: (t[0]["rows"], t[0]["seq"])):
                print(f"  rows {w['rows']:6d} seq {w['seq']:7d} {'later' if w['later'] else 'first'} free "
                      f"{w['free'] >> 20:5d} MiB loan {w.get('loan', 0) >> 20:5d} MiB allocated {w['alloc'] >> 20:5d} MiB "
                      f"(x{RESV_FACTOR} fit {(w['alloc'] * RESV_FACTOR - r) / 2**20:7.0f}, residual {r / 2**20:+6.0f})",
                      file=sys.stderr)
            print(f"  {len(fws)} forwards: fixed {b[0] / 2**20:.0f} MiB (envelope {env / 2**20:.0f}, q {ENVELOPE}), "
                  f"{b[1] / 1024:.1f} KiB a row, {b[2]:.0f} B a sequence token, {b[3] / 2**20:.0f} MiB carried "
                  f"by a later chunk", file=sys.stderr)
# the decode -> window-start step of free memory (the smallest the ranks recorded)
deltas = [r["pf_delta_mb"] for r in recs if r.get("pf_delta_mb")]
delta = min(deltas) if deltas else 0
print(int(free), int(row_kb) + 1 if row_kb else 0, cost, delta)
