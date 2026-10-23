"""kt_service_affinity: CPU only, no model server, no GPU.

    python -I tests/test_kt_service_affinity.py            (from affinity_implementation/)
    KT_SERVICE_AFFINITY_HELPER=<path> python -I <this file>  (another copy of the helper)

Most cases drive the helper through FakeThreads, a stand-in for /proc/self/task and
sched_{get,set}affinity that records every write. On Linux, test_real_threads also runs
the helper in a child process on its own named helper threads, with CPUs taken from the
child's own allowed mask; it is skipped elsewhere or with fewer than 3 allowed CPUs.
"""
import importlib.util
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

HELPER = Path(
    os.environ.get(
        "KT_SERVICE_AFFINITY_HELPER",
        Path(__file__).resolve().parents[2] / "third_party/sglang/python/sglang/srt/utils/kt_service_affinity.py",
    )
)
_spec = importlib.util.spec_from_file_location("kt_service_affinity", HELPER)
ksa = importlib.util.module_from_spec(_spec)
sys.modules["kt_service_affinity"] = ksa
_spec.loader.exec_module(ksa)

LAUNCH = frozenset(range(64))
SERVICE = frozenset({40, 41})
ARGS = SimpleNamespace(kt_cpuinfer=6, kt_threadpool_count=2)
ONE_RANK = SimpleNamespace(tp_size=1, pp_size=1, num_dp_ranks=1)
# two pools of three expert workers, as worker_pool.cpp names them, plus service threads
EXPERTS = {
    11: ("numa_0_m_0", {2}),
    12: ("numa_0_t_1", {3}),
    13: ("numa_0_t_2", {4}),
    21: ("numa_1_m_0", {10}),
    22: ("numa_1_t_1", {11}),
    23: ("numa_1_t_2", {12}),
}
SERVICES = {
    100: ("python", set(range(0, 8))),
    101: ("pt_main_thread", set(range(0, 8))),
    102: ("cuda-EvtHandlr", set(range(0, 8))),
}


class FakeThreads:
    """tid -> [start, comm, cpus]; before_write runs before every write and may mutate
    the table (another thread acting concurrently)."""

    def __init__(self, rows=None):
        rows = rows if rows is not None else {**EXPERTS, **SERVICES}
        self.rows = {tid: [1000 + tid, comm, frozenset(cpus)] for tid, (comm, cpus) in rows.items()}
        self.writes = []
        self.before_write = None
        self.fail_write = set()
        self.unstable = False

    def tids(self):
        if self.unstable:
            self.rows[max(self.rows) + 1] = [7, "spawner", frozenset({0})]
        return sorted(self.rows)

    def identity(self, tid):
        if tid not in self.rows:
            raise FileNotFoundError(tid)
        return self.rows[tid][0], self.rows[tid][1]

    def get(self, tid):
        if tid not in self.rows:
            raise ProcessLookupError(tid)
        return self.rows[tid][2]

    def set(self, tid, cpus):
        self.writes.append((tid, frozenset(cpus)))
        if self.before_write:
            self.before_write(self, tid)
        if tid in self.fail_write:
            raise OSError(22, "Invalid argument")
        if tid not in self.rows:
            raise ProcessLookupError(tid)
        self.rows[tid][2] = frozenset(cpus)

    def masks(self):
        return {tid: row[2] for tid, row in self.rows.items()}


def plan(**kw):
    env = {"KT_SERVICE_CPUS": "40-41"}
    env.update(kw.pop("environ", {}))
    return ksa.plan_service_affinity(
        kw.pop("server_args", ARGS),
        kw.pop("parallel", ONE_RANK),
        launch_mask_known=kw.pop("launch_mask_known", True),
        environ=env,
        getaffinity=kw.pop("getaffinity", lambda pid: LAUNCH),
        platform=kw.pop("platform", "linux"),
    )


def no_affinity_read(pid):
    raise AssertionError("the launcher mask must not be read when KT_SERVICE_CPUS is unset")


class OptIn(unittest.TestCase):
    def test_unset_is_a_no_op(self):
        # nothing is read: not the platform, the ranks, the server args or the CPU mask
        for env in ({}, {"KT_GPU_GLUE": "1"}, {"KT_GPU_GLUE": "0"}):
            got = ksa.plan_service_affinity(
                None, None, launch_mask_known=False, environ=env,
                getaffinity=no_affinity_read, platform="win32",
            )
            self.assertIsNone(got, env)

    def test_independent_of_gpu_glue(self):
        for glue in ("0", "1"):
            p = plan(environ={"KT_GPU_GLUE": glue})
            self.assertEqual(p.service_cpus, SERVICE)
            self.assertEqual(p.launch_cpus, LAUNCH)

    def test_service_cpus_may_leave_the_post_bind_mask(self):
        # 40-41 is outside a NUMA-bound 0-7 but inside the launcher's mask: accepted
        self.assertEqual(plan().service_cpus, SERVICE)


class InvalidPlans(unittest.TestCase):
    def refused(self, **kw):
        with self.assertRaises(ksa.ServiceAffinityError) as ctx:
            plan(**kw)
        return str(ctx.exception)

    def test_bad_lists(self):
        for text in ("", " ", "a", "5-3", "1,,2", "-1", "1-", "1-2-3", "0x3", "1;2", "+4"):
            self.refused(environ={"KT_SERVICE_CPUS": text})

    def test_good_lists(self):
        self.assertEqual(ksa.parse_cpu_list("3"), {3})
        self.assertEqual(ksa.parse_cpu_list("1-3, 7"), {1, 2, 3, 7})
        self.assertEqual(ksa.fmt({0, 1, 2, 5, 7, 8}), "0-2,5,7-8")

    def test_outside_launcher_mask(self):
        msg = self.refused(getaffinity=lambda pid: frozenset(range(8)))
        self.assertIn("outside the launcher", msg)

    def test_unsupported_platform(self):
        self.assertIn("Linux", self.refused(platform="win32"))

    def test_more_than_one_rank(self):
        for name in ("tp_size", "pp_size", "num_dp_ranks"):
            parallel = SimpleNamespace(**{**vars(ONE_RANK), name: 2})
            self.assertIn("per-rank", self.refused(parallel=parallel))

    def test_launcher_mask_not_visible(self):
        self.assertIn("NUMA bind v2", self.refused(launch_mask_known=False))

    def test_unknown_rank_count_is_rejected(self):
        self.assertIn("per-rank", self.refused(parallel=SimpleNamespace(tp_size=1, pp_size=1)))

    def test_needs_kt_experts(self):
        for args in (
            SimpleNamespace(),
            SimpleNamespace(kt_cpuinfer=None, kt_threadpool_count=1),
            SimpleNamespace(kt_cpuinfer=0, kt_threadpool_count=0),
            SimpleNamespace(kt_cpuinfer=2, kt_threadpool_count=3),
            SimpleNamespace(kt_cpuinfer=True, kt_threadpool_count=1),
        ):
            self.refused(server_args=args)


class Apply(unittest.TestCase):
    def test_moves_service_threads_and_preserves_experts(self):
        fake = FakeThreads()
        report = ksa.apply_service_affinity(plan(), fake)
        masks = fake.masks()
        for tid, (_, cpus) in EXPERTS.items():
            self.assertEqual(masks[tid], frozenset(cpus))
        for tid in SERVICES:
            self.assertEqual(masks[tid], SERVICE)
        self.assertEqual(sorted(t for t, _ in fake.writes), sorted(SERVICES))
        self.assertEqual(report["expert_threads"], 6)
        self.assertEqual(report["service_threads"], 3)
        self.assertEqual(report["changed"], 3)
        self.assertEqual(report["service_cpus"], "40-41")
        self.assertEqual(report["expert_cpus"], "2-4,10-12")

    def test_idempotent(self):
        fake = FakeThreads()
        ksa.apply_service_affinity(plan(), fake)
        fake.writes.clear()
        self.assertEqual(ksa.apply_service_affinity(plan(), fake)["changed"], 0)
        self.assertEqual(fake.writes, [])

    def test_thread_exiting_before_its_write_is_skipped(self):
        fake = FakeThreads()

        def exit_102(f, tid):
            f.rows.pop(102, None)

        fake.before_write = exit_102
        report = ksa.apply_service_affinity(plan(), fake)
        self.assertEqual(report["service_threads"], 2)
        self.assertNotIn(102, [t for t, _ in fake.writes])

    def test_new_thread_inheriting_the_service_mask_is_accepted(self):
        fake = FakeThreads()

        def spawn(f, tid):
            if tid == 100 and 200 not in f.rows:
                f.rows[200] = [5000, "child", SERVICE]

        fake.before_write = spawn
        self.assertEqual(ksa.apply_service_affinity(plan(), fake)["service_threads"], 4)


class Refusals(unittest.TestCase):
    """Unexpected topology before any write: nothing is written."""

    def refused(self, fake, p=None):
        with self.assertRaises(ksa.ServiceAffinityError) as ctx:
            ksa.apply_service_affinity(p or plan(), fake)
        self.assertEqual(fake.writes, [])
        return str(ctx.exception)

    def test_missing_expert(self):
        rows = {**EXPERTS, **SERVICES}
        del rows[23]
        self.assertIn("5 expert threads", self.refused(FakeThreads(rows)))

    def test_unnamed_expert_is_not_moved(self):
        # pthread_setname_np failed for one worker: it looks like a service thread
        rows = {**EXPERTS, **SERVICES, 23: ("python", {12})}
        self.assertIn("expected 6", self.refused(FakeThreads(rows)))

    def test_extra_pool(self):
        rows = {**EXPERTS, **SERVICES, 31: ("numa_2_m_0", {20})}
        self.refused(FakeThreads(rows))

    def test_wrong_pool_count(self):
        rows = {**EXPERTS, **SERVICES, 21: ("numa_1_t_0", {10})}
        self.assertIn("numa_*_m_*", self.refused(FakeThreads(rows)))

    def test_expert_not_singleton(self):
        # hwloc binding failed (worker_pool.cpp only logs it): the inherited mask remains
        rows = {**EXPERTS, **SERVICES, 12: ("numa_0_t_1", {3, 5})}
        self.assertIn("not one CPU", self.refused(FakeThreads(rows)))

    def test_shared_expert_cpu(self):
        rows = {**EXPERTS, **SERVICES, 12: ("numa_0_t_1", {2})}
        self.assertIn("share a CPU", self.refused(FakeThreads(rows)))

    def test_service_cpus_overlap_experts(self):
        p = plan(environ={"KT_SERVICE_CPUS": "4,40"})
        self.assertIn("are expert CPUs", self.refused(FakeThreads(), p))

    def test_unstable_thread_set(self):
        fake = FakeThreads()
        fake.unstable = True
        self.assertIn("not stable", self.refused(fake))


class Rollback(unittest.TestCase):
    """A failure after a write: every changed thread gets its own mask back, expert
    threads and threads that are not the snapshotted ones are never written."""

    def rolled_back(self, fake, rebound=()):
        original = fake.masks()
        with self.assertRaises(ksa.ServiceAffinityError) as ctx:
            ksa.apply_service_affinity(plan(), fake)
        written = [t for t, _ in fake.writes]
        for tid, (_, cpus) in EXPERTS.items():
            self.assertNotIn(tid, written)
            if tid not in rebound:
                self.assertEqual(fake.masks()[tid], frozenset(cpus))
        for tid in SERVICES:
            if tid in fake.rows and fake.rows[tid][0] == 1000 + tid:
                self.assertEqual(fake.masks()[tid], original[tid], tid)
        self.assertIn("rolled back", str(ctx.exception))
        return str(ctx.exception)

    def test_failed_write(self):
        fake = FakeThreads()
        fake.fail_write = {101}
        msg = self.rolled_back(fake)
        self.assertIn("Invalid argument", msg)
        self.assertIn("rolled back 1 changed", msg)

    def test_tid_reused_before_write(self):
        fake = FakeThreads()

        def reuse_102(f, tid):
            if tid == 101:
                f.rows[102] = [99999, "other", frozenset(range(8))]

        fake.before_write = reuse_102
        msg = self.rolled_back(fake)
        self.assertIn("changed identity", msg)
        self.assertEqual(fake.masks()[102], frozenset(range(8)))  # not ours: untouched
        self.assertEqual(fake.writes.count((102, SERVICE)), 0)

    def test_service_thread_becomes_expert_name(self):
        fake = FakeThreads()

        def rename(f, tid):
            if tid == 100:
                f.rows[102][1] = "numa_0_t_9"

        fake.before_write = rename
        self.assertIn("changed identity", self.rolled_back(fake))

    def test_expert_rebound_during_writes(self):
        fake = FakeThreads()

        def rebind(f, tid):
            f.rows[13][2] = frozenset({5})

        fake.before_write = rebind
        self.assertIn("expert workers changed", self.rolled_back(fake, rebound={13}))

    def test_expert_added_during_writes(self):
        fake = FakeThreads()

        def add(f, tid):
            f.rows.setdefault(300, [1, "numa_1_t_3", frozenset({13})])

        fake.before_write = add
        self.assertIn("expert workers changed", self.rolled_back(fake))
        self.assertNotIn(300, [t for t, _ in fake.writes])

    def test_new_thread_with_old_mask(self):
        fake = FakeThreads()

        def spawn_unmoved(f, tid):
            f.rows.setdefault(201, [6000, "late", frozenset(range(8))])

        fake.before_write = spawn_unmoved
        msg = self.rolled_back(fake)
        self.assertIn("service thread 201", msg)
        self.assertNotIn(201, [t for t, _ in fake.writes])  # mask unknown: never written

    def test_rollback_failure_is_reported(self):
        fake = FakeThreads()
        fake.fail_write = {102}
        calls = []

        def fail_restore_of_100(f, tid):
            # writes: 100, 101 forward, 102 fails, then 100 back (fails here), 101 back
            calls.append(tid)
            if len(calls) == 4:
                f.fail_write.add(100)

        fake.before_write = fail_restore_of_100
        with self.assertRaises(ksa.ServiceAffinityError) as ctx:
            ksa.apply_service_affinity(plan(), fake)
        self.assertEqual(calls, [100, 101, 102, 100, 101])
        self.assertIn("NOT restored", str(ctx.exception))
        self.assertIn("100 (python)", str(ctx.exception))
        self.assertEqual(fake.masks()[101], frozenset(range(8)))

    def test_mixed_original_masks_are_restored(self):
        fake = FakeThreads()
        fake.rows[101][2] = LAUNCH  # early CUDA thread retains the broader mask
        fake.fail_write = {102}
        self.rolled_back(fake)
        self.assertEqual(fake.masks()[101], LAUNCH)


CHILD = r"""
import ctypes, json, os, sys, threading, importlib.util
from types import SimpleNamespace
spec = importlib.util.spec_from_file_location("kt_service_affinity", sys.argv[1])
ksa = importlib.util.module_from_spec(spec); sys.modules[spec.name] = ksa; spec.loader.exec_module(ksa)
libc = ctypes.CDLL(None, use_errno=True)
allowed = sorted(os.sched_getaffinity(0))
expert_cpus, service = allowed[:2], allowed[2:3]
names = ["numa_0_m_0", "numa_0_t_1", "helper"]
pins = [{expert_cpus[0]}, {expert_cpus[1]}, None]
ready, stop, tids = threading.Barrier(4), threading.Event(), {}
def body(name, pin):
    libc.prctl(15, name.encode(), 0, 0, 0)  # PR_SET_NAME, what pthread_setname_np does
    if pin: os.sched_setaffinity(0, pin)
    tids[name] = threading.get_native_id()
    ready.wait(); stop.wait()
threads = [threading.Thread(target=body, args=a, daemon=True) for a in zip(names, pins)]
for t in threads: t.start()
ready.wait()
env = {"KT_SERVICE_CPUS": ",".join(map(str, service))}
plan = ksa.plan_service_affinity(SimpleNamespace(kt_cpuinfer=2, kt_threadpool_count=1),
                                 SimpleNamespace(tp_size=1, pp_size=1, num_dp_ranks=1),
                                 launch_mask_known=True, environ=env)
report = ksa.apply_service_affinity(plan)
masks = {n: sorted(os.sched_getaffinity(t)) for n, t in tids.items()}
masks["main"] = sorted(os.sched_getaffinity(threading.get_native_id()))
late = threading.Thread(target=lambda: masks.__setitem__("late", sorted(os.sched_getaffinity(0))))
late.start(); late.join()
stop.set()
print(json.dumps(dict(report=report, masks=masks, expert_cpus=expert_cpus, service=service)))
"""


@unittest.skipUnless(sys.platform.startswith("linux"), "needs Linux /proc and sched_setaffinity")
class RealThreads(unittest.TestCase):
    def test_real_threads(self):
        if len(os.sched_getaffinity(0)) < 3:
            self.skipTest("needs 3 allowed CPUs")
        out = subprocess.run(
            [sys.executable, "-I", "-c", CHILD, str(HELPER)],
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(out.returncode, 0, out.stderr)
        got = json.loads(out.stdout.strip().splitlines()[-1])
        e, s = got["expert_cpus"], got["service"]
        self.assertEqual(got["masks"]["numa_0_m_0"], [e[0]])
        self.assertEqual(got["masks"]["numa_0_t_1"], [e[1]])
        self.assertEqual(got["masks"]["helper"], s)
        self.assertEqual(got["masks"]["main"], s)
        self.assertEqual(got["masks"]["late"], s)  # spawned by the moved main thread
        self.assertEqual(got["report"]["expert_threads"], 2)
        self.assertEqual(got["report"]["service_threads"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
