"""CPU-only checks of the opt-in retained-session admission methods.

Run after setup, or set SGLANG_SOURCE_ROOT to a patched SGLang source tree.
The isolated methods come from the actual source AST; this does not import the
model, allocate KV memory, or start a server.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(os.environ.get("SGLANG_SOURCE_ROOT", "third_party/sglang"))
SRT = ROOT / "python/sglang/srt"


def method(path: Path, class_name: str, method_name: str) -> ast.FunctionDef:
    module = ast.parse(path.read_text(encoding="utf-8"))
    owner = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    return next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == method_name)


def isolated_scheduler():
    path = SRT / "managers/scheduler.py"
    methods = [method(path, "Scheduler", name) for name in (
        "_streaming_session_admission", "_prefill_admission_budget", "_streaming_row_available")]
    cls = ast.ClassDef(name="Port", bases=[], keywords=[], body=methods, decorator_list=[])
    module = ast.Module(body=[ast.ImportFrom(module="__future__",
                                             names=[ast.alias(name="annotations")], level=0), cls],
                        type_ignores=[])
    namespace = {"os": os, "get_parallel": lambda: SimpleNamespace(pp_max_micro_batch_size=4)}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["Port"]


class Pool:
    def __init__(self, free: int):
        self.free = free

    def available_size(self):
        return self.free


class Cache:
    def __init__(self, pool, records):
        self.pool, self.records = pool, records
        self.released = []
        self.demand = None

    def session_records(self):
        return self.records

    def set_session_row_reclaim_demand(self, count):
        self.demand = count

    def release_session(self, sid):
        self.released.append(sid)
        del self.records[sid]
        self.pool.free += 1


def request(*, session=None, held=False, to_finish=None):
    return SimpleNamespace(session=session, kv=SimpleNamespace(holds_kv=held),
                           to_finish=to_finish, beam_group=None)


class RetainedAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.port = isolated_scheduler()()
        self.pool = Pool(0)
        sessions = {str(n): SimpleNamespace(session_id=str(n), streaming=True,
                    _inflight=False, last_active_time=float(n)) for n in range(4)}
        records = {sid: SimpleNamespace(holds_kv=True) for sid in sessions}
        self.port.session_controller = SimpleNamespace(sessions=sessions)
        self.port.req_to_token_pool = self.pool
        self.port.tree_cache = Cache(self.pool, records)
        self.port.beam_coordinator = SimpleNamespace(pending_member_rows=lambda _: 0)
        self.port.running_batch = SimpleNamespace(reqs=[])
        self.port.max_running_requests = 4
        self.port.get_num_allocatable_reqs = lambda *a, **k: 0

    def test_default_off_delegates_original_budget(self):
        with patch.dict(os.environ, {"KT_STREAMING_SESSION_ADMISSION": "0"}):
            self.assertFalse(self.port._streaming_session_admission())
            self.assertEqual(self.port._prefill_admission_budget(self.port.running_batch), 0)

    def test_four_held_rows_do_not_close_active_request_budget(self):
        with patch.dict(os.environ, {"KT_STREAMING_SESSION_ADMISSION": "1"}):
            self.assertEqual(self.port._prefill_admission_budget(self.port.running_batch), 4)

    def test_fifth_request_evicts_one_oldest_idle_row(self):
        with patch.dict(os.environ, {"KT_STREAMING_SESSION_ADMISSION": "1"}):
            self.assertTrue(self.port._streaming_row_available(request(), []))
        self.assertEqual(self.port.tree_cache.released, ["0"])
        self.assertEqual(self.pool.free, 1)
        self.assertEqual(len(self.port.tree_cache.records), 3)

    def test_continuation_reuses_held_row(self):
        continuation = request(session=self.port.session_controller.sessions["3"])
        self.assertTrue(self.port._streaming_row_available(continuation, []))
        self.assertEqual(self.port.tree_cache.released, [])

    def test_no_active_row_is_evicted(self):
        for session in self.port.session_controller.sessions.values():
            session._inflight = True
        self.assertFalse(self.port._streaming_row_available(request(), []))
        self.assertEqual(self.port.tree_cache.released, [])
        self.assertEqual(self.port.tree_cache.demand, 1)

    def test_rejected_overlap_detaches_before_abort_cleanup(self):
        path = SRT / "session/session_controller.py"
        create = method(path, "Session", "create_req")
        branch = next(n for n in create.body if isinstance(n, ast.If)
                      and isinstance(n.test, ast.Name) and n.test.id == "abort")
        fn = ast.FunctionDef(name="abort_branch",
                             args=ast.arguments(posonlyargs=[], args=[ast.arg(arg=n) for n in
                                 ("self", "new_req", "abort", "abort_message")],
                                 vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
                             body=[branch], decorator_list=[])
        module = ast.Module(body=[fn], type_ignores=[])
        namespace = {"os": os}
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
        owner = SimpleNamespace(streaming=True, _inflight=True)
        rejected = SimpleNamespace(session=owner, set_finish_with_abort=lambda _: None)
        with patch.dict(os.environ, {"KT_STREAMING_SESSION_ADMISSION": "1"}):
            namespace["abort_branch"](owner, rejected, True, "overlap")
        self.assertIsNone(rejected.session)
        self.assertTrue(owner._inflight)
        original_path = SimpleNamespace(session=owner, set_finish_with_abort=lambda _: None)
        with patch.dict(os.environ, {"KT_STREAMING_SESSION_ADMISSION": "0"}):
            namespace["abort_branch"](owner, original_path, True, "overlap")
        self.assertIs(original_path.session, owner)


if __name__ == "__main__":
    unittest.main()
