"""CPU regressions for the actual KV lending module in the tier-1 patches.

Run: python tools/hot-expert-spare-lending-test.py
Needs git; no SGLang installation, torch, weights or GPU are required.
"""

import ast
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


class Pages(list):
    device = "cuda:0"
    dtype = "int64"

    def __getitem__(self, key):
        value = super().__getitem__(key)
        return Pages(value) if isinstance(key, slice) else value

    def clone(self):
        return Pages(self)

    def numel(self):
        return len(self)

    def to(self, dtype):
        return self


class HotExpertSpareLendingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        recipe = Path(__file__).resolve().parent.parent
        module = "python/sglang/srt/layers/moe/kt_kv_prefill_lend.py"
        with tempfile.TemporaryDirectory() as directory:
            for name in ("15-kv-lend-plan", "16-hot-lend", "17-host-kv", "21-hicache"):
                subprocess.run(
                    [
                        "git",
                        "apply",
                        "--whitespace=nowarn",
                        "--include=" + module,
                        str(recipe / "patches/sglang-tier1" / (name + ".patch")),
                    ],
                    cwd=directory,
                    check=True,
                    stdout=subprocess.DEVNULL,
                )
            tree = ast.parse((Path(directory) / module).read_text())
        names = {
            "_lend_spare",
            "decode_lend",
            "restore",
            "_wrap_allocator",
            "_extend_pages",
        }
        cls.code = compile(
            ast.Module(
                body=[
                    node
                    for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name in names
                ],
                type_ignores=[],
            ),
            "kt_kv_prefill_lend.py",
            "exec",
        )

    def setUp(self):
        self.allocator = SimpleNamespace(free_pages=Pages([7, 11]), page_size=256)
        self.state = {
            "alloc": self.allocator,
            "ps": 256,
            "lent": None,
            "ranges": [],
            "bytes": 0,
            "owner": None,
            "hot_lends": 0,
            "tensors": [(None, 1024 * 1024, 2 * 1024 * 1024, 0)],
            "windows": 0,
        }
        self.free_count = Mock(return_value=2)
        self.release_free = Mock(return_value=None)
        self.setup = Mock(return_value=True)
        self.events = []
        self.scope = {
            "HOT": True,
            "STATE": self.state,
            "_setup": self.setup,
            "_free_count": self.free_count,
            "_release_free": self.release_free,
            "time": time,
            "VERIFY": False,
            "CHECK": False,
            "_log": lambda _: None,
            "PRE_RESTORE": [lambda why: self.events.append("release-experts")],
            "torch": SimpleNamespace(
                cat=lambda values: Pages(sum(values, Pages())),
                cuda=SimpleNamespace(
                    synchronize=lambda dev: self.events.append("sync"),
                    empty_cache=lambda: self.events.append("empty-cache"),
                ),
            ),
            "kt_kv_vmm": SimpleNamespace(
                restore=Mock(
                    side_effect=AssertionError(
                        "A spare-only loan must not remap KV storage"
                    )
                )
            ),
        }
        exec(self.code, self.scope)

    def lend(self, minimum=1):
        return self.scope["decode_lend"](object(), 65536, minimum)

    def assert_spare_only(self):
        self.assertEqual(self.state["owner"], "hot")
        self.assertEqual(self.state["lent"].numel(), 0)
        self.assertEqual(self.state["ranges"], [])
        self.assertEqual(self.state["bytes"], 0)
        self.assertEqual(self.state["hot_lends"], 1)
        self.assertEqual(self.allocator.free_pages, [7, 11])
        self.release_free.assert_not_called()

    def test_cached_prefixes_do_not_block_spare_driver_memory(self):
        self.assertEqual(self.lend(), 1)
        self.assert_spare_only()

    def test_reserve_boundary_keeps_every_kv_page(self):
        self.free_count.return_value = 257
        self.assertEqual(self.lend(), 1)
        self.assert_spare_only()

    def test_empty_free_list_can_still_borrow_driver_memory(self):
        self.allocator.free_pages = Pages()
        self.free_count.return_value = 0
        self.assertEqual(self.lend(), 1)
        self.assertEqual(self.state["lent"].numel(), 0)
        self.assertEqual(self.allocator.free_pages, [])

    def test_insufficient_spare_memory_preserves_reserve(self):
        self.assertEqual(self.lend(2), 0)
        self.assertIsNone(self.state["lent"])
        self.assertEqual(self.state["hot_lends"], 0)
        self.release_free.assert_not_called()

    def test_no_whole_chunk_does_not_block_spare_memory(self):
        self.free_count.return_value = 258
        self.assertEqual(self.lend(), 1)
        self.assert_spare_only()

    def test_fragmented_pool_without_candidates_can_use_spare_memory(self):
        self.free_count.return_value = 1024
        self.assertEqual(self.lend(), 1)
        self.assertEqual(self.state["bytes"], 0)
        self.release_free.assert_called_once_with(257)

    def test_existing_device_page_loan_is_preserved(self):
        self.free_count.return_value = 1024
        pages = Pages([19, 23])
        self.release_free.return_value = (pages, None, [(0, 0, 8192)], 8192)
        self.assertEqual(self.lend(4096), 8192)
        self.assertIs(self.state["lent"], pages)
        self.assertEqual(self.state["bytes"], 8192)
        self.release_free.assert_called_once_with(257)

    def test_existing_host_pool_spare_path_is_preserved(self):
        self.setup.return_value = False
        self.state["spare_only"] = True
        self.assertEqual(self.lend(), 1)
        self.assert_spare_only()

    def test_unsupported_pool_is_not_bypassed(self):
        self.setup.return_value = False
        self.assertEqual(self.lend(), 0)
        self.assertIsNone(self.state["lent"])

    def test_disabled_lending_and_missing_runner_do_nothing(self):
        self.scope["HOT"] = False
        self.assertEqual(self.lend(), 0)
        self.scope["HOT"] = True
        self.assertEqual(self.scope["decode_lend"](None, 65536, 1), 0)
        self.setup.assert_not_called()

    def test_second_loan_cannot_overwrite_active_loan(self):
        self.assertEqual(self.lend(), 1)
        self.assertEqual(self.lend(), 0)
        self.assertEqual(self.state["hot_lends"], 1)

    def wrapped_allocator(self):
        self.state["lent"] = Pages()
        self.state["take_backs"] = 0
        self.free_count.return_value = 0
        self.allocator.alloc = Mock(return_value="allocated")
        self.allocator.alloc_extend = Mock(return_value="extended")
        self.allocator.alloc_decode = Mock(return_value="decoded")
        self.allocator.available_size = Mock(return_value=256)
        self.allocator.clear = Mock(return_value="cleared")
        self.original = SimpleNamespace(**vars(self.allocator))
        self.scope["restore"] = Mock()
        self.scope["_wrap_allocator"](self.allocator)
        return self.scope["restore"]

    def test_spare_only_alloc_does_not_release_experts(self):
        restore = self.wrapped_allocator()
        self.assertEqual(self.allocator.alloc(512), "allocated")
        self.original.alloc.assert_called_once_with(512)
        restore.assert_not_called()
        self.assertEqual(self.state["take_backs"], 0)

    def test_spare_only_extend_does_not_release_experts(self):
        restore = self.wrapped_allocator()
        self.assertEqual(
            self.allocator.alloc_extend(prefix_lens=[1], extend_num_tokens=512),
            "extended",
        )
        self.original.alloc_extend.assert_called_once_with(
            prefix_lens=[1], extend_num_tokens=512
        )
        restore.assert_not_called()

    def test_spare_only_decode_does_not_release_experts(self):
        restore = self.wrapped_allocator()
        self.assertEqual(self.allocator.alloc_decode(seq_lens=[1024000]), "decoded")
        self.original.alloc_decode.assert_called_once_with(seq_lens=[1024000])
        restore.assert_not_called()

    def test_real_page_loan_is_restored_before_allocation(self):
        restore = self.wrapped_allocator()
        self.state["lent"] = Pages([19, 23])
        events = []
        restore.side_effect = lambda why: events.append("restore")
        self.original.alloc.side_effect = lambda size: events.append("allocate")
        self.allocator.alloc(512)
        self.assertEqual(events, ["restore", "allocate"])
        self.assertEqual(self.state["take_backs"], 1)

    def test_available_size_counts_only_real_borrowed_pages(self):
        self.wrapped_allocator()
        self.assertEqual(self.allocator.available_size(), 256)
        self.state["lent"] = Pages([19, 23])
        self.assertEqual(self.allocator.available_size(), 768)

    def test_clear_still_releases_spare_only_experts(self):
        restore = self.wrapped_allocator()
        events = []
        restore.side_effect = lambda why: events.append(why)
        self.original.clear.side_effect = lambda: events.append("clear-pool")
        self.allocator.clear()
        self.assertEqual(events, ["clear", "clear-pool"])

    def test_restore_releases_experts_without_changing_kv_pages(self):
        self.assertEqual(self.lend(), 1)
        self.scope["restore"]("window start")
        self.assertEqual(self.events[0], "release-experts")
        self.assertEqual(self.allocator.free_pages, [7, 11])
        self.assertIsNone(self.state["lent"])
        self.assertIsNone(self.state["owner"])
        self.assertEqual(self.state["bytes"], 0)
        self.scope["kt_kv_vmm"].restore.assert_not_called()


if __name__ == "__main__":
    unittest.main()
