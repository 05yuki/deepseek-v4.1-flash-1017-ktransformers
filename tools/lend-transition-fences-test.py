"""CPU-only regression for the actual kt_lend.py embedded in tier 1.

Run: python tools/lend-transition-fences-test.py
No SGLang installation, model weights or CUDA device are needed.
"""

import ast
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class Paused(Exception):
    pass


class LendingTransitionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        patch = (
            Path(__file__).resolve().parent.parent
            / "patches/sglang-tier1/13-prefill-lend.patch"
        ).read_text()
        section = patch.split(
            "diff --git a/python/sglang/srt/layers/moe/kt_lend.py ", 1
        )[1]
        section = section.split("\ndiff --git ", 1)[0]
        source = (
            "\n".join(
                line[1:]
                for line in section.splitlines()
                if line.startswith("+") and not line.startswith("+++")
            )
            + "\n"
        )
        tree = ast.parse(source)
        cls.code = compile(
            ast.Module(
                body=[
                    node
                    for node in tree.body
                    if isinstance(node, ast.FunctionDef)
                    and node.name in ("before_forward", "end_window")
                ],
                type_ignores=[],
            ),
            "lend-transitions",
            "exec",
        )

    def setUp(self):
        self.events = []
        self.device = SimpleNamespace(index=0)
        self.state = {"phase": "decode", "device": self.device}
        cuda = SimpleNamespace(
            mem_get_info=lambda _: (1 << 30, 2 << 30),
            synchronize=lambda device: self.events.append(("sync", device)),
            max_memory_reserved=lambda _: 0,
            empty_cache=lambda: None,
        )
        self.namespace = {
            "torch": SimpleNamespace(cuda=cuda),
            "STATE": self.state,
            "ENABLED": True,
            "DEBUG": False,
            "LONG_SEQ": 0,
            "TAG_WEIGHTS": "weights",
            "TAG_SCRATCH": "scratch",
            "time": time,
            "_streamed": lambda _: True,
            "_kv_mod": lambda: None,
            "_tms": lambda: SimpleNamespace(pause=self.pause),
            "_fw_close": lambda: None,
            "_leave": lambda: None,
        }
        exec(self.code, self.namespace)
        self.batch = SimpleNamespace(input_ids=SimpleNamespace(shape=(135,)))

    def pause(self, tag):
        self.events.append(("pause", tag))
        raise Paused

    def assert_fenced_pause(self, tag):
        self.assertEqual(self.events, [("sync", self.device), ("pause", tag)])

    def test_decode_to_prefill_waits_before_unmapping_weights(self):
        with self.assertRaises(Paused):
            self.namespace["before_forward"](self.batch)
        self.assert_fenced_pause("weights")

    def test_prefill_to_decode_waits_before_unmapping_scratch(self):
        self.state["phase"] = "prefill"
        runtime = SimpleNamespace(get_parallel=lambda: SimpleNamespace(tp_rank=0))
        with patch.dict(sys.modules, {"sglang.srt.runtime_context": runtime}):
            with self.assertRaises(Paused):
                self.namespace["end_window"]()
        self.assert_fenced_pause("scratch")

    def test_disabled_lending_has_no_synchronization_cost(self):
        self.namespace["ENABLED"] = False
        self.namespace["before_forward"](self.batch)
        self.state["phase"] = "prefill"
        self.namespace["end_window"]()
        self.assertEqual(self.events, [])

    def test_ordinary_decode_does_not_synchronize(self):
        self.namespace["_streamed"] = lambda _: False
        self.namespace["before_forward"](self.batch)
        self.namespace["end_window"]()
        self.assertEqual(self.events, [])


if __name__ == "__main__":
    unittest.main()
