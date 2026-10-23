"""CPU-only selection proof for the optional hot glue source and native patch."""

import os
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
BASH = os.environ.get("GLUE_BASH", "bash")


def selector_script():
    source = (ROOT / "setup.sh").read_text(encoding="utf-8")
    selection = source[source.index('TIER="${TIER:-2}"'):
                       source.index('export CUDA_HOME=')]
    # Git checkouts on Windows may carry CRLF in requires.txt; CI/Linux uses LF.
    selection = selection.replace(
        'done < "$here/patches/sglang-tier1/requires.txt"',
        'done < <(tr -d "\\r" < "$here/patches/sglang-tier1/requires.txt")')
    native = source[source.index('patch_hash()'):
                    source.index('# a patch is checked')]
    return ("set -Eeuo pipefail\nhere=$PWD\n" + selection + native +
            "printf 'SG=%s\\n' \"${sglang_patches[@]}\"\n"
            "printf 'KT=%s\\n' \"${kt_patches[@]}\"\n"
            "printf 'SG_STATE=%s\\n' \"$sglang_state\"\n"
            "printf 'KT_VARIANT=%s\\n' \"$kt_variant\"\n")


class HotGlueSelection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = selector_script()

    def run_selector(self, *, tier="1", features="all", selected=None):
        env = dict(os.environ, TIER=tier, SGLANG_FEATURES=features)
        env.pop("HOT_GLUE_PATCH", None)
        if selected is not None:
            env["HOT_GLUE_PATCH"] = selected
        return subprocess.run([BASH, "-c", self.script], cwd=ROOT,
                              env=env, text=True, capture_output=True, timeout=15)

    def test_default_off_and_tier1_tier2_on(self):
        optional = ("SG=sglang-optional/01-gpu-glue.patch",
                    "SG=sglang-optional/02-hot-slots.patch")
        for tier in ("1", "2"):
            off = self.run_selector(tier=tier)
            on = self.run_selector(tier=tier, selected="1")
            self.assertEqual(off.returncode, 0, off.stderr)
            self.assertEqual(on.returncode, 0, on.stderr)
            self.assertFalse(any(line in off.stdout for line in optional))
            self.assertNotIn("KT=kt-kernel-gpu-glue-only.patch", off.stdout)
            self.assertIn("\n".join(optional), on.stdout)
            self.assertIn("KT=kt-kernel-gpu-glue-only.patch", on.stdout)
            self.assertNotEqual([s for s in off.stdout.splitlines() if s.startswith("SG_STATE=")],
                                [s for s in on.stdout.splitlines() if s.startswith("SG_STATE=")])
            self.assertNotEqual([s for s in off.stdout.splitlines() if s.startswith("KT_VARIANT=")],
                                [s for s in on.stdout.splitlines() if s.startswith("KT_VARIANT=")])

    def test_hot_glue_requires_full_tier1_and_valid_flag(self):
        for tier, features in (("0", "all"), ("1", "none"),
                               ("2", "fp8-decode")):
            result = self.run_selector(tier=tier, features=features, selected="1")
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("requires TIER=1 or 2 and SGLANG_FEATURES=all", result.stderr)
        invalid = self.run_selector(selected="yes")
        self.assertEqual(invalid.returncode, 2)

    def test_off_selection_does_not_add_optional_native_patch(self):
        result = self.run_selector(tier="1", features="fp8-decode", selected="0")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("SG=sglang-optional/", result.stdout)
        self.assertNotIn("KT=kt-kernel-gpu-glue-only.patch", result.stdout)

    def test_runtime_gates_remain_independent_and_default_off(self):
        launcher = (ROOT / "serve.sh").read_text(encoding="utf-8")
        hot_patch = (ROOT / "patches/sglang-optional/02-hot-slots.patch").read_text(
            encoding="utf-8")
        self.assertIn('case "${KT_GPU_GLUE:-0}" in', launcher)
        self.assertIn('KT_GPU_GLUE=1 requires HOT_GLUE_PATCH=1 at setup', launcher)
        self.assertIn('os.environ.get("KT_KV_HOT_PORT", "0") == "1"', hot_patch)
        self.assertIn('Independent of KT_GPU_GLUE',
                      (ROOT / "patches/sglang-optional/01-gpu-glue.patch").read_text(
                          encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
