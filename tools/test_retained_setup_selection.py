"""CPU-only check of the optional retained-session source selection."""

import os
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
BASH = os.environ.get("GLUE_BASH", "bash")


class Selection(unittest.TestCase):
    def test_full_tier_required_and_optional_patch_last(self):
        source = (ROOT / "setup.sh").read_text(encoding="utf-8")
        block = source[source.index('TIER="${TIER:-2}"'):
                       source.index('export CUDA_HOME=')]
        # This checkout's Windows requires.txt is CRLF; Git blobs and Linux
        # runtime input are LF, so normalize only the test's read stream.
        block = block.replace(
            'done < "$here/patches/sglang-tier1/requires.txt"',
            'done < <(tr -d "\\r" < "$here/patches/sglang-tier1/requires.txt")')
        script = ("set -e\nhere=$PWD\n" + block +
                  "printf '%s\\n' \"${sglang_patches[@]}\"\n")

        def run(tier, features, selected):
            env = dict(os.environ, TIER=tier, SGLANG_FEATURES=features,
                       SESSION_CONCURRENCY_PATCH=selected)
            return subprocess.run([BASH, "-c", script], cwd=ROOT, env=env,
                                  capture_output=True, text=True, timeout=10)

        for tier, features in (("0", "all"), ("1", "none"),
                               ("2", "fp8-decode")):
            self.assertEqual(run(tier, features, "1").returncode, 2)
        for tier in ("1", "2"):
            off = run(tier, "all", "0")
            on = run(tier, "all", "1")
            self.assertEqual(off.returncode, 0, off.stderr)
            self.assertEqual(on.returncode, 0, on.stderr)
            name = "sglang-optional/01-retained-session-admission.patch"
            self.assertNotIn(name, off.stdout)
            self.assertEqual(on.stdout.splitlines()[-1], name)


if __name__ == "__main__":
    unittest.main()
