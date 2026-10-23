"""CPU-only checks of the actual opt-in setup and serve shell blocks."""

import os
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]
BASH = os.environ.get("GLUE_BASH", "bash")
KNOBS = ("KT_IDX_LIVE_TOPK", "KT_SM89_FUSED_MERGE", "KT_SHARED_AFTER_SUBMIT",
         "KT_HANDOFF_PACK", "KT_MEMOP_POLL_HINT", "KT_TASKQUEUE_SPIN_US")


def shell(script, env=None):
    return subprocess.run([BASH, "-c", script], cwd=ROOT,
                          env={**os.environ, **(env or {})},
                          text=True, capture_output=True, timeout=10)


class BundleConfiguration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        serve = (ROOT / "serve.sh").read_text(encoding="utf-8")
        cls.block = serve[serve.index('case "${KT_GPU_GLUE:-0}" in'):
                          serve.index('export KT_MXFP4_BACKEND=')]

    def run_serve(self, value, installed=True, service=False, overrides=None):
        env = {k: "" for k in KNOBS}
        # Unset knobs in the subshell so inherited developer settings cannot
        # accidentally make an off-default test pass.
        body = "\n".join(f"unset {k}" for k in KNOBS)
        body += "\nunset KT_SERVICE_CPUS\n"
        if service:
            body += "export KT_SERVICE_CPUS=7,15,23,31,39,47,55,63\n"
        if overrides:
            body += "\n".join(f"export {k}={v}" for k, v in overrides.items()) + "\n"
        body += f"export KT_GPU_GLUE={value}\nhas() {{ return {0 if installed else 1}; }}\n"
        body += self.block
        body += "\nprintf '%s\\n' " + " ".join(f'"${{{k}-unset}}"' for k in KNOBS)
        return shell("set -e\n" + body, env)

    def test_default_off(self):
        result = self.run_serve("0", installed=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["unset"] * 5 + ["300"])

    def test_enabled_and_overrides(self):
        result = self.run_serve("1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["1"] * 5 + ["2000"])
        result = self.run_serve("1", overrides={k: "0" for k in KNOBS})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["0"] * 6)

    def test_feature_required_for_glue_and_service_affinity(self):
        self.assertEqual(self.run_serve("1", installed=False).returncode, 3)
        self.assertEqual(self.run_serve("0", installed=False, service=True).returncode, 3)
        self.assertEqual(self.run_serve("2").returncode, 2)

    def test_optional_native_patch_changes_build_marker(self):
        source = (ROOT / "setup.sh").read_text(encoding="utf-8")
        block = source[source.index("kt_variant=pinned"):
                       source.index("# a patch is checked")]
        variants = []
        for enabled in ("0", "1"):
            script = ("set -e\nTIER=2\nHOT_GLUE_PATCH=" + enabled +
                      "\npatch_hash() { printf '%s' \"$*\"; }\n" + block +
                      "\nprintf '%s' \"$kt_variant\"")
            result = shell(script)
            self.assertEqual(result.returncode, 0, result.stderr)
            variants.append(result.stdout)
        self.assertNotEqual(*variants)
        self.assertNotIn("kt-kernel-gpu-glue-only.patch", variants[0])
        self.assertIn("kt-kernel-gpu-glue-only.patch", variants[1])

    def test_source_selection_requires_full_tier_and_orders_optional_patches(self):
        source = (ROOT / "setup.sh").read_text(encoding="utf-8")
        block = source[source.index('TIER="${TIER:-2}"'):
                       source.index('export CUDA_HOME=')]
        # Git's Windows checkout makes requires.txt CRLF; the shipped Git blob
        # and Linux setup input are LF. Normalize only this test's read stream.
        block = block.replace(
            'done < "$here/patches/sglang-tier1/requires.txt"',
            'done < <(tr -d "\\r" < "$here/patches/sglang-tier1/requires.txt")')
        preamble = "set -e\nhere=$PWD\n"
        for tier, features in (("0", "all"), ("1", "fp8-decode"),
                               ("2", "fp8-decode")):
            result = shell(preamble + block,
                           {"TIER": tier, "SGLANG_FEATURES": features,
                            "HOT_GLUE_PATCH": "1"})
            self.assertEqual(result.returncode, 2, result.stderr)
        script = (preamble + block +
                  "printf '%s\\n' \"${sglang_patches[@]}\"\n"
                  "printf 'feature:%s\\n' \"${features[@]}\"\n")
        for tier in ("1", "2"):
            off = shell(script, {"TIER": tier, "SGLANG_FEATURES": "all",
                                 "HOT_GLUE_PATCH": "0"})
            on = shell(script, {"TIER": tier, "SGLANG_FEATURES": "all",
                                "HOT_GLUE_PATCH": "1"})
            self.assertEqual(off.returncode, 0, off.stderr)
            self.assertEqual(on.returncode, 0, on.stderr)
            self.assertNotIn("sglang-optional/", off.stdout)
            self.assertEqual(on.stdout.splitlines()[-1], "feature:gpu-glue")
            self.assertLess(on.stdout.index("sglang-optional/01-gpu-glue.patch"),
                            on.stdout.index("sglang-optional/02-hot-slots.patch"))

    def test_eight_node_profile_is_explicit(self):
        result = shell("source profiles/eight-node-hot-glue.sh\n"
                       "printf '%s\\n' \"$KT_GPU_GLUE\" \"$KT_KV_HOT_PORT\" "
                       "\"$KT_CPU_THREADS\" \"$KT_THREADPOOL_COUNT\" "
                       "\"$KT_NUMA_NODES\" \"$KT_SERVICE_CPUS\"")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(),
                         ["1", "1", "56", "8", "0,1,2,3,4,5,6,7",
                          "7,15,23,31,39,47,55,63"])


if __name__ == "__main__":
    unittest.main()
