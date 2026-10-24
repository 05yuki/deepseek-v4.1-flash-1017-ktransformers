"""CPU-only contracts for the tier-1 overlap feature and native selection."""
import ast
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
TREE = Path(sys.argv.pop(1))
MOE = TREE / 'python/sglang/srt/layers/moe'
BASH = os.environ.get('GLUE_BASH', 'bash')


class OverlapContracts(unittest.TestCase):
    def test_switches_independent_default_off(self):
        tree = ast.parse((MOE / 'kt_handoff_pack.py').read_text())
        nodes = [n for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id in ('PACK', 'SHARED_AFTER_SUBMIT') for t in n.targets)]
        for pack, shared in [('0', '0'), ('1', '0'), ('0', '1'), ('1', '1')]:
            ns = {'os': type('OS', (), {'environ': {'KT_HANDOFF_PACK': pack, 'KT_SHARED_AFTER_SUBMIT': shared}})}
            exec(compile(ast.Module(body=nodes, type_ignores=[]), '<switches>', 'exec'), ns)
            self.assertEqual((ns['PACK'], ns['SHARED_AFTER_SUBMIT']), (pack == '1', shared == '1'))

    def selector(self, features, tier='1'):
        source = (ROOT / 'setup.sh').read_text()
        text = source[source.index('TIER="${TIER:-2}"'):source.index('export CUDA_HOME=')]
        text = text.replace('done < "$here/patches/sglang-tier1/requires.txt"',
                            'done < <(tr -d "\\r" < "$here/patches/sglang-tier1/requires.txt")')
        text += source[source.index('patch_hash()'):source.index('# a patch is checked')]
        script = 'set -Eeuo pipefail\nhere=$PWD\n' + text
        script += '\nprintf "%s\\n" "${kt_patches[@]}" "$kt_variant"\n'
        return subprocess.run([BASH, '-c', script], cwd=ROOT,
                              env=dict(os.environ, TIER=tier, SGLANG_FEATURES=features),
                              capture_output=True, text=True, timeout=15)

    def test_native_hook_selected_and_hashed_with_feature(self):
        off, on = self.selector('none'), self.selector('all')
        self.assertEqual(off.returncode, 0, off.stderr)
        self.assertEqual(on.returncode, 0, on.stderr)
        self.assertNotIn('kt-kernel-cpu-handoff.patch', off.stdout)
        self.assertIn('kt-kernel-cpu-handoff.patch', on.stdout)
        self.assertNotEqual(off.stdout.splitlines()[-1], on.stdout.splitlines()[-1])
        self.assertEqual(self.selector('none', tier='0').stdout.splitlines()[-1], 'pinned')

    def test_missing_prerequisites_refused(self):
        result = self.selector('cpu-gpu-overlap')
        self.assertEqual(result.returncode, 2)
        self.assertIn('needs', result.stderr)

    def test_shared_stream_waits_on_submit_or_fallback_event(self):
        text = (TREE / 'python/sglang/srt/models/deepseek_v2.py').read_text()
        module = ast.parse(text)
        fn = next(n for n in ast.walk(module) if isinstance(n, ast.FunctionDef)
                  and n.name == 'forward_normal_dual_stream')
        fork = next(n for n in fn.body if isinstance(n, ast.If) and isinstance(n.test, ast.Name)
                    and n.test.id == 'kt_late_fork' and any(isinstance(v, ast.Call)
                    and isinstance(v.func, ast.Attribute) and v.func.attr == 'wait_event' for v in ast.walk(n)))
        code = compile(ast.Module(body=[fork], type_ignores=[]), '<shared fork>', 'exec')
        for submitted in (None, object()):
            waits = []
            method = type('Method', (), {'_kt_post_submit_event': submitted})()
            obj = type('MoE', (), {'experts': type('Experts', (), {'quant_method': method})(),
                                  'alt_stream': type('Stream', (), {'wait_event': lambda s, e: waits.append(e)})()})()
            fallback = object()
            exec(code, {'self': obj, 'kt_late_fork': True, 'kt_fork_event': fallback})
            self.assertEqual(waits, [submitted if submitted is not None else fallback])
            self.assertIsNone(method._kt_post_submit_event)


if __name__ == '__main__':
    unittest.main()
