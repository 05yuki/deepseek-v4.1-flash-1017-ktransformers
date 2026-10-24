"""Exercise the actual hot configuration and early-launch validator without CUDA."""
import ast
import os
from pathlib import Path
import sys
import unittest

TREE = Path(sys.argv.pop(1))
SOURCE = TREE / 'python/sglang/srt/layers/moe/kt_kv_hot_lend.py'


def settings(env):
    module = ast.parse(SOURCE.read_text())
    nodes = []
    for n in module.body:
        if isinstance(n, ast.Try):
            break
        if isinstance(n, (ast.Assign, ast.AnnAssign)):
            nodes.append(n)
        elif isinstance(n, ast.FunctionDef) and n.name == '_dyn_flag':
            nodes.append(n)
    validate = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'validate_launch')
    nodes.append(validate)
    ns = {'os': type('OS', (), {'environ': env}), 'kt_kv_vmm': type('VMM', (), {'ENABLED': True})}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<actual hot configuration>', 'exec'), ns)
    return ns


class DynamicHotLaunch(unittest.TestCase):
    def test_legacy_defaults_and_tp2_still_available(self):
        ns = settings({'KT_KV_HOT_LEND': '1', 'KT_KV_HOT_DYNAMIC_SLOTS': '1'})
        self.assertEqual((ns['DYN_SWAPS'], ns['DYN_HYST'], ns['DYN_DWELL'], ns['DYN_BYTES']), (2, 1.0, 0, 0))
        self.assertFalse(ns['DYN_DEFER'])
        ns['validate_launch'](2)

    def test_each_deferred_mode_rejected_at_tp2(self):
        for knob in ('DEFER', 'ASYNC', 'REFILL_FAST'):
            env = {'KT_KV_HOT_LEND': '1', 'KT_KV_HOT_DYNAMIC_SLOTS': '1', 'KT_KV_HOT_DYN_' + knob: '1'}
            ns = settings(env)
            ns['validate_launch'](1)
            with self.assertRaisesRegex(ValueError, 'TP=1'):
                ns['validate_launch'](2)

    def test_existing_dynamic_gate_controls_all_improvements(self):
        env = {'KT_KV_HOT_LEND': '1'}
        for name in ('DEFER', 'ASYNC', 'REFILL_FAST', 'BATCH_LOAD', 'PIN_ARENAS', 'ROUTE_FOLD'):
            env['KT_KV_HOT_DYN_' + name] = '1'
        ns = settings(env)
        self.assertFalse(any(ns[n] for n in ('DYN_DEFER', 'DYN_ASYNC', 'REFILL_FAST', 'BATCH_LOAD', 'PIN_ARENAS', 'ROUTE_FOLD')))
        ns['validate_launch'](2)

    def test_independent_knobs_and_implied_defer(self):
        ns = settings({'KT_KV_HOT_LEND': '1', 'KT_KV_HOT_DYNAMIC_SLOTS': '1', 'KT_KV_HOT_DYN_ASYNC': '1',
                       'KT_KV_HOT_DYN_DWELL': '2', 'KT_KV_HOT_DYN_BYTES_MB': '512'})
        self.assertTrue(ns['DYN_ASYNC'] and ns['DYN_DEFER'])
        self.assertFalse(ns['BATCH_LOAD'])
        self.assertEqual((ns['DYN_DWELL'], ns['DYN_BYTES']), (2, 512 << 20))

    def test_validator_called_before_model_initialization(self):
        module = ast.parse((TREE / 'python/sglang/srt/model_executor/model_runner.py').read_text())
        cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == 'ModelRunner')
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
        calls = [n for n in ast.walk(init) if isinstance(n, ast.Call)]
        validate = next(n for n in calls if isinstance(n.func, ast.Name) and n.func.id == 'validate_launch')
        initializers = [n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr in ('initialize', 'load_model')]
        self.assertTrue(initializers)
        self.assertTrue(all(validate.lineno < n.lineno for n in initializers))


if __name__ == '__main__':
    unittest.main()
