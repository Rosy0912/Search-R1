"""Tiny CPU reproduction of PEFT named-storage rounding; no real assets/GPU."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import recovery_common as common
import trainer_launcher as recovery

ORIGINAL = Path(os.environ.get('OPSD_SIGNAL_RECOVERY_TEST_ORIGINAL',
                HERE.parent / 'opsd_signal_weight_v1'))
spec = importlib.util.spec_from_file_location('_actual_original_signal_targets', ORIGINAL / 'targets.py')
native_targets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native_targets)


class NamedAdapters(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList()
        for _ in range(252):
            module = torch.nn.Module()
            module.lora_A = torch.nn.ModuleDict({
                'default': torch.nn.Linear(1, 1, bias=False, dtype=torch.float32),
                'outer_old': torch.nn.Linear(1, 1, bias=False, dtype=torch.bfloat16)})
            module.lora_B = torch.nn.ModuleDict({
                'default': torch.nn.Linear(1, 1, bias=False, dtype=torch.float32),
                'outer_old': torch.nn.Linear(1, 1, bias=False, dtype=torch.bfloat16)})
            self.blocks.append(module)
        self.active_adapter = 'default'

    def set_adapter(self, name):
        self.active_adapter = name


def original_setter(model, weights, *args, **kwargs):
    adapter = kwargs.get('adapter_name', args[0] if args else 'default')
    actual = recovery.adapter_values(model, adapter)
    with torch.no_grad():
        for name, parameter in actual.items():
            parameter.copy_(weights[name])
    return ORIGINAL_RETURN


ORIGINAL_RETURN = object()


def source_weights(model):
    return {name: torch.tensor([[0.12345679 + index / 100000]], dtype=torch.float32)
            for index, name in enumerate(recovery.adapter_values(model, 'outer_old'))}


class StorageRecoveryTests(unittest.TestCase):
    def test_native_bf16_setter_fails_actual_frozen_guard_and_postcast_cannot_recover(self):
        model = NamedAdapters(); weights = source_weights(model)
        identity = {'adapter_tensor_sha256': native_targets.tensor_digest(weights)}
        original_setter(model, weights, adapter_name='outer_old')
        with self.assertRaisesRegex(ValueError, 'actual tensors differ'):
            native_targets.SharedTargets(model, object(), old_adapter='outer_old', old_identity=identity)
        for parameter in recovery.adapter_values(model, 'outer_old').values():
            parameter.data = parameter.data.float()
        self.assertNotEqual(native_targets.tensor_digest(recovery.adapter_values(model, 'outer_old')),
                            identity['adapter_tensor_sha256'])

    def test_pre_setter_recovery_passes_unchanged_native_guard_and_preserves_current_optimizer(self):
        model = NamedAdapters(); weights = source_weights(model); before = native_targets.tensor_digest(
            recovery.adapter_values(model, 'default'))
        optimizer = torch.optim.AdamW(recovery.adapter_values(model, 'default').values(), lr=1e-6)
        parameter_ids = [id(parameter) for group in optimizer.param_groups for parameter in group['params']]
        old_ids = {id(parameter) for parameter in recovery.adapter_values(model, 'outer_old').values()}
        input_digest = native_targets.tensor_digest(weights); records = []
        torch_rng = torch.get_rng_state().clone(); python_rng = random.getstate()
        setter = recovery.repaired_setter(original_setter, native_targets.tensor_digest, input_digest, records.append)
        self.assertIs(setter(model, weights, adapter_name='outer_old'), ORIGINAL_RETURN)
        self.assertTrue(torch.equal(torch_rng, torch.get_rng_state())); self.assertEqual(python_rng, random.getstate())
        targets = native_targets.SharedTargets(model, object(), old_adapter='outer_old',
                    old_identity={'adapter_tensor_sha256': input_digest})
        targets.check_old()
        self.assertEqual(native_targets.tensor_digest(recovery.adapter_values(model, 'default')), before)
        self.assertEqual(parameter_ids, [id(parameter) for group in optimizer.param_groups for parameter in group['params']])
        self.assertEqual(old_ids, {id(parameter) for parameter in recovery.adapter_values(model, 'outer_old').values()})
        self.assertTrue(all(parameter.dtype == torch.float32 and parameter.requires_grad
                            for parameter in recovery.adapter_values(model, 'default').values()))
        self.assertTrue(all(parameter.dtype == torch.float32 and not parameter.requires_grad
                            for parameter in recovery.adapter_values(model, 'outer_old').values()))
        self.assertEqual(len(records), 1); self.assertEqual(records[0]['old_storage_dtypes_before'], ['torch.bfloat16'])
        self.assertEqual(records[0]['original_input_tensor_sha256'], records[0]['actual_old_tensor_sha256'])
        self.assertTrue(records[0]['nonold_parameter_objects_versions_dtypes_flags_unchanged'])

    def test_non_old_setter_is_exact_passthrough(self):
        calls = []; returned = object()
        def original(*args, **kwargs):
            calls.append((args, kwargs)); return returned
        model = object(); weights = object()
        setter = recovery.repaired_setter(original, None, None, lambda value: self.fail('Not an old load'))
        self.assertIs(setter(model, weights, 'default', fixture=42), returned)
        self.assertEqual(calls, [((model, weights, 'default'), {'fixture': 42})])

    def test_wrong_digest_nonfinite_missing_inventory_and_shape_fail_before_mutation(self):
        for mutation in ('wrong_digest', 'nan', 'missing', 'shape'):
            with self.subTest(mutation=mutation):
                model = NamedAdapters(); weights = source_weights(model)
                expected = native_targets.tensor_digest(weights)
                if mutation == 'wrong_digest': expected = 'f' * 64
                if mutation == 'nan': weights[next(iter(weights))].fill_(float('nan'))
                if mutation == 'missing': weights.pop(next(iter(weights)))
                if mutation == 'shape': weights[next(iter(weights))] = torch.ones(2, dtype=torch.float32)
                before = native_targets.tensor_digest(recovery.adapter_values(model, 'outer_old'))
                setter = recovery.repaired_setter(original_setter, native_targets.tensor_digest, expected,
                    lambda value: self.fail('Failed repair must not be accepted'))
                with self.assertRaises(ValueError): setter(model, weights, adapter_name='outer_old')
                self.assertEqual(native_targets.tensor_digest(recovery.adapter_values(model, 'outer_old')), before)
                self.assertTrue(all(parameter.dtype == torch.bfloat16
                                    for parameter in recovery.adapter_values(model, 'outer_old').values()))

    def test_unexpected_current_mutation_is_rejected(self):
        model = NamedAdapters(); weights = source_weights(model)
        def bad(model, weights, **kwargs):
            returned = original_setter(model, weights, **kwargs)
            with torch.no_grad(): next(iter(recovery.adapter_values(model, 'default').values())).add_(1)
            return returned
        setter = recovery.repaired_setter(bad, native_targets.tensor_digest, native_targets.tensor_digest(weights),
                    lambda value: self.fail('Unexpected current mutation accepted'))
        with self.assertRaisesRegex(ValueError, 'current adapter'): setter(model, weights, adapter_name='outer_old')

    def test_native_old_mutation_guard_remains_enforced_after_repair(self):
        model = NamedAdapters(); weights = source_weights(model); digest = native_targets.tensor_digest(weights)
        recovery.repaired_setter(original_setter, native_targets.tensor_digest, digest, lambda value: None)(
                model, weights, adapter_name='outer_old')
        target = native_targets.SharedTargets(model, object(), old_adapter='outer_old',
                    old_identity={'adapter_tensor_sha256': digest})
        with torch.no_grad(): next(iter(recovery.adapter_values(model, 'outer_old').values())).add_(1)
        with self.assertRaisesRegex(ValueError, 'changed'): target.check_old()

    def test_source_loader_refuses_preimport_shadowing(self):
        manifest = {'files': {'occupied_fixture.py': {'sha256': 'a' * 64}}}
        with patch.dict(sys.modules, {'occupied_fixture': object()}):
            with self.assertRaisesRegex(ValueError, 'already imported'):
                common.install_original(Path('/unstarted-fixture'), manifest)

    def test_original_file_sha_checks_refuse_changed_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.py'; path.write_text('immutable = 1\n')
            ref = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            self.assertEqual(common.checked(ref), path.read_bytes())
            path.write_text('immutable = 2\n')
            with self.assertRaisesRegex(ValueError, 'SHA differs'): common.checked(ref)


if __name__ == '__main__':
    unittest.main()
