"""Small CPU tensors and real immutable-checkpoint interface guards; no model."""
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

try:
    from . import movement as m
except ImportError:
    import movement as m


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path); result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result); return result


TRAINING = Path(os.environ.get('OPSD_MOVEMENT_TEST_TRAINING_ROOT',
                              str(Path(__file__).resolve().parents[1] / 'opsd_signal_weight_v1'))).resolve()
NATIVE_CP = module(TRAINING / 'checkpoint.py', '_movement_test_native_checkpoint')
TARGETS = module(TRAINING / 'targets.py', '_movement_test_native_targets')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, sort_keys=True, allow_nan=False).encode(); path.write_bytes(raw)
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


class Metadata(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup); self.root = Path(self.temp.name)
        self.cfg_ref = write(self.root / 'config.json', {})
        self.config = dict(run_id='a' * 32, variant='G_POS', precision=dict(mode='bf16'),
            _config_sha256=self.cfg_ref['sha256'], runtime_manifest=dict(sha256='b' * 64),
            source_model_manifest=dict(sha256='c' * 64), common_initial_adapter_manifest=dict(path='initial', sha256='d' * 64))
        cp_root = self.root / 'checkpoint'; cp_root.mkdir()
        adapter_files = [write(cp_root / 'adapter' / 'adapter_config.json', {}),
                         write(cp_root / 'adapter' / 'adapter_model.safetensors', {})]
        outer_old = write(self.root / 'outer-old.json', {})
        training = {key: write(self.root / (key + '.json'), {}) for key in ('outer_manifest', 'pairs', 'order', 'encoding')}
        self.cp = dict(schema='opsd-online-checkpoint-v1', accepted=True, run_id=self.config['run_id'],
            config_sha256=self.cfg_ref['sha256'], runtime_identity='b' * 64,
            contract=NATIVE_CP.science_contract(self.config), common_initial_adapter=self.config['common_initial_adapter_manifest'],
            outer_old_adapter=outer_old, all_rank_scaler_state=[{}] * 4, source_model_identity='c' * 64,
            optimizer_updates=20, cumulative_pair_exposure=610, complete_rank_count=4, equal_optimizer_steps=True,
            ranks=[dict(rank=i, file=write(cp_root / ('rank-%d.pt' % i), dict(rank=i))) for i in range(4)],
            optimizer_storage='shared_exact_four_rank_optimizer_v1', shared_optimizer=write(cp_root / 'optimizer-shared.pt', {}),
            all_rank_optimizer_tensor_sha256=['e' * 64] * 4,
            adapter=dict(path=str(cp_root / 'adapter'), files=adapter_files), all_rank_adapter_sha256=['f' * 64] * 4,
            all_rng=['python', 'numpy', 'torch_cpu', 'cuda_all'], scheduler=dict(kind='none', warmup_updates=0),
            cursor=dict(outer_index=1, next_pair_cursor=300, usable_pair_count=300), training_input=training,
            sampling_model_identity=dict(adapter_tensor_sha256='f' * 64))
        self.cp_ref = write(cp_root / 'manifest.json', self.cp)
        self.endpoint = dict(checkpoint=self.cp_ref, optimizer_updates=20, outer_index=1, cumulative_pair_exposure=610)

    def audit(self, cp=None, endpoint=None):
        if cp is not None:
            self.endpoint['checkpoint'] = write(Path(self.cp_ref['path']), cp)
        return m.checkpoint_metadata(endpoint or self.endpoint, self.config, self.cfg_ref, dict(checkpoint=NATIVE_CP))

    def test_real_native_complete_checkpoint_interface(self):
        value, evidence = self.audit()
        self.assertEqual(value['optimizer_updates'], 20)
        self.assertTrue(evidence['optimizer_payloads_hashed_not_deserialized'])
        self.assertEqual(set(evidence['adapter_model']), {'path', 'sha256', 'bytes'})

    def test_wrong_outer_rejected(self):
        endpoint = dict(self.endpoint, outer_index=14)
        with self.assertRaisesRegex(ValueError, 'Wrong outer'): self.audit(endpoint=endpoint)

    def test_uncommitted_cursor_rejected(self):
        cp = copy.deepcopy(self.cp); cp['cursor']['next_pair_cursor'] = 299
        with self.assertRaisesRegex(ValueError, 'K not fully committed'): self.audit(cp)

    def test_false_native_accepted_rejected(self):
        cp = dict(self.cp, accepted=False)
        with self.assertRaisesRegex(ValueError, 'Incomplete checkpoint'): self.audit(cp)

    def test_missing_rank_file_rejected(self):
        Path(self.cp['ranks'][0]['file']['path']).unlink()
        with self.assertRaises(FileNotFoundError): self.audit()

    def test_wrong_actual_pair_exposure_rejected(self):
        endpoint = dict(self.endpoint, cumulative_pair_exposure=611)
        with self.assertRaisesRegex(ValueError, 'Pair exposure'): self.audit(endpoint=endpoint)

    def test_best_adapter_outside_raw_checkpoint_rejected(self):
        cp = copy.deepcopy(self.cp); cp['adapter'] = dict(path=str(self.root / 'best'), files=[
            write(self.root / 'best' / 'adapter_config.json', {}), write(self.root / 'best' / 'adapter_model.safetensors', {})])
        with self.assertRaisesRegex(ValueError, 'cache/best refused'): self.audit(cp)

    def test_sha_mutation_rejected(self):
        Path(self.cp['adapter']['files'][1]['path']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'SHA mismatch'): self.audit()

    def test_false_complete_coverage_rejected(self):
        cli = dict(accepted=True, final_accepted=False, output=dict(path='coverage', sha256='0' * 64))
        report = dict(schema='opsd-signal-final-coverage-v1', accepted=True, coverage_accepted=True,
            final_accepted=False, complete_groups=4, complete_outers=99, train_trajectory_slots=253440,
            formal_native512_slots=4608, torch_imported=False, study_manifest=dict(path='study', sha256='1' * 64))
        packet = dict(actual_stdout=dict(path='stdout', sha256='2' * 64), output=cli['output'])
        with mock.patch.object(m, 'execution_packet', return_value=(cli, {})), mock.patch.object(m, 'read', return_value=(report, cli['output'])):
            with self.assertRaisesRegex(ValueError, 'Complete100/4608 coverage'):
                m.validate_coverage(packet, report['study_manifest'], {}, {})

    def test_coverage_actual_stdout_artifact_ref_rejected(self):
        cli = dict(accepted=True, final_accepted=False, output=dict(path='wrong', sha256='0' * 64))
        packet = dict(actual_stdout=dict(path='stdout', sha256='2' * 64), output=dict(path='coverage', sha256='0' * 64))
        with mock.patch.object(m, 'execution_packet', return_value=(cli, {})):
            with self.assertRaisesRegex(ValueError, 'actual CLI output differs'):
                m.validate_coverage(packet, {}, {}, {})


class Numerics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Missing Torch is a hard dependency failure, never an advertised SKIP/PASS.
        import torch
        cls.torch = torch

    def values(self, a=1., b=0.):
        return {('base_model.model.layers.%d.module%d.lora_%s.weight' % (i // 7, i % 7, factor)):
                self.torch.full((16, 2) if factor == 'A' else (2, 16), a if factor == 'A' else b,
                                dtype=self.torch.float32) for i in range(252) for factor in ('A', 'B')}

    def test_known_delta_and_factor_norms(self):
        left, right = self.values(), self.values(a=4., b=4.)
        result = m.compare(left, right, self.torch)
        count = 252 * 32
        self.assertAlmostEqual(result['aggregate']['delta_l2'], 5 * math.sqrt(count), places=9)
        self.assertAlmostEqual(result['aggregate']['relative_l2'], 5., places=12)
        self.assertAlmostEqual(result['by_factor']['A']['relative_l2'], 3., places=12)
        self.assertEqual(result['aggregate']['changed_tensors'], 504)
        self.assertFalse(result['effective_model_weight_delta_computed'])

    def test_zero_B_relative_is_na_even_when_B_changes(self):
        result = m.compare(self.values(), self.values(b=1.), self.torch)
        self.assertIsNone(result['by_factor']['B']['relative_l2'])
        self.assertEqual(result['by_factor']['B']['relative_l2_na_reason'], 'ZERO_SOURCE_PARAMETER_NORM')
        self.assertGreater(result['by_factor']['B']['delta_l2'], 0)

    def test_equal_endpoints_zero_displacement(self):
        values = self.values(); result = m.compare(values, values, self.torch)
        self.assertEqual(result['aggregate']['delta_l2'], 0)
        self.assertEqual(result['aggregate']['relative_l2'], 0)
        self.assertEqual(result['aggregate']['changed_tensors'], 0)

    def test_native_digest_and_zero_initial_inventory(self):
        values = self.values(); actual = TARGETS.tensor_digest(values)
        self.assertEqual(m.validate_tensors(values, actual, TARGETS.tensor_digest, self.torch, zero=True), dict(A=252, B=252))
        with self.assertRaisesRegex(ValueError, 'digest differs'):
            m.validate_tensors(values, '0' * 64, TARGETS.tensor_digest, self.torch)

    def test_missing_504_tensor_rejected(self):
        values = self.values(); values.pop(next(iter(values)))
        with self.assertRaisesRegex(ValueError, 'Exactly504'):
            m.validate_tensors(values, '', TARGETS.tensor_digest, self.torch)

    def test_wrong_dtype_rejected(self):
        values = self.values(); key = next(iter(values)); values[key] = values[key].half()
        with self.assertRaisesRegex(ValueError, 'finite FP32'):
            m.validate_tensors(values, '', TARGETS.tensor_digest, self.torch)

    def test_nonzero_initial_B_rejected(self):
        values = self.values(b=1.)
        with self.assertRaisesRegex(ValueError, 'initial B tensor is nonzero'):
            m.validate_tensors(values, '', TARGETS.tensor_digest, self.torch, zero=True)

    def test_nonfinite_input_rejected(self):
        values = self.values(); values[next(iter(values))][0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'finite FP32'):
            m.validate_tensors(values, '', TARGETS.tensor_digest, self.torch)

    def test_wrong_rank_dimension_rejected(self):
        values = self.values(); values[next(iter(values))] = self.torch.ones(2, 2)
        with self.assertRaisesRegex(ValueError, 'rank16'):
            m.validate_tensors(values, '', TARGETS.tensor_digest, self.torch)

    def test_shape_mismatch_rejected(self):
        left, right = self.values(), self.values(); right[next(iter(right))] = self.torch.ones(16, 3)
        with self.assertRaisesRegex(ValueError, 'shape/dtype'):
            m.compare(left, right, self.torch)

    def test_fp64_subtraction_not_fp32_norm_overflow(self):
        left, right = self.values(a=1e30, b=1e30), self.values(a=2e30, b=2e30)
        result = m.compare(left, right, self.torch)
        self.assertTrue(math.isfinite(result['aggregate']['delta_l2']))
        self.assertAlmostEqual(result['aggregate']['relative_l2'], 1., places=10)

    def test_parameter_factorization_is_not_effective_weight_delta(self):
        left, right = self.values(a=1., b=2.), self.values(a=2., b=1.)
        a = next(v for k, v in left.items() if '.lora_A.' in k); b = next(v for k, v in left.items() if '.lora_B.' in k)
        aa = next(v for k, v in right.items() if '.lora_A.' in k); bb = next(v for k, v in right.items() if '.lora_B.' in k)
        self.assertTrue(self.torch.equal(b @ a, bb @ aa))
        self.assertGreater(m.compare(left, right, self.torch)['aggregate']['delta_l2'], 0)
        self.assertFalse(self.torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
