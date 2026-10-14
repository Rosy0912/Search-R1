"""Exercise the real native validator without Torch/CUDA/Ray imports."""
import ast
import copy
import json
import os
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from . import bootstrap


REPO = Path(__file__).resolve().parents[3]
GUARD_PROTOCOL = 'tree_guard_continue_fresh_rollout_v1'
TRAINING_PROTOCOL = 'tree_fresh200_five_leaf_v1'


def native_function(relative_path, name, namespace):
    path = REPO / relative_path
    node = next(value for value in ast.parse(path.read_text()).body
                if isinstance(value, ast.FunctionDef) and value.name == name)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 str(path), 'exec'), namespace)
    return namespace[name]


def worker_module():
    namespace = dict(Path=Path, os=os, copy=copy, json=json,
        enabled=lambda actor: actor.get('guard_continue_protocol') == GUARD_PROTOCOL,
        PROTOCOL='tree_partial_state_v1',
        IDENTITY_KEYS={'job_id', 'run_id', 'code_commit', 'config_sha256', 'data_identity',
                       'model_identity', 'training_protocol', 'attempted_outer', 'completed_outer'},
        ENV_KEYS=dict(job_id='SLURM_JOB_ID', run_id='SEARCH_R1_BASELINE_RUN_ID',
            code_commit='SEARCH_R1_BASELINE_COMMIT', config_sha256='SEARCH_R1_BASELINE_CONFIG_SHA256',
            data_identity='SEARCH_R1_BASELINE_DATA_IDENTITY', model_identity='SEARCH_R1_MODEL_IDENTITY'))
    module = ModuleType('native_partial_worker_under_test')
    module._validated_context = native_function('search_r1/baselines/partial_worker_state.py',
                                                '_validated_context', namespace)
    return module


def changed_native_root(worker, data):
    own_run = Path(os.environ['SEARCH_R1_BASELINE_OUTPUT']).parent
    return own_run


class NamespaceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name).resolve() / 'C1-root-only'
        self.checkpoints = self.base / 'checkpoints'
        self.output = self.base / 'output' / ('attempt-' + 'a' * 32)
        self.output.mkdir(parents=True)
        self.checkpoints.mkdir()
        self.record = dict(base=str(self.base), run_id='run1', source_commit='c' * 40,
            config_sha256='d' * 64, data_identity='e' * 64, actual_update_limit=150)
        self.config = dict(trainer=dict(default_local_dir=str(self.checkpoints), actual_update_limit=150))
        self.env = dict(RECURSIVE_ABLATION_RECEIPT=str(self.base / 'receipts/configuration.json'),
            SEARCH_R1_BASELINE_OUTPUT=str(self.output), SLURM_JOB_ID='3393',
            SEARCH_R1_BASELINE_RUN_ID='run1', SEARCH_R1_BASELINE_COMMIT='c' * 40,
            SEARCH_R1_BASELINE_CONFIG_SHA256='d' * 64, SEARCH_R1_BASELINE_DATA_IDENTITY='e' * 64,
            SEARCH_R1_MODEL_IDENTITY='official-model')
        self.identity = dict(job_id='3393', run_id='run1', code_commit='c' * 40,
            config_sha256='d' * 64, data_identity='e' * 64, model_identity='official-model',
            training_protocol=TRAINING_PROTOCOL, attempted_outer=1, completed_outer=0, attempt_id=1)
        self.name = 'partial_attempt_000001_outer_000001'
        self.worker = SimpleNamespace(config=SimpleNamespace(actor=dict(
            fresh200_protocol=TRAINING_PROTOCOL, guard_continue_protocol=GUARD_PROTOCOL)))
        self.data = SimpleNamespace(meta_info=dict(baseline_outer_step=1, baseline_attempt_id=1,
            tree_partial_context=dict(protocol='tree_partial_state_v1', identity=copy.deepcopy(self.identity),
                directory=str(self.checkpoints / 'training_state' / self.name))))
        self.addCleanup(patch.stopall)
        patch.dict(os.environ, self.env, clear=True).start()
        patch.object(bootstrap, '_frozen_namespace', return_value=(self.record, self.checkpoints)).start()

    def repaired(self):
        module = worker_module()
        bootstrap.install_into(module)
        return module._validated_context

    def test_exact_reported_failure_then_correct_group_namespace(self):
        module = worker_module()
        with self.assertRaisesRegex(ValueError, 'Partial state directory'):
            module._validated_context(self.worker, self.data)
        before = copy.deepcopy(self.data.meta_info)
        bootstrap.install_into(module)
        accepted = module._validated_context(self.worker, self.data)
        self.assertEqual(accepted, self.data.meta_info['tree_partial_context'])
        self.assertIsNot(accepted, self.data.meta_info['tree_partial_context'])
        self.assertEqual(before, self.data.meta_info)

    def test_all_native_source_run_bindings_still_reject_mismatches(self):
        validated = self.repaired()
        for key in ('job_id', 'run_id', 'code_commit', 'config_sha256', 'data_identity', 'model_identity'):
            with self.subTest(key=key):
                data = copy.deepcopy(self.data)
                data.meta_info['tree_partial_context']['identity'][key] = 'wrong'
                with self.assertRaisesRegex(ValueError, 'source/run identity'):
                    validated(self.worker, data)

    def test_native_context_profile_counters_and_keys_remain_strict(self):
        validated = self.repaired()
        modifications = [dict(attempt_id=0), dict(attempted_outer=2),
                         dict(completed_outer=200), dict(extra='unknown')]
        for changes in modifications:
            with self.subTest(changes=changes):
                data = copy.deepcopy(self.data)
                data.meta_info['tree_partial_context']['identity'].update(changes)
                with self.assertRaises(ValueError):
                    validated(self.worker, data)
        worker = copy.deepcopy(self.worker)
        worker.config.actor['fresh200_protocol'] = 'unknown'
        with self.assertRaisesRegex(ValueError, 'explicit fresh200 profile'):
            validated(worker, self.data)

    def test_sibling_nested_wrong_name_and_symlink_escape_are_rejected(self):
        validated = self.repaired()
        for directory in (self.base.parent / 'C2-random-final/checkpoints/training_state' / self.name,
                          self.checkpoints / 'other/training_state' / self.name,
                          self.checkpoints / 'training_state/partial_attempt_000002_outer_000001'):
            data = copy.deepcopy(self.data)
            data.meta_info['tree_partial_context']['directory'] = str(directory)
            with self.assertRaisesRegex(ValueError, 'exact group/checkpoint/attempt'):
                validated(self.worker, data)
        outside = self.base.parent / 'outside'
        outside.mkdir()
        (self.checkpoints / 'training_state').symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'exact group/checkpoint/attempt'):
            validated(self.worker, self.data)

    def test_frozen_namespace_rejects_changed_output_checkpoint_and_receipt_bindings(self):
        self.assertEqual(bootstrap.checkpoint_root(self.record, self.config, self.env), self.checkpoints)
        for key in ('SEARCH_R1_BASELINE_RUN_ID', 'SEARCH_R1_BASELINE_COMMIT',
                    'SEARCH_R1_BASELINE_CONFIG_SHA256', 'SEARCH_R1_BASELINE_DATA_IDENTITY'):
            env = dict(self.env, **{key: 'different'})
            with self.assertRaisesRegex(ValueError, 'source/run binding'):
                bootstrap.checkpoint_root(self.record, self.config, env)
        for output in (self.base.parent / 'other/output' / self.output.name,
                       self.base / 'output/no-attempt', self.base / 'output' / ('attempt-' + 'g' * 32)):
            env = dict(self.env, SEARCH_R1_BASELINE_OUTPUT=str(output))
            with self.assertRaisesRegex(ValueError, 'namespaces'):
                bootstrap.checkpoint_root(self.record, self.config, env)
        config = copy.deepcopy(self.config)
        config['trainer']['default_local_dir'] = str(self.base / 'output')
        with self.assertRaisesRegex(ValueError, 'namespaces'):
            bootstrap.checkpoint_root(self.record, config, self.env)

    def test_inactive_ablation_uses_original_native_check(self):
        validated = self.repaired()
        os.environ.pop('RECURSIVE_ABLATION_RECEIPT')
        with self.assertRaisesRegex(ValueError, 'Partial state directory'):
            validated(self.worker, self.data)

    def test_install_is_idempotent_and_changed_native_root_stops(self):
        module = worker_module()
        bootstrap.install_into(module)
        first = module._validated_context
        bootstrap.install_into(module)
        self.assertIs(first, module._validated_context)
        changed = SimpleNamespace(_validated_context=changed_native_root)
        with self.assertRaisesRegex(ValueError, 'root expression changed'):
            bootstrap.install_into(changed)
        self.assertIs(changed._validated_context, changed_native_root)

    def test_exact_first_replay_namespace_keeps_attempt_and_cannot_overwrite_original(self):
        os.environ['RECURSIVE_ABLATION_ZERO_WRITE_MANIFEST'] = '/exact-original/manifest.json'
        namespace = dict(Path=Path, os=os, enabled=lambda actor: True, PROTOCOL='tree_partial_state_v1')
        module = ModuleType('native_partial_driver_under_test')
        module.partial_context = native_function('search_r1/baselines/tree_partial_state.py',
                                                'partial_context', namespace)
        trainer = SimpleNamespace(global_steps=1, actual_outer_updates=0, training_attempts=1,
            config=SimpleNamespace(actor_rollout_ref=SimpleNamespace(actor={}),
                baselines=SimpleNamespace(training_protocol=TRAINING_PROTOCOL),
                trainer=SimpleNamespace(default_local_dir=str(self.checkpoints))))
        bootstrap.install_driver_into(module)
        context = module.partial_context(trainer)
        expected = self.checkpoints / ('replay-' + self.output.name) / 'training_state' / self.name
        self.assertEqual(context['directory'], str(expected))
        self.assertEqual(context['identity'], self.identity)
        data = copy.deepcopy(self.data)
        data.meta_info['tree_partial_context'] = context
        self.assertEqual(self.repaired()(self.worker, data), context)
        trainer.global_steps, trainer.actual_outer_updates, trainer.training_attempts = 2, 1, 2
        self.assertEqual(module.partial_context(trainer)['directory'], str(self.checkpoints /
            'training_state/partial_attempt_000002_outer_000002'))
        os.environ.pop('RECURSIVE_ABLATION_ZERO_WRITE_MANIFEST')
        with self.assertRaisesRegex(ValueError, 'exact group/checkpoint/attempt'):
            self.repaired()(self.worker, data)


if __name__ == '__main__':
    unittest.main()
