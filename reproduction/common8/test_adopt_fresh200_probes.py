"""CPU byte/schema fixtures, not native execution or diagnostic-health proof.

The original frozen subprocess authentication and native target allocation are
explicit mocked boundaries. Real file identities, calibration NPZ checks,
scientific compatibility, publication and fail-closed partial claims execute.
"""
import copy
import hashlib
import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from reproduction.common8 import adopt_fresh200_probes as adoption
from reproduction.common8 import adopt_initial_probes as original
from reproduction.common8.formal_contract import FORMAL_MATHEMATICS
from reproduction.common8.policy_calibration_plan import (
    COMMON_MANIFEST_SHA256, DISCOVERY_ROLE, MATHEMATICS, MODEL_ID,
    MODEL_REVISION, PlanError, canonical_bytes, document_sha256,
)
from reproduction.common8.verify_data import digest


def write(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(document))
    return digest(path)


class FreshProbeAdoptionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name).resolve()
        self.source, self.base, self.model = root / 'diagnostic', root / 'formal', root / 'official'
        for base in (self.source, self.base):
            for folder in ('logs', 'preparation', 'artifacts', 'repo'):
                (base / folder).mkdir(parents=True)
        self.model.mkdir()
        model_names = ('config.json', 'model.safetensors.index.json',
                       'model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors',
                       'tokenizer.json', 'tokenizer_config.json', 'vocab.json',
                       'merges.txt', 'generation_config.json')
        model_hashes = {}
        for name in model_names:
            path = self.model / name
            path.write_bytes(('CPU official-byte fixture ' + name).encode())
            model_hashes[name] = digest(path)
        common = dict(profile='common8', model_id=MODEL_ID, model_revision=MODEL_REVISION,
                      common_manifest_sha256=COMMON_MANIFEST_SHA256)
        # Keep the original manifest's sparse budget: its additional scientific
        # settings are bound by the authenticated original resolved recipe.
        old_budget = dict(common, **{name: MATHEMATICS[name] for name in (
            'fresh_actor', 'fixed_original_reference', 'world_size', 'train_batch_size',
            'n_agent', 'rollout_n', 'sequence_parallel_size', 'trajectory_count',
            'ppo_mini_batch_size', 'ppo_epochs', 'optimizer_steps_per_outer',
            'ppo_micro_batch_size', 'per_rank_actor_micro_batch', 'log_prob_micro_batch_size',
            'per_rank_log_prob_micro_batch', 'learning_rate', 'gradient_clip_global_l2',
            'target_actual_outer_updates', 'split_counts', 'training_start_checkpoint',
            'lr_warmup_steps_ratio', 'learning_rate_schedule')})
        source_hashes = {}
        for name in adoption.PROBE_SOURCE_FILES:
            raw = ('CPU stable scientific implementation ' + name).encode()
            for base in (self.source, self.base):
                path = base / 'repo' / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
            source_hashes[name] = hashlib.sha256(raw).hexdigest()
        inputs = {'manifest.json': COMMON_MANIFEST_SHA256}
        inputs.update({name: hashlib.sha256(name.encode()).hexdigest() for name in (
            'train.parquet', 'validation.parquet', 'stage.parquet', 'final.parquet',
            'calibration-300.parquet', 'exclusions-audit.json', 'audit.json')})
        self.old = dict(profile='common8', run_id='old-diagnostic', completed_updates=0,
                        model_revision=MODEL_REVISION, initial_model=str(self.model),
                        reference_model=str(self.model), initial_model_sha256=model_hashes,
                        source_commit='f' * 40, source_sha256=source_hashes, input_sha256=inputs,
                        calibration_question_ids=[f'q{i}' for i in range(300)], budget=old_budget,
                        policy_calibration_role=DISCOVERY_ROLE, parent_production_run_id='old-parent',
                        policy_calibration_slurm_job_id='100',
                        initial_probe_version='old-diagnostic:0:fixture')
        new_budget = dict(common, **FORMAL_MATHEMATICS, contract_kind='fresh200_warmup57')
        self.proof = self.base / 'preparation/original-diagnostic-proof.json'
        self.terminal = self.base / 'preparation/original-native-terminal.json'
        self.run = dict(self.old, run_id='fresh200', source_commit='a' * 40,
                        source_sha256=dict(source_hashes, **{'reproduction/common8/new-bootstrap.py': 'b' * 64}),
                        budget=new_budget, policy_calibration_role='production',
                        contract_kind='fresh200_warmup57', target_updates=200, acceptance_update=5,
                        policy_safety_mode=FORMAL_MATHEMATICS['policy_safety_mode'],
                        initial_probe_version=None,
                        checkpoint_dir=str(self.base / 'checkpoints/search-g1-common8'),
                        train_log=str(self.base / 'logs/train.log'),
                        policy_calibration_discovery_base=str(self.source),
                        diagnostic_proof=str(self.proof), diagnostic_terminal_proof=str(self.terminal))
        for key in ('parent_production_run_id', 'policy_calibration_slurm_job_id'):
            self.run.pop(key)
        self.round = self.source / 'artifacts/round_0-attempt-fixture'
        heads, dump = self.round / 'heads', self.round / 'dumps'
        heads.mkdir(parents=True)
        dump.mkdir()
        self.heads, self.dump = heads, dump
        for name in ('commit_probe.pkl', 'boundary_probe.pkl'):
            (heads / name).write_bytes(('CPU unpickled boundary fixture ' + name).encode())
        report = dict(commit_cv_auc=.7, boundary_cv_auc=.8, commit_position='Xa', commit_label='flip',
                      unique_question_count=300, commit_trajectory_count=6, question_grouped_cv=True,
                      boundary_unique_question_weighting=True, require_all_cv_folds=True,
                      configured_cv_folds=2, boundary_class_counts=[150, 150], commit_class_counts=[3, 3])
        label = dict(question_count=300, group_size=5, question_ids=self.old['calibration_question_ids'],
                     searched_valid_trajectory_count=6, boundary_class_counts=[150, 150],
                     reliance_class_counts=[3, 3], closed_book_label='exact_match_no_retrieval',
                     reliance_label='normalized_answer_change_after_evidence_deletion',
                     model_authored_answer_spans=True, boundary_all_reserved_questions=True,
                     calibration_protocol=dict(rollout_max_new=500, correction_feedback=True),
                     evidence_deletion='typed_retrieval_ranges_only_preserve_model_and_correction')
        write(heads / 'calibration-report.json', report)
        write(dump / 'calibration-label-proof.json', label)
        records = [dict(qi=i, em=i % 2, flip=i % 2, cb=i % 2, answer='CPU raw answer',
                        flip_answer='CPU intervention', counterfactual_valid=True,
                        counterfactual_strict_valid=True) for i in range(6)]
        (dump / 'nq-hotpot-common_0.jsonl').write_bytes(
            b'\n'.join(canonical_bytes(row) for row in records) + b'\n')
        self.arrays = dict(boundary_question_id=np.array(label['question_ids']),
                           boundary_q_cb=np.array([0, 1] * 150), y_flip=np.array([0, 1] * 3),
                           y_em=np.array([0, 1] * 3), q_cb=np.array([0, 1] * 3),
                           gid=np.arange(6, dtype=np.int32), question_id=np.array(label['question_ids'][:6]))
        for layer in (21, 24, 27):
            self.arrays['Xa_' + str(layer)] = np.zeros((6, 2), dtype=np.float16)
            self.arrays['Xp_question_' + str(layer)] = np.zeros((300, 2), dtype=np.float16)
        np.savez(dump / 'nq-hotpot-common_0.npz', **self.arrays)
        self.live = dict(run_id=self.old['run_id'], round=0, version=self.old['initial_probe_version'],
                         encoder=str(self.model), commit=str(heads / 'commit_probe.pkl'),
                         boundary=str(heads / 'boundary_probe.pkl'),
                         encoder_sha256={name: sha for name, sha in model_hashes.items()
                                         if name not in ('vocab.json', 'merges.txt', 'generation_config.json')},
                         head_sha256={path.name: digest(path) for path in heads.glob('*.pkl')},
                         calibration_report=report, label_proof=label,
                         calibration_report_sha256=digest(heads / 'calibration-report.json'),
                         dump_sha256={path.name: digest(path) for path in dump.iterdir()}, published_unix=1.)
        self.plan = dict(role=DISCOVERY_ROLE, run_id=self.old['run_id'], parent_production_run_id='old-parent',
                         slurm_job_id='100', mathematics=copy.deepcopy(MATHEMATICS),
                         identity=dict(original_source='f' * 40, original_contract='constant300'),
                         initial_reward_fingerprint=dict(probe_round=0,
                             encoder_sha256=self.live['encoder_sha256'], head_sha256=self.live['head_sha256'],
                             reward_math_sha256='c' * 64))
        self.segment = dict(run_id=self.old['run_id'], role=DISCOVERY_ROLE + '_evidence',
                            mathematics=copy.deepcopy(MATHEMATICS),
                            scope='CPU authentication boundary fixture, never actual native execution')
        write(self.source / 'logs/run-manifest.json', self.old)
        write(self.source / 'logs/policy-safety-calibration-plan.json', self.plan)
        write(self.source / 'logs/policy-calibration-segment.json', self.segment)
        write(self.source / 'preparation/parent-run-manifest.json', dict(run_id='old-parent', budget=old_budget))
        write(self.round / 'version-manifest.json', self.live)
        write(self.source / 'artifacts/live/manifest.json', self.live)
        write(self.source / 'logs/initial-probes-accepted.json', dict(run_id=self.old['run_id'], accepted=True,
              version=self.live['version'], live_manifest_sha256=digest(self.source / 'artifacts/live/manifest.json'),
              grouped_calibration_cv=report, label_proof=label))
        write(self.proof, dict(role='CPU original subprocess-authentication fixture', original_contract='constant300'))
        write(self.terminal, dict(role='CPU native terminal fixture', State='COMPLETED', ExitCode='0:0'))
        self.sync_run()
        self.refresh_authentication()

    def sync_run(self):
        write(self.base / 'logs/run-manifest.json', self.run)

    def refresh_authentication(self):
        self.authenticated = dict(run=copy.deepcopy(self.old), plan=copy.deepcopy(self.plan),
                                 segment=copy.deepcopy(self.segment),
                                 source_file_sha256={str(path.relative_to(self.source)): digest(path)
                                                     for path in self.source.rglob('*') if path.is_file()},
                                 diagnostic_proof_sha256=digest(self.proof), terminal_proof_sha256=digest(self.terminal))

    def adopt(self, **kwargs):
        with mock.patch.object(adoption, 'authenticate_original_diagnostic',
                               return_value=self.authenticated) as authenticate, \
             mock.patch.object(original, '_actual_target', return_value='101'), \
             mock.patch.object(original, '_fitted_heads', **kwargs):
            result = adoption.adopt(self.base, self.source, self.proof, self.terminal)
        authenticate.assert_called_once_with(self.base, self.source, self.proof, self.terminal)
        return result

    def assert_unclaimed(self):
        self.assertFalse((self.base / 'logs/initial-probe-adoption-claim.json').exists())
        self.assertFalse((self.base / 'artifacts/live/manifest.json').exists())

    def test_new_version_original_contract_preserved_and_no_health_or_state_inherited(self):
        old_bytes = {path: path.read_bytes() for path in self.source.rglob('*') if path.is_file()}
        receipt = self.adopt()
        live = json.loads((self.base / 'artifacts/live/manifest.json').read_text())
        self.assertTrue(receipt['adopted'])
        self.assertEqual(receipt['schema_version'], 3)
        self.assertFalse(receipt['production_admission'])
        self.assertFalse(receipt['healthy_checkpoint_accepted'])
        self.assertTrue(receipt['child_http_admission_required'])
        self.assertTrue(receipt['new_native_model_admission_required'])
        self.assertTrue(receipt['new_all2560_behavior_admission_required'])
        self.assertTrue(live['version'].startswith('fresh200:0:'))
        self.assertNotEqual(live['version'], self.live['version'])
        self.assertEqual(live['head_sha256'], self.live['head_sha256'])
        self.assertEqual(live['dump_sha256'], self.live['dump_sha256'])
        self.assertEqual(live['adoption_origin']['original_mathematics'], MATHEMATICS)
        self.assertEqual(live['adoption_origin']['original_identity'], self.plan['identity'])
        self.assertEqual(live['adoption_origin']['run_id'], self.old['run_id'])
        self.assertEqual(live['adoption_origin']['parent_production_run_id'], 'old-parent')
        archive = self.base / receipt['initial_archive']
        self.assertEqual(json.loads(archive.read_text()), live)
        self.assertEqual(receipt['initial_archive_sha256'], digest(archive))
        self.assertEqual(receipt['initial_archive_sha256'], receipt['live_manifest_sha256'])
        for key in ('commit', 'boundary'):
            self.assertEqual(Path(live[key]).read_bytes(), Path(self.live[key]).read_bytes())
        self.assertEqual(old_bytes, {path: path.read_bytes() for path in self.source.rglob('*') if path.is_file()})
        self.assertFalse((self.base / 'logs/initial-probes-accepted.json').exists())
        self.assertFalse(Path(self.run['checkpoint_dir']).exists())
        self.assertIsNone(json.loads((self.base / 'logs/run-manifest.json').read_text())['initial_probe_version'])
        self.assertNotIn('healthy_checkpoint_accepted', live)

    def test_original_authentication_failure_never_claims_or_publishes(self):
        with mock.patch.object(adoption, 'authenticate_original_diagnostic',
                               side_effect=PlanError('original subprocess rejected its own actual recipe')):
            with self.assertRaises(PlanError):
                adoption.adopt(self.base, self.source, self.proof, self.terminal)
        self.assert_unclaimed()

    def test_stale_original_proof_terminal_or_run_rejected_before_copy(self):
        for field in ('diagnostic_proof_sha256', 'terminal_proof_sha256'):
            with self.subTest(field=field):
                saved = self.authenticated[field]
                self.authenticated[field] = '0' * 64
                with self.assertRaises(PlanError):
                    self.adopt()
                self.authenticated[field] = saved
                self.assert_unclaimed()
        self.authenticated['run']['run_id'] = 'relabeled-fresh200'
        with self.assertRaises(PlanError):
            self.adopt()
        self.assert_unclaimed()

    def test_original_object_or_graph_omissions_rejected(self):
        for name in ('plan', 'source_file_sha256'):
            with self.subTest(name=name):
                saved = self.authenticated.pop(name)
                with self.assertRaises(PlanError):
                    self.adopt()
                self.authenticated[name] = saved
                self.assert_unclaimed()
        name = 'logs/policy-calibration-segment.json'
        self.authenticated['source_file_sha256'].pop(name)
        with self.assertRaises(PlanError):
            self.adopt()
        self.assert_unclaimed()

    def test_exact_model_data_calibration_ids_required(self):
        for name in ('initial_model_sha256', 'input_sha256', 'calibration_question_ids'):
            with self.subTest(name=name):
                saved = copy.deepcopy(self.run[name])
                if isinstance(saved, dict):
                    self.run[name] = dict(saved)
                    self.run[name][next(iter(saved))] = '0' * 64
                else:
                    self.run[name] = saved[::-1]
                self.sync_run()
                with self.assertRaises((PlanError, AssertionError)):
                    self.adopt()
                self.run[name] = saved
                self.sync_run()
                self.assert_unclaimed()

    def test_probe_implementation_change_is_not_admission_source_change(self):
        name = 'ts_probe_train.py'
        changed = self.base / 'repo' / name
        changed.write_bytes(b'changed scientific labels or fit protocol')
        self.run['source_sha256'][name] = digest(changed)
        self.sync_run()
        with self.assertRaises(PlanError):
            self.adopt()
        self.assert_unclaimed()

    def test_actual_stable_source_mutation_rejected_even_with_equal_manifest_maps(self):
        (self.base / 'repo/search_g1/signals.py').write_bytes(b'changed after prepare')
        with self.assertRaises(PlanError):
            self.adopt()
        self.assert_unclaimed()

    def test_unapproved_budget_field_or_training_inheritance_rejected(self):
        self.run['budget']['unexpected_reward_temperature'] = .9
        self.sync_run()
        with self.assertRaises(PlanError):
            self.adopt()
        self.run['budget'].pop('unexpected_reward_temperature')
        self.run['actor_checkpoint'] = '/historical/global_step_1'
        self.sync_run()
        with self.assertRaises(PlanError):
            self.adopt()
        self.assert_unclaimed()

    def test_wrong_goal_and_same_native_job_rejected(self):
        self.run['target_updates'] = 300
        self.sync_run()
        with self.assertRaises(AssertionError):
            self.adopt()
        self.run['target_updates'] = 200
        self.sync_run()
        with mock.patch.object(adoption, 'authenticate_original_diagnostic', return_value=self.authenticated), \
             mock.patch.object(original, '_actual_target', return_value='100'):
            with self.assertRaises(PlanError):
                adoption.adopt(self.base, self.source, self.proof, self.terminal)
        self.assert_unclaimed()

    def test_trainer_claim_blocks_before_original_authentication(self):
        write(self.base / 'logs/training-launch-claim.json', {'partial': True})
        with mock.patch.object(adoption, 'authenticate_original_diagnostic') as authenticate:
            with self.assertRaises(PlanError):
                adoption.adopt(self.base, self.source, self.proof, self.terminal)
        authenticate.assert_not_called()
        self.assert_unclaimed()

    def test_fitted_arrays_and_heads_rechecked_after_authentication(self):
        self.arrays['Xa_21'][0, 0] = np.nan
        np.savez(self.dump / 'nq-hotpot-common_0.npz', **self.arrays)
        self.live['dump_sha256']['nq-hotpot-common_0.npz'] = digest(self.dump / 'nq-hotpot-common_0.npz')
        write(self.round / 'version-manifest.json', self.live)
        write(self.source / 'artifacts/live/manifest.json', self.live)
        accepted = original._read(self.source / 'logs/initial-probes-accepted.json')
        accepted['live_manifest_sha256'] = digest(self.source / 'artifacts/live/manifest.json')
        write(self.source / 'logs/initial-probes-accepted.json', accepted)
        self.refresh_authentication()
        with self.assertRaises(PlanError):
            self.adopt()
        self.assert_unclaimed()

    def test_head_loader_rejection_is_not_bypassed(self):
        with self.assertRaises(PlanError):
            self.adopt(side_effect=PlanError('nonfinite actual fitted arrays'))
        self.assert_unclaimed()

    def test_actual_initial_artifacts_must_belong_to_old_authentication_graph(self):
        self.authenticated['source_file_sha256'].pop(str((self.heads / 'commit_probe.pkl').relative_to(self.source)))
        with self.assertRaises(PlanError):
            self.adopt()
        self.assert_unclaimed()

    def test_symlink_or_escaping_authentication_files_rejected(self):
        self.authenticated['source_file_sha256']['../outside'] = '0' * 64
        with self.assertRaises(PlanError):
            self.adopt()
        self.assert_unclaimed()
        self.authenticated['source_file_sha256'].pop('../outside')
        path = self.heads / 'boundary_probe.pkl'
        raw = path.read_bytes()
        path.unlink()
        other = self.base.parent / 'outside-head.pkl'
        other.write_bytes(raw)
        path.symlink_to(other)
        with self.assertRaises(PlanError):
            self.adopt()
        self.assert_unclaimed()

    def test_partial_copy_preserves_exclusive_claim_and_cannot_reenter(self):
        durable_write = original._write
        def fail_copy(path, raw):
            if Path(path).parent.name == 'heads':
                raise OSError('injected write failure after durable claim')
            return durable_write(path, raw)
        with mock.patch.object(original, '_write', side_effect=fail_copy):
            with self.assertRaises(OSError):
                self.adopt()
        claim = self.base / 'logs/initial-probe-adoption-claim.json'
        original_claim = claim.read_bytes()
        self.assertEqual(len(list((self.base / 'artifacts').glob('round_0-adopted-*'))), 1)
        with self.assertRaises(PlanError):
            self.adopt()
        self.assertEqual(claim.read_bytes(), original_claim)
        self.assertFalse((self.base / 'artifacts/live/manifest.json').exists())

    def test_atomic_live_publish_does_not_overwrite_competing_version(self):
        hard_link = os.link
        def race(source, destination):
            Path(destination).write_bytes(b'owned competing publication')
            return hard_link(source, destination)
        with mock.patch.object(adoption.os, 'link', side_effect=race):
            with self.assertRaises(FileExistsError):
                self.adopt()
        self.assertEqual((self.base / 'artifacts/live/manifest.json').read_bytes(), b'owned competing publication')
        self.assertTrue((self.base / 'logs/initial-probe-adoption-claim.json').exists())
        self.assertFalse((self.base / 'logs/initial-probe-adoption.json').exists())

    def test_target_change_during_authenticated_copy_preserves_claim(self):
        durable_write = original._write
        def change_after_copy(path, raw):
            result = durable_write(path, raw)
            if Path(path).parent.name == 'heads':
                self.run['run_id'] = 'mutated-target'
                self.sync_run()
            return result
        with mock.patch.object(original, '_write', side_effect=change_after_copy):
            with self.assertRaises(PlanError):
                self.adopt()
        self.assertTrue((self.base / 'logs/initial-probe-adoption-claim.json').exists())
        self.assertFalse((self.base / 'artifacts/live/manifest.json').exists())


if __name__ == '__main__':
    unittest.main()
