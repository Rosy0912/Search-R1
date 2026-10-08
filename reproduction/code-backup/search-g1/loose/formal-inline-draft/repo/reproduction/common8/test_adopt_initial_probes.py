"""CPU byte/schema fixtures, never native execution or healthy-policy evidence."""
import copy
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from reproduction.common8 import adopt_initial_probes as adoption
from reproduction.common8.policy_calibration_plan import (
    CONFIRMATION_ROLE, DISCOVERY_ROLE, PlanError, canonical_bytes,
    create_discovery_plan, document_sha256,
)
from reproduction.common8 import test_policy_calibration_plan as plan_fixtures
from reproduction.common8 import test_freeze_calibration_plan as config_fixtures
from reproduction.common8 import test_model_smoke as model_fixtures
from reproduction.common8 import test_calibration_evaluation as evaluation_fixtures
from reproduction.common8 import calibration_evaluation as evaluation_module
from reproduction.common8.test_safety_calibration_evidence import fabricated_segment
from reproduction.common8.safety_calibration_evidence import verify_segment
from reproduction.common8.verify_data import digest


def write(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(document))
    return digest(path)


class AdoptionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name).resolve()
        self.source, self.base, self.model = root / 'discovery', root / 'production', root / 'official'
        for base in (self.source, self.base):
            (base / 'logs').mkdir(parents=True); (base / 'preparation').mkdir()
            (base / 'artifacts').mkdir()
        fixture = plan_fixtures.CalibrationPlanTests(); fixture.setUp()
        self.parent = copy.deepcopy(fixture.parent)
        self.model.mkdir()
        for name in self.parent['initial_model_sha256']:
            path = self.model / name; path.write_bytes(('CPU fixture official ' + name).encode())
            self.parent['initial_model_sha256'][name] = digest(path)
        self.parent.update(initial_model=str(self.model), reference_model=str(self.model),
                           initial_probe_version=None, calibration_question_ids=[f'q{i}' for i in range(300)],
                           checkpoint_dir=str(self.base / 'checkpoints'), train_log=str(self.base / 'logs/train.log'))
        self.round = self.source / 'artifacts/round_0-attempt-fixture'
        heads, dump = self.round / 'heads', self.round / 'dumps'
        heads.mkdir(parents=True); dump.mkdir()
        for name in ('commit_probe.pkl', 'boundary_probe.pkl'):
            (heads / name).write_bytes(('authenticated CPU bytes ' + name).encode())
        self.report = dict(commit_cv_auc=.7, boundary_cv_auc=.8, commit_position='Xa', commit_label='flip',
                           unique_question_count=300, commit_trajectory_count=6, question_grouped_cv=True,
                           boundary_unique_question_weighting=True, require_all_cv_folds=True, configured_cv_folds=2,
                           boundary_class_counts=[150, 150], commit_class_counts=[3, 3])
        self.label = dict(question_count=300, group_size=5, question_ids=self.parent['calibration_question_ids'],
                          searched_valid_trajectory_count=6, boundary_class_counts=[150, 150], reliance_class_counts=[3, 3],
                          closed_book_label='exact_match_no_retrieval',
                          reliance_label='normalized_answer_change_after_evidence_deletion',
                          model_authored_answer_spans=True, boundary_all_reserved_questions=True,
                          calibration_protocol=dict(rollout_max_new=500, correction_feedback=True),
                          evidence_deletion='typed_retrieval_ranges_only_preserve_model_and_correction')
        write(heads / 'calibration-report.json', self.report)
        write(dump / 'calibration-label-proof.json', self.label)
        records = [dict(qi=i, em=i % 2, flip=i % 2, cb=i % 2,
                        answer='CPU fixture raw answer', flip_answer='CPU fixture intervention',
                        counterfactual_valid=True, counterfactual_strict_valid=True) for i in range(6)]
        (dump / 'nq-hotpot-common_0.jsonl').write_bytes(b'\n'.join(canonical_bytes(r) for r in records) + b'\n')
        arrays = dict(boundary_question_id=np.array(self.label['question_ids']), boundary_q_cb=np.array([0, 1]*150),
                      y_flip=np.array([0, 1]*3), y_em=np.array([0, 1]*3), q_cb=np.array([0, 1]*3),
                      gid=np.arange(6, dtype=np.int32), question_id=np.array(self.label['question_ids'][:6]))
        for layer in (21, 24, 27):
            arrays['Xa_' + str(layer)] = np.zeros((6, 2), dtype=np.float16)
            arrays['Xp_question_' + str(layer)] = np.zeros((300, 2), dtype=np.float16)
        self.arrays = arrays
        np.savez(dump / 'nq-hotpot-common_0.npz', **arrays)
        self.live = dict(run_id='discovery', round=0, version='discovery:0:fixture', encoder=str(self.model),
                         commit=str(heads / 'commit_probe.pkl'), boundary=str(heads / 'boundary_probe.pkl'),
                         encoder_sha256={n: s for n, s in self.parent['initial_model_sha256'].items()
                                         if n not in ('vocab.json', 'merges.txt', 'generation_config.json')},
                         head_sha256={p.name: digest(p) for p in heads.glob('*.pkl')},
                         calibration_report=self.report, label_proof=self.label,
                         calibration_report_sha256=digest(heads / 'calibration-report.json'),
                         dump_sha256={p.name: digest(p) for p in dump.iterdir()}, published_unix=1.)
        write(self.round / 'version-manifest.json', self.live)
        write(self.source / 'artifacts/live/manifest.json', self.live)
        write(self.source / 'logs/initial-probes-accepted.json',
              dict(run_id='discovery', accepted=True, version=self.live['version'],
                   live_manifest_sha256=digest(self.source / 'artifacts/live/manifest.json'),
                   grouped_calibration_cv=self.report, label_proof=self.label))
        reward = dict(probe_round=0, encoder_sha256=self.live['encoder_sha256'],
                      head_sha256=self.live['head_sha256'], reward_math_sha256='7'*64)
        cfg_run = dict(self.parent, checkpoint_dir=str(self.source / 'checkpoints'))
        self.config = config_fixtures.FreezeCalibrationPlanTests.config_fixture(types.SimpleNamespace(base=self.source, run=cfg_run))
        config_sha = document_sha256(self.config)
        write(self.source / 'logs/resolved-training-config.json', dict(run_id='discovery', slurm_job_id='100',
              resolved_config=self.config, resolved_config_sha256=config_sha))
        model_rows, self.stage_ids, _ = model_fixtures.ModelSmokeTests().proofs()
        for phase in ('initial', 'coexistence'):
            rank_proofs, rank_shas = [], {}
            for rank in range(8):
                row = dict(model_rows[rank], run_id='discovery', slurm_job_id='100', phase=phase,
                           cuda_uuid='GPU-' + str(rank), initial_parameters_finite=True,
                           resolved_config_sha256=config_sha, actual_grpo_updates=0)
                rank_shas[str(rank)] = write(self.source / 'logs' / ('model-smoke-' + phase) / f'rank-{rank}.json', row)
                rank_proofs.append(row)
            write(self.source / 'logs' / ('model-smoke-' + phase) / 'complete.json',
                  dict(accepted=True, run_id='discovery', slurm_job_id='100', phase=phase,
                       ranks=rank_proofs, rank_file_sha256=rank_shas, actual_grpo_updates=0,
                       scope='Actual reader-schema CPU fixture only; no GPU execution'))
        self.plan = create_discovery_plan(self.parent, 'discovery', slurm_job_id='100',
                                          probability_proof_sha256=digest(self.source / 'logs/model-smoke-initial/complete.json'),
                                          resolved_config_sha256=config_sha,
                                          initial_reward_fingerprint=reward)
        sha = document_sha256(self.plan)
        parent_bytes = canonical_bytes(self.parent)
        self.source_run = dict(self.parent, run_id='discovery', initial_probe_version=self.live['version'],
                               checkpoint_dir=str(self.source / 'checkpoints'),
                               train_log=str(self.source / 'logs/train.log'),
                               parent_production_run_id='production', policy_calibration_role=DISCOVERY_ROLE,
                               policy_calibration_plan_sha256=sha, policy_calibration_slurm_job_id='100',
                               policy_calibration_parent_manifest_sha256=hashlib.sha256(parent_bytes).hexdigest())
        write(self.source / 'logs/run-manifest.json', self.source_run)
        (self.source / 'preparation/parent-run-manifest.json').write_bytes(parent_bytes)
        write(self.source / 'logs/policy-safety-calibration-plan.json', self.plan)
        self.run = dict(self.parent, policy_calibration_discovery_base=str(self.source))
        write(self.base / 'logs/run-manifest.json', self.run)
        self.segment = self.make_segment(sha)
        write(self.source / 'logs/policy-calibration-segment.json', self.segment)

    def make_segment(self, sha):
        logs, binding = fabricated_segment(role=DISCOVERY_ROLE)
        binding.update(run_id='discovery', calibration_sha256=sha)
        log_shas, state_shas, states, births = {}, {}, [], []
        for rank in range(8):
            for row in logs[rank]:
                row['run_id'] = 'discovery'
                if row['kind'] == 'POLICY_MONITOR': row['calibration_sha256'] = sha
            path = self.source / 'logs' / f'optimizer-rank-{rank}.jsonl'
            path.write_bytes(b'\n'.join(canonical_bytes(r) for r in logs[rank]) + b'\n')
            log_shas[str(rank)] = digest(path)
            state = dict(run_id='discovery', execution_role=DISCOVERY_ROLE, parent_production_run_id='production',
                         plan_sha256=sha, slurm_job_id='100', rank=rank, initial_optimizer_writes=0,
                         initial_adam_state_entries=0, final_optimizer_writes=20, pid=1000+rank, start_ticks=2000+rank,
                         parameter_tensor_count=1, parameter_numel=2, adam_state_entries=1,
                         adam_state_tensor_count=3, adam_state_numel=5, adam_steps=[20],
                         all_parameters_and_adam_finite=True, cuda_uuid='GPU-' + str(rank))
            state_shas[str(rank)] = write(self.source / 'logs' / f'policy-calibration-optimizer-rank-{rank}.json', state)
            states.append(state); births.append(dict(pid=state['pid'], start_ticks=state['start_ticks'], sid=10,
                                                    uid=os.getuid(), state='S'))
        write(self.source / 'logs/native-allocation-admission.json', dict(accepted=True, run_id='discovery',
              slurm_job_id='100', actual_cuda_runtime_uuids=['GPU-' + str(r) for r in range(8)]))
        parents = [dict(upstream_group_sha256=hashlib.sha256(str(i).encode()).hexdigest()) for i in range(512)]
        repeated = [r['upstream_group_sha256'] for r in parents for _ in range(5)]
        inputs = dict(run_id='discovery', plan_sha256=sha, actual_trajectory_count=2560, actual_parent_count=512,
                      n_agent=5, train_sha256=self.parent['input_sha256']['train.parquet'], parents=parents,
                      repeated_group_sha256=repeated)
        groups = dict(run_id='discovery', plan_sha256=sha, actual_trajectory_count=2560,
                      balanced_group_sha256=repeated[::-1], upstream_group_multiset_unchanged=True)
        input_shas = dict(parents=write(self.source / 'logs/policy-calibration-training-inputs.json', inputs),
                          groups=write(self.source / 'logs/policy-calibration-training-groups.json', groups))
        measured, self.validation_ids, self.validation_sources = evaluation_fixtures.canonical_rows()
        before = dict(evaluation_fixtures.aggregate(measured, self.validation_ids, self.validation_sources),
                      run_id='discovery', plan_sha256=sha, step=0, parent_production_run_id='production',
                      phase_role=DISCOVERY_ROLE, slurm_job_id='100',
                      validation_sha256=self.parent['input_sha256']['validation.parquet'],
                      question_ids=self.validation_ids, data_sources=self.validation_sources)
        after = dict(before, step=1)
        evaluation_shas = dict(before=write(self.source / 'logs/policy-calibration-evaluation-before.json', before),
                               after=write(self.source / 'logs/policy-calibration-evaluation-after.json', after))
        updates = [dict(run_id='discovery', step=1, probe_version=self.live['version'],
                        metrics={'actor/optimizer_steps_this_update': 20, 'critic/rewards/mean': .5})]
        (self.source / 'logs/actual-updates.jsonl').write_bytes(canonical_bytes(updates[0]) + b'\n')
        checkpoint = self.source / 'checkpoints/actor/global_step_1'
        checkpoint.mkdir(parents=True)
        (checkpoint / 'weights.safetensors').write_bytes(b'CPU fixture, not actual model tensors')
        saved = dict(run_id='discovery', step=1, finite_all_saved_tensors=True, validated_tensor_count=435,
                     checkpoint=str(checkpoint),
                     files_sha256={'weights.safetensors': digest(checkpoint / 'weights.safetensors')})
        write(checkpoint / 'complete.json', saved)
        return dict(role=DISCOVERY_ROLE + '_evidence', origin='actual_native_policy_segment', run_id='discovery',
                    parent_production_run_id='production', slurm_job_id='100', plan_sha256=sha,
                    identity=self.plan['identity'], initial_reward_fingerprint=self.plan['initial_reward_fingerprint'],
                    resolved_config_sha256=self.plan['resolved_config_sha256'], ranks=list(range(8)),
                    completed_outer_updates=1, optimizer_writes_per_rank=20, structural=verify_segment(logs, binding),
                    checkpoint=saved, changed_parameters=['fixture-key'], actual_optimizer_states=states,
                    actual_worker_births_at_segment_close=births, optimizer_state_proof_sha256=state_shas,
                    actual_training_inputs=inputs, actual_training_groups=groups, training_input_proof_sha256=input_shas,
                    worker_log_sha256=log_shas, evaluation_before=before, evaluation_after=after,
                    evaluation_sha256=evaluation_shas, actual_updates=updates,
                    production_admission=False, healthy_checkpoint_accepted=False)

    def adopt(self):
        # Native resources and fitted sklearn-pickle loading are explicit CPU
        # boundaries; real byte/plan/segment/NPZ checks and copying are executed.
        def expected(base, split):
            if split == 'stage': return None, None, self.stage_ids
            return None, {'data_source': np.array(self.validation_sources)}, self.validation_ids
        with mock.patch.object(adoption, '_actual_target', return_value='101'), \
             mock.patch.object(adoption, '_fitted_heads'), \
             mock.patch.dict(sys.modules, {'reproduction.common8.calibration_evaluation': evaluation_module}), \
             mock.patch('reproduction.common8.validation_identity.expected', side_effect=expected):
            return adoption.adopt(self.base, self.source)

    def test_real_bytes_new_version_no_admission_source_unchanged(self):
        original = {p: p.read_bytes() for p in self.source.rglob('*') if p.is_file()}
        receipt = self.adopt()
        live = json.loads((self.base / 'artifacts/live/manifest.json').read_text())
        self.assertTrue(receipt['adopted']); self.assertFalse(receipt['production_admission'])
        self.assertFalse(receipt['healthy_checkpoint_accepted']); self.assertTrue(receipt['child_http_admission_required'])
        self.assertTrue(live['version'].startswith('production:0:')); self.assertNotEqual(live['version'], self.live['version'])
        self.assertEqual(live['head_sha256'], self.live['head_sha256'])
        self.assertEqual(live['dump_sha256'], self.live['dump_sha256'])
        for key in ('commit', 'boundary'):
            self.assertEqual(Path(live[key]).read_bytes(), Path(self.live[key]).read_bytes())
        self.assertEqual(original, {p: p.read_bytes() for p in self.source.rglob('*') if p.is_file()})
        self.assertFalse((self.base / 'logs/initial-probes-accepted.json').exists())
        self.assertIsNone(json.loads((self.base / 'logs/run-manifest.json').read_text())['initial_probe_version'])
        with self.assertRaises(PlanError): self.adopt()

    def test_confirmation_child_same_parent_accepts_only_new_identity(self):
        self.run.update(run_id='confirmation', policy_calibration_role=CONFIRMATION_ROLE,
                        parent_production_run_id='production')
        write(self.base / 'logs/run-manifest.json', self.run)
        receipt = self.adopt()
        self.assertTrue(receipt['version'].startswith('confirmation:0:'))
        self.assertEqual(receipt['parent_production_run_id'], 'production')

    def test_mutable_parent_placement_does_not_change_frozen_parent_bytes(self):
        self.run.update(retrieval_placement='new actual native resource placement', probe_visible_device=7)
        write(self.base / 'logs/run-manifest.json', self.run)
        self.assertTrue(self.adopt()['adopted'])

    def test_scope_identity_and_explicit_source_reject_before_claim(self):
        for field, value in (('policy_calibration_discovery_base', '/unbound'), ('run_id', 'other-parent'),
                             ('policy_calibration_role', DISCOVERY_ROLE), ('initial_probe_version', 'copied-old'),
                             ('completed_updates', 1), ('source_commit', '9'*40)):
            with self.subTest(field=field):
                changed = dict(self.run, **{field: value}); write(self.base / 'logs/run-manifest.json', changed)
                with self.assertRaises(PlanError): self.adopt()
                self.assertFalse((self.base / 'logs/initial-probe-adoption-claim.json').exists())
        write(self.base / 'logs/run-manifest.json', self.run)

    def test_existing_segment_or_claim_never_reset(self):
        for name in ('policy-calibration-segment.json', 'training-launch-claim.json', 'actual-updates.jsonl'):
            path = self.base / 'logs' / name; path.write_bytes(b'existing evidence')
            with self.assertRaises(PlanError): self.adopt()
            self.assertEqual(path.read_bytes(), b'existing evidence'); path.unlink()
        (self.base / 'logs/initial-probe-adoption-claim.json').write_bytes(b'partial original claim')
        with self.assertRaises(FileExistsError): self.adopt()
        self.assertEqual((self.base / 'logs/initial-probe-adoption-claim.json').read_bytes(), b'partial original claim')

    def test_changed_actual_rank_segment_weights_reject(self):
        for path in (self.source / 'logs/optimizer-rank-7.jsonl',
                     self.source / 'checkpoints/actor/global_step_1/weights.safetensors',
                     self.source / 'logs/policy-calibration-evaluation-after.json'):
            raw = path.read_bytes(); path.write_bytes(raw + b' ')
            with self.assertRaises(ValueError): self.adopt()
            path.write_bytes(raw)
        changed = dict(self.segment, ranks=[False, True, 2, 3, 4, 5, 6, 7])
        write(self.source / 'logs/policy-calibration-segment.json', changed)
        with self.assertRaises(PlanError): self.adopt()

    def test_actual_probability_and_recipe_and_fixed512_arithmetic_reject(self):
        rank = self.source / 'logs/model-smoke-initial/rank-7.json'
        original = rank.read_bytes(); rank.write_bytes(original + b' ')
        with self.assertRaises(ValueError): self.adopt()
        rank.write_bytes(original)
        path = self.source / 'logs/resolved-training-config.json'
        original = path.read_bytes(); config = json.loads(original)
        config['resolved_config']['actor_rollout_ref']['actor']['ppo_mini_batch_size'] = 129
        write(path, config)
        with self.assertRaises(ValueError): self.adopt()
        path.write_bytes(original)
        path = self.source / 'logs/policy-calibration-evaluation-after.json'
        original = path.read_bytes(); after = json.loads(original)
        after['weighted_em'] += .1
        raw = canonical_bytes(after); path.write_bytes(raw)
        segment = copy.deepcopy(self.segment)
        segment['evaluation_after'] = after
        segment['evaluation_sha256']['after'] = hashlib.sha256(raw).hexdigest()
        write(self.source / 'logs/policy-calibration-segment.json', segment)
        # Rebound plausible scalar summaries still fail the real row aggregation.
        with self.assertRaises(ValueError): self.adopt()

    def test_missing_archive_extra_dump_or_symlink_reject(self):
        archive = self.round / 'version-manifest.json'; raw = archive.read_bytes(); archive.unlink()
        with self.assertRaises(PlanError): self.adopt()
        archive.write_bytes(raw)
        extra = self.round / 'dumps/unfamiliar.txt'; extra.write_text('extra')
        with self.assertRaises(PlanError): self.adopt()
        extra.unlink()
        head = self.round / 'heads/commit_probe.pkl'; raw = head.read_bytes(); head.unlink()
        outside = Path(self.temporary.name) / 'foreign.pkl'; outside.write_bytes(raw); head.symlink_to(outside)
        with self.assertRaises(PlanError): self.adopt()

    def test_actual_npz_nonfinite_or_labels_or_question_group_reject(self):
        dump = self.round / 'dumps'
        for key, value in (('Xa_21', np.full((6, 2), np.inf)), ('y_flip', np.array([0, 1, 2, 0, 1, 0])),
                           ('gid', np.array([0, 1, 2, 3, 4, 300], dtype=np.int32))):
            with self.subTest(key=key):
                np.savez(dump / 'nq-hotpot-common_0.npz', **dict(self.arrays, **{key: value}))
                with self.assertRaises(PlanError): adoption._calibration_arrays(dump, self.report, self.label)

    def test_partial_copy_claim_preserved_without_new_live_or_accepted(self):
        real = adoption._write
        def fail(path, raw):
            if Path(path).name == 'boundary_probe.pkl': raise OSError('CPU fixture interrupted copy')
            return real(path, raw)
        with mock.patch.object(adoption, '_write', side_effect=fail):
            with self.assertRaises(OSError): self.adopt()
        self.assertTrue((self.base / 'logs/initial-probe-adoption-claim.json').exists())
        self.assertFalse((self.base / 'artifacts/live/manifest.json').exists())
        self.assertFalse((self.base / 'logs/initial-probes-accepted.json').exists())
        with self.assertRaises(FileExistsError): self.adopt()

    @unittest.skipUnless(importlib.util.find_spec('joblib') and importlib.util.find_spec('sklearn'),
                         'Mac lacks joblib/sklearn; native target must execute this real fitted CPU test')
    def test_real_fitted_head_arrays_and_semantics(self):
        import joblib
        from sklearn.linear_model import LogisticRegression
        heads = self.round / 'heads'
        clf = LogisticRegression().fit(np.array([[0., 0.], [1., 1.], [0., 1.], [1., 0.]]), [0, 1, 0, 1])
        for name, key, position, label in (('commit_probe.pkl', 'commit', 'Xa', 'flip'),
                                           ('boundary_probe.pkl', 'boundary', 'Xp', 'closedbook')):
            joblib.dump(dict(clf=copy.deepcopy(clf), position=position, label=label,
                             auc=self.report[key + '_cv_auc']), heads / name)
        adoption._fitted_heads(heads, self.report)
        clf.coef_[0, 0] = np.inf
        joblib.dump(dict(clf=clf, position='Xa', label='flip', auc=.7), heads / 'commit_probe.pkl')
        with self.assertRaises(AssertionError): adoption._fitted_heads(heads, self.report)


if __name__ == '__main__':
    unittest.main()
