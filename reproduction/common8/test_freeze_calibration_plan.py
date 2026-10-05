"""Real CPU file/hash/atomic tests with explicit mocked native/HTTP boundaries.

These fixtures neither execute GPUs nor provide real calibration/health proof.
"""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from reproduction.common8.freeze_calibration_plan import freeze, _mathematics, _resolved_document
from reproduction.common8.policy_calibration_plan import (
    CONFIRMATION_ROLE, DISCOVERY_ROLE, PlanError, canonical_bytes, document_sha256, freeze_rule,
)
from reproduction.common8 import test_policy_calibration_plan as plan_fixtures


class FreezeCalibrationPlanTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name).resolve()
        for name in ('logs/model-smoke-initial', 'preparation', 'artifacts/live', 'model', 'heads'):
            (self.base / name).mkdir(parents=True)
        fixture = plan_fixtures.CalibrationPlanTests(); fixture.setUp()
        self.parent = copy.deepcopy(fixture.parent)
        repo = Path(__file__).resolve().parents[2]
        names = ('verl/trainer/main_ppo_eco.py', 'search_g1/signals.py',
                 'verl/utils/reward_score/qa_em.py', 'search_r1/qa_protocol.py')
        self.parent['source_sha256'] = {name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in names}
        self.parent['initial_model'] = self.parent['reference_model'] = str(self.base / 'model')
        for name in self.parent['initial_model_sha256']:
            (self.base / 'model' / name).write_bytes(name.encode())
        parent_bytes = canonical_bytes(self.parent)
        (self.base / 'preparation/parent-run-manifest.json').write_bytes(parent_bytes)
        self.run = dict(self.parent, run_id='discovery', policy_calibration_role=DISCOVERY_ROLE,
                        parent_production_run_id='production',
                        policy_calibration_parent_manifest_sha256=hashlib.sha256(parent_bytes).hexdigest(),
                        checkpoint_dir=str(self.base / 'checkpoints/search-g1-common8'),
                        train_log=str(self.base / 'logs/search-g1-common8-training.log'),
                        initial_probe_version='discovery:0:fixture')
        self.env = dict(SLURM_JOB_ID='3265', ECO_TRUST_VARIANT='balanced', ECO_TRUST_SIGNAL='probe', ECO_PROBE_GATE='1',
                        ECO_BAL_GROUND='0.6', ECO_BAL_NOSEARCH_BONUS='0.5', ECO_W_COST='0.02',
                        ECO_WRONG_GROUND_SCALE='0', ECO_PROBE_BASE='0', ECO_PROBE_ALPHA_COST='0',
                        ECO_USE_CF='0', ECO_NO_SELF='1', ECO_PROBE_MAX_WAIT='300')
        self.document = self.config_fixture()
        self.write_child_identity()

    def write_child_identity(self):
        (self.base / 'logs/run-manifest.json').write_bytes(canonical_bytes(self.run))
        # This is explicitly a CPU stub file, not an accepted native model proof.
        (self.base / 'logs/model-smoke-initial/complete.json').write_bytes(
            canonical_bytes(dict(cpu_fixture=True, run_id=self.run['run_id'])))
        heads = {name: self.base / 'heads' / name for name in ('commit_probe.pkl', 'boundary_probe.pkl')}
        for name, path in heads.items(): path.write_bytes(name.encode())
        encoder = {name: digest for name, digest in self.run['initial_model_sha256'].items()
                   if name not in ('vocab.json', 'merges.txt', 'generation_config.json')}
        self.live = dict(run_id=self.run['run_id'], round=0, version=self.run['initial_probe_version'],
                         encoder=self.run['initial_model'], encoder_sha256=encoder,
                         head_sha256={name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in heads.items()},
                         commit=str(heads['commit_probe.pkl']), boundary=str(heads['boundary_probe.pkl']))
        raw = canonical_bytes(self.live)
        (self.base / 'artifacts/live/manifest.json').write_bytes(raw)
        accepted = dict(accepted=True, run_id=self.run['run_id'], version=self.run['initial_probe_version'],
                        live_manifest_sha256=hashlib.sha256(raw).hexdigest())
        (self.base / 'logs/initial-probes-accepted.json').write_bytes(canonical_bytes(accepted))

    def config_fixture(self):
        actor = dict(ppo_mini_batch_size=128, ppo_micro_batch_size=16, ppo_epochs=1, grad_clip=1.0,
                     clip_ratio=0.2, state_masking=True, use_kl_loss=True, kl_loss_coef=0.001,
                     kl_loss_type='low_var_kl', use_dynamic_bsz=False, ulysses_sequence_parallel_size=1,
                     compute_dtype='float32', allow_tf32=False,
                     optim=dict(lr=1e-6, lr_warmup_steps_ratio=0.0, warmup_style='constant'),
                     fsdp_config=dict(model_dtype='fp32', mixed_precision={k: 'fp32' for k in
                                      ('param_dtype', 'reduce_dtype', 'buffer_dtype')}),
                     policy_safety=dict(enabled=True, calibration_manifest=str(self.base / 'logs/policy-safety-calibration.json'),
                                        dual_clip=3.0, target_old_policy_kl=0.01, kl_log_bound=30.0, negative_tail_limit=None))
        return dict(actor_rollout_ref=dict(actor=actor,
                    ref=dict(log_prob_micro_batch_size=32, compute_dtype='float32', allow_tf32=False,
                             fsdp_config=dict(model_dtype='fp32')),
                    rollout=dict(n=1, n_agent=5, temperature=1.0, top_p=.95, log_prob_micro_batch_size=32, dtype='float32'),
                    model=dict(path=self.run['initial_model'])),
                    algorithm=dict(adv_estimator='grpo', no_think_rl=False),
                    data=dict(train_batch_size=512, val_batch_size=64, max_prompt_length=4096, max_response_length=500,
                              max_start_length=2048, max_obs_length=500, train_files=str(self.base / 'data/train.parquet'),
                              val_files=str(self.base / 'data/stage.parquet')),
                    trainer=dict(n_gpus_per_node=8, nnodes=1, total_training_steps=301, total_epochs=15,
                                 save_freq=50, test_freq=50, acceptance_step=5,
                                 default_local_dir=self.run['checkpoint_dir'], val_only=False, val_before_train=True))

    def invoke(self):
        with mock.patch('reproduction.common8.freeze_calibration_plan._resolved_document', return_value=self.document), \
             mock.patch('reproduction.common8.freeze_calibration_plan._actual_prerequisites') as native:
            result = freeze(self.base, self.env)
            if self.run.get('policy_calibration_role') != 'production': native.assert_called_once()
            return result

    def test_discovery_freezes_actual_file_hashes_and_atomic_manifest_without_health(self):
        result = self.invoke()
        self.assertTrue(result['plan_created'])
        self.assertFalse(result['production_admission'])
        self.assertFalse(result['healthy_checkpoint_accepted'])
        plan = json.loads((self.base / 'logs/policy-safety-calibration-plan.json').read_text())
        child = json.loads((self.base / 'logs/run-manifest.json').read_text())
        self.assertEqual(child['policy_calibration_plan_sha256'], document_sha256(plan))
        self.assertEqual(child['policy_calibration_slurm_job_id'], '3265')
        self.assertEqual(plan['resolved_config_sha256'], document_sha256(self.document))
        self.assertEqual(plan['initial_probability_gate']['proof_sha256'],
                         hashlib.sha256((self.base / 'logs/model-smoke-initial/complete.json').read_bytes()).hexdigest())
        self.assertEqual(plan['initial_reward_fingerprint']['head_sha256'], self.live['head_sha256'])
        self.assertEqual(len(plan['initial_reward_fingerprint']['encoder_sha256']), 6)
        self.assertIsNone(plan['candidate_thresholds']['negative_tail_limit'])
        self.assertFalse((self.base / 'logs/training-launch-claim.json').exists())

    def test_production_is_read_only_and_never_creates_plan(self):
        self.run.pop('policy_calibration_role')
        raw = canonical_bytes(self.run); (self.base / 'logs/run-manifest.json').write_bytes(raw)
        with mock.patch('reproduction.common8.freeze_calibration_plan._actual_prerequisites') as native:
            result = freeze(self.base, {})
            native.assert_not_called()
        self.assertEqual(result['role'], 'production')
        self.assertFalse(result['plan_created'])
        self.assertEqual((self.base / 'logs/run-manifest.json').read_bytes(), raw)
        self.assertFalse((self.base / 'logs/policy-safety-calibration-plan.json').exists())

    def test_identical_preclaim_replay_does_not_overwrite_frozen_plan(self):
        first = self.invoke()
        target = self.base / 'logs/policy-safety-calibration-plan.json'
        original = target.read_bytes(); before = target.stat().st_mtime_ns
        second = self.invoke()
        self.assertFalse(second['plan_created'])
        self.assertEqual(first['plan_sha256'], second['plan_sha256'])
        self.assertEqual(target.read_bytes(), original)
        self.assertEqual(target.stat().st_mtime_ns, before)

    def test_crash_between_plan_and_manifest_can_only_complete_identical_unclaimed_binding(self):
        with mock.patch('reproduction.common8.freeze_calibration_plan._replace_manifest', side_effect=RuntimeError('fixture interruption')):
            with self.assertRaises(RuntimeError): self.invoke()
        target = self.base / 'logs/policy-safety-calibration-plan.json'
        original = target.read_bytes()
        self.assertNotIn('policy_calibration_plan_sha256', json.loads((self.base / 'logs/run-manifest.json').read_text()))
        result = self.invoke()
        self.assertFalse(result['plan_created'])
        self.assertEqual(target.read_bytes(), original)
        self.assertIn('policy_calibration_plan_sha256', json.loads((self.base / 'logs/run-manifest.json').read_text()))

    def test_claim_partial_writes_log_or_checkpoint_blocks_even_same_plan(self):
        self.invoke()
        paths = [self.base / 'logs/training-launch-claim.json', self.base / 'logs/optimizer-rank-7.jsonl',
                 self.base / 'logs/policy-calibration-optimizer-rank-0.json', self.base / 'logs/actual-updates.jsonl',
                 Path(self.run['train_log']), Path(self.run['checkpoint_dir']) / 'partial']
        before = (self.base / 'logs/run-manifest.json').read_bytes()
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'partial evidence')
            with self.subTest(path=path), self.assertRaises(PlanError): self.invoke()
            self.assertEqual((self.base / 'logs/run-manifest.json').read_bytes(), before)
            path.unlink()

    def test_native_or_model_admission_failure_cannot_write_plan(self):
        with mock.patch('reproduction.common8.freeze_calibration_plan._resolved_document', return_value=self.document), \
             mock.patch('reproduction.common8.freeze_calibration_plan._actual_prerequisites', side_effect=PlanError('fixture native rejection')):
            with self.assertRaises(PlanError): freeze(self.base, self.env)
        self.assertFalse((self.base / 'logs/policy-safety-calibration-plan.json').exists())
        self.assertNotIn('policy_calibration_plan_sha256', json.loads((self.base / 'logs/run-manifest.json').read_text()))

    def test_existing_segment_alone_cannot_be_reset_or_refrozen(self):
        path = self.base / 'logs/policy-calibration-segment.json'
        evidence = canonical_bytes(dict(run_id='discovery', actual_outer_updates=1, fixture=True))
        path.write_bytes(evidence)
        before = (self.base / 'logs/run-manifest.json').read_bytes()
        with self.assertRaises(PlanError): self.invoke()
        self.assertEqual(path.read_bytes(), evidence)
        self.assertEqual((self.base / 'logs/run-manifest.json').read_bytes(), before)
        self.assertFalse((self.base / 'logs/policy-safety-calibration-plan.json').exists())

    def test_actual_math_precision_sampling_or_scheduler_cannot_drift(self):
        for field in ('mini', 'micro', 'precision', 'lr', 'temperature', 'horizon', 'tail'):
            document = copy.deepcopy(self.document)
            actor = document['actor_rollout_ref']['actor']
            if field == 'mini': actor['ppo_mini_batch_size'] = 256
            elif field == 'micro': actor['ppo_micro_batch_size'] = 128
            elif field == 'precision': actor['compute_dtype'] = 'bfloat16'
            elif field == 'lr': actor['optim']['lr'] = 1e-5
            elif field == 'temperature': document['actor_rollout_ref']['rollout']['temperature'] = .8
            elif field == 'horizon': document['trainer']['total_training_steps'] = 2
            else: actor['policy_safety']['negative_tail_limit'] = 1
            with self.subTest(field=field), mock.patch('reproduction.common8.freeze_calibration_plan._resolved_document', return_value=document):
                with self.assertRaises(PlanError): freeze(self.base, self.env)
            self.assertFalse((self.base / 'logs/policy-safety-calibration-plan.json').exists())

    def test_real_hydra_resolved_recipe_matches_math_without_model_or_data_execution(self):
        try:
            import hydra  # noqa: F401
        except ImportError:
            self.skipTest('Local Hydra unavailable; actual target must run this test without skipping')
        from reproduction.common8 import configuration
        run = copy.deepcopy(self.run)
        for name in ('reproduction/common8/recipe.json', 'verl/trainer/config/ppo_trainer.yaml'):
            run['source_sha256'][name] = hashlib.sha256((configuration.REPO / name).read_bytes()).hexdigest()
        (self.base / 'logs/run-manifest.json').write_bytes(canonical_bytes(run))
        document = _resolved_document(self.base, run)
        _mathematics(document, self.base, run)
        self.assertEqual(document['actor_rollout_ref']['actor']['ppo_mini_batch_size'], 128)
        self.assertEqual(document['actor_rollout_ref']['actor']['ppo_micro_batch_size'], 16)
        self.assertEqual(document['trainer']['total_training_steps'], 301)
        self.assertEqual(document['actor_rollout_ref']['actor']['optim']['warmup_style'], 'constant')

    def test_changed_parent_head_gate_or_reward_cannot_overwrite_plan(self):
        self.invoke()
        before = (self.base / 'logs/policy-safety-calibration-plan.json').read_bytes()
        for field in ('parent', 'head', 'gate', 'reward'):
            if field == 'parent': path = self.base / 'preparation/parent-run-manifest.json'
            elif field == 'head': path = Path(self.live['commit'])
            elif field == 'gate': path = self.base / 'logs/model-smoke-initial/complete.json'
            else: path = None
            raw = path.read_bytes() if path else None
            if path: path.write_bytes(raw + b' ')
            else: self.env['ECO_UNEXPECTED_FLAG'] = 'changed'
            with self.subTest(field=field), self.assertRaises(PlanError): self.invoke()
            self.assertEqual((self.base / 'logs/policy-safety-calibration-plan.json').read_bytes(), before)
            if path: path.write_bytes(raw)
            else: self.env.pop('ECO_UNEXPECTED_FLAG')

    def prepare_confirmation(self):
        self.invoke()
        discovery = json.loads((self.base / 'logs/policy-safety-calibration-plan.json').read_text())
        evidence = dict(role='policy_safety_calibration_discovery_evidence', origin='actual_native_policy_segment',
                        run_id='discovery', parent_production_run_id='production', slurm_job_id='3265',
                        plan_sha256=document_sha256(discovery), identity=discovery['identity'],
                        initial_reward_fingerprint=discovery['initial_reward_fingerprint'],
                        resolved_config_sha256=discovery['resolved_config_sha256'],
                        ranks=list(range(8)), completed_outer_updates=1, optimizer_writes_per_rank=20)
        raw = canonical_bytes(evidence)
        rule = freeze_rule(dict(role='policy_safety_health_rule', criteria=dict(definition='CPU fixture only'),
                                provenance=dict(origin='fixture, not a real healthy judgment')))
        thresholds = dict(discovery['candidate_thresholds'], negative_tail_limit=.02)
        selection = freeze_rule(dict(role='policy_safety_threshold_selection', provenance=dict(method='CPU fixture only'),
                    criteria=dict(discovery_evidence_sha256=hashlib.sha256(raw).hexdigest(), health_rule_sha256=rule['sha256'],
                                  candidate_thresholds=thresholds)))
        candidate = dict(schema_version=1, role='policy_safety_confirmation_candidates', parent_production_run_id='production',
                         confirmation_run_id='confirmation', discovery_run_id='discovery',
                         discovery_evidence_sha256=hashlib.sha256(raw).hexdigest(), negative_tail_limit=.02,
                         health_rule=rule, threshold_selection=selection)
        for name, data in (('policy-safety-discovery-plan.json', canonical_bytes(discovery)),
                           ('policy-safety-discovery-evidence.json', raw),
                           ('policy-safety-confirmation-candidates.json', canonical_bytes(candidate))):
            (self.base / 'preparation' / name).write_bytes(data)
        (self.base / 'logs/policy-safety-calibration-plan.json').unlink()  # Fixture namespace becomes a separate simulated child.
        self.run.update(run_id='confirmation', policy_calibration_role=CONFIRMATION_ROLE,
                        initial_probe_version='confirmation:0:fixture')
        self.env['SLURM_JOB_ID'] = '3266'
        self.write_child_identity()
        return candidate

    def test_confirmation_requires_explicit_frozen_real_provenance_and_same_reward(self):
        self.prepare_confirmation()
        result = self.invoke()
        plan = json.loads((self.base / 'logs/policy-safety-calibration-plan.json').read_text())
        self.assertEqual(plan['role'], CONFIRMATION_ROLE)
        self.assertEqual(plan['candidate_thresholds']['negative_tail_limit'], .02)
        self.assertFalse(result['healthy_checkpoint_accepted'])

    def test_confirmation_missing_candidates_or_changed_heads_rejects(self):
        self.prepare_confirmation()
        candidate = self.base / 'preparation/policy-safety-confirmation-candidates.json'
        raw = candidate.read_bytes(); candidate.unlink()
        with self.assertRaises(FileNotFoundError): self.invoke()
        candidate.write_bytes(raw)
        head = Path(self.live['commit']); head.write_bytes(b'newly refitted different head')
        live = dict(self.live, head_sha256=dict(self.live['head_sha256'],
                    **{'commit_probe.pkl': hashlib.sha256(head.read_bytes()).hexdigest()}))
        data = canonical_bytes(live); (self.base / 'artifacts/live/manifest.json').write_bytes(data)
        accepted = dict(accepted=True, run_id=self.run['run_id'], version=self.run['initial_probe_version'],
                        live_manifest_sha256=hashlib.sha256(data).hexdigest())
        (self.base / 'logs/initial-probes-accepted.json').write_bytes(canonical_bytes(accepted))
        with self.assertRaises(PlanError): self.invoke()
        self.assertFalse((self.base / 'logs/policy-safety-calibration-plan.json').exists())


if __name__ == '__main__':
    unittest.main()
