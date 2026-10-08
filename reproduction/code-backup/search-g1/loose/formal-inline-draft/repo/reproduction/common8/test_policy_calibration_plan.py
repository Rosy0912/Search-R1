"""CPU-only structural fixtures; none are native calibration or admission."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from reproduction.common8.policy_calibration_plan import (
    COMMON_MANIFEST_SHA256, CONFIRMATION_ROLE, DISCOVERY_ROLE, MATHEMATICS,
    MODEL_ID, MODEL_REVISION, PlanError, canonical_bytes, create_confirmation_plan,
    create_discovery_plan, document_sha256, freeze_rule, load_plan, verify_plan,
)


class CalibrationPlanTests(unittest.TestCase):
    def setUp(self):
        budget = json.loads((Path(__file__).parent / 'budget.json').read_text())
        # These fixtures authenticate the immutable old diagnostic contract;
        # changing today's formal recipe must not relabel its mathematics.
        budget.update(copy.deepcopy(MATHEMATICS))
        for name in ('contract_kind','lr_warmup_steps','entropy_coefficient','policy_safety_mode',
                     'safeguards','optimizer_opportunities_per_outer','optimizer_completion_rule',
                     'optimizer_step_count_semantics'):
            budget.pop(name,None)
        self.parent = dict(profile='common8', run_id='production', completed_updates=0,
                           model_revision=MODEL_REVISION, initial_model='/official', reference_model='/official',
                           source_commit='1'*40, source_sha256={'source.py': '1'*64},
                           initial_model_sha256={name: hashlib.sha256(name.encode()).hexdigest() for name in
                                                ('config.json', 'model.safetensors.index.json',
                                                 'model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors',
                                                 'tokenizer.json', 'tokenizer_config.json',
                                                 'vocab.json', 'merges.txt', 'generation_config.json')},
                           input_sha256=dict({'manifest.json': COMMON_MANIFEST_SHA256},
                                             **{name + '.parquet': hashlib.sha256(name.encode()).hexdigest()
                                                for name in ('train', 'validation', 'stage', 'final')}), budget=budget)
        self.rule = freeze_rule(dict(role='policy_safety_health_rule',
                                     provenance=dict(review='CPU fixture, not a real health decision'),
                                     criteria=dict(definition='Caller-defined rule kept opaque')))
        self.reward = dict(probe_round=0,
                           encoder_sha256={name: sha for name, sha in self.parent['initial_model_sha256'].items()
                                           if name not in ('vocab.json', 'merges.txt', 'generation_config.json')},
                           head_sha256={'commit_probe.pkl': '5'*64, 'boundary_probe.pkl': '6'*64},
                           reward_math_sha256='7'*64)
        self.discovery = create_discovery_plan(self.parent, 'discovery',
                                               slurm_job_id='3265', probability_proof_sha256='3'*64,
                                               resolved_config_sha256='8'*64, initial_reward_fingerprint=self.reward)

    def verify(self, plan, **kwargs):
        return verify_plan(plan, self.parent, expected_plan_sha256=document_sha256(plan), **kwargs)

    def confirmation(self):
        evidence = dict(role='policy_safety_calibration_discovery_evidence', origin='actual_native_policy_segment',
                        run_id='discovery', parent_production_run_id='production', slurm_job_id='3265',
                        plan_sha256=document_sha256(self.discovery), identity=self.discovery['identity'],
                        initial_reward_fingerprint=self.reward, resolved_config_sha256='8'*64,
                        ranks=list(range(8)), completed_outer_updates=1, optimizer_writes_per_rank=20)
        raw = canonical_bytes(evidence)
        threshold = dict(self.discovery['candidate_thresholds'], negative_tail_limit=0.02)
        selection = freeze_rule(dict(role='policy_safety_threshold_selection',
                                    provenance=dict(method='fixture only; no empirical threshold selection'),
                                    criteria=dict(discovery_evidence_sha256=hashlib.sha256(raw).hexdigest(),
                                                  health_rule_sha256=self.rule['sha256'], candidate_thresholds=threshold)))
        plan = create_confirmation_plan(self.discovery, self.parent, 'confirmation',
                                        slurm_job_id='3266', probability_proof_sha256='4'*64, resolved_config_sha256='9'*64,
                                        discovery_evidence_bytes=raw,
                                        negative_tail_limit=0.02, threshold_selection=selection, health_rule=self.rule)
        return plan, raw

    def test_discovery_never_grants_admission_or_fake_tail(self):
        result = self.verify(self.discovery)
        self.assertEqual(result['execution_role'], DISCOVERY_ROLE)
        self.assertFalse(result['production_admission'])
        self.assertFalse(result['healthy_checkpoint_accepted'])
        self.assertFalse(result['physical_execution_authenticated'])
        self.assertEqual(result['actual_grpo_updates_proven'], 0)
        self.assertNotIn('accepted', result)
        self.assertIsNone(self.discovery['candidate_thresholds']['negative_tail_limit'])
        self.assertEqual(self.discovery['threshold_policy']['negative_tail'], 'observe_only')
        self.assertIsNone(self.discovery['health_rule'])

    def test_confirmation_is_separate_fresh_and_frozen_not_execution_proof(self):
        plan, raw = self.confirmation()
        result = self.verify(plan, discovery_plan=self.discovery, discovery_evidence_bytes=raw)
        self.assertEqual(result['execution_role'], CONFIRMATION_ROLE)
        self.assertFalse(result['production_admission'])
        self.assertEqual(plan['phase_budget']['outer_count'], 1)
        self.assertEqual(plan['phase_budget']['optimizer_writes_per_rank'], 20)
        self.assertEqual(plan['initialization']['adam_state'], 'empty')
        self.assertIsNone(plan['initialization']['inherited_policy_run_id'])

    def write_plan_files(self, directory):
        base = Path(directory)
        (base / 'logs').mkdir(); (base / 'preparation').mkdir()
        parent_bytes = canonical_bytes(self.parent)
        run = dict(self.parent, run_id='discovery', parent_production_run_id='production',
                   policy_calibration_role=DISCOVERY_ROLE,
                   policy_calibration_plan_sha256=document_sha256(self.discovery),
                   policy_calibration_parent_manifest_sha256=hashlib.sha256(parent_bytes).hexdigest(),
                   policy_calibration_slurm_job_id='3265')
        (base / 'logs/run-manifest.json').write_bytes(canonical_bytes(run))
        (base / 'preparation/parent-run-manifest.json').write_bytes(parent_bytes)
        (base / 'logs/policy-safety-calibration-plan.json').write_bytes(canonical_bytes(self.discovery))
        return base, run

    def test_hash_mismatch_duplicate_keys_and_nonfinite_json(self):
        with self.assertRaises(PlanError):
            verify_plan(self.discovery, self.parent, expected_plan_sha256='0'*64)
        with tempfile.TemporaryDirectory() as directory:
            base, run = self.write_plan_files(directory)
            plan, digest = load_plan(base, run, expected_slurm_job_id='3265')
            self.assertEqual(plan, self.discovery)
            self.assertEqual(digest, document_sha256(plan))
            for raw in (b'{"run_id":"a","run_id":"b"}', b'{"x":NaN}', b'not json'):
                (base / 'logs/policy-safety-calibration-plan.json').write_bytes(raw)
                with self.assertRaises(PlanError): load_plan(base, run, expected_slurm_job_id='3265')

    def test_load_checks_actual_child_manifest_parent_bytes_role_and_job(self):
        for change in ('childsource', 'parentbytes', 'role', 'job', 'planhash', 'stale'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                base, run = self.write_plan_files(directory)
                expected_job = '3265'
                if change == 'childsource': run['source_sha256']['source.py'] = 'a'*64
                elif change == 'parentbytes':
                    with (base / 'preparation/parent-run-manifest.json').open('ab') as f: f.write(b' ')
                elif change == 'role': run['policy_calibration_role'] = 'production'
                elif change == 'job': expected_job = '3266'
                elif change == 'planhash': run['policy_calibration_plan_sha256'] = 'a'*64
                else: run['completed_updates'] = 1
                if change != 'stale': (base / 'logs/run-manifest.json').write_bytes(canonical_bytes(run))
                with self.assertRaises(PlanError): load_plan(base, run, expected_slurm_job_id=expected_job)

    def test_default_loader_uses_native_environment_not_manifest_job(self):
        with tempfile.TemporaryDirectory() as directory:
            base, run = self.write_plan_files(directory)
            with mock.patch.dict('os.environ', {'SLURM_JOB_ID': '3265'}, clear=True):
                self.assertEqual(load_plan(base, run)[1], document_sha256(self.discovery))
            for environment in ({}, {'SLURM_JOB_ID': '3266'}, {'SLURM_JOB_ID': 'CPU-fixture'}):
                with self.subTest(environment=environment), mock.patch.dict('os.environ', environment, clear=True):
                    with self.assertRaises(PlanError): load_plan(base, run)

    def test_confirmation_loader_reads_frozen_original_discovery_artifacts(self):
        plan, raw = self.confirmation()
        with tempfile.TemporaryDirectory() as directory:
            base, run = self.write_plan_files(directory)
            run.update(run_id='confirmation', policy_calibration_role=CONFIRMATION_ROLE,
                       policy_calibration_plan_sha256=document_sha256(plan),
                       policy_calibration_slurm_job_id='3266')
            (base / 'logs/run-manifest.json').write_bytes(canonical_bytes(run))
            (base / 'logs/policy-safety-calibration-plan.json').write_bytes(canonical_bytes(plan))
            discovery_path = base / 'preparation/policy-safety-discovery-plan.json'
            evidence_path = base / 'preparation/policy-safety-discovery-evidence.json'
            discovery_path.write_bytes(canonical_bytes(self.discovery))
            evidence_path.write_bytes(raw)
            self.assertEqual(load_plan(base, run, expected_slurm_job_id='3266')[1], document_sha256(plan))
            evidence_path.write_bytes(raw + b' ')
            with self.assertRaises(PlanError): load_plan(base, run, expected_slurm_job_id='3266')

    def test_reject_cross_run_source_model_data_and_math_even_with_new_plan_hash(self):
        for field in ('parent_production_run_id', 'source', 'model', 'data', 'math'):
            plan = copy.deepcopy(self.discovery)
            if field == 'parent_production_run_id': plan[field] = 'other'
            elif field == 'source': plan['identity']['source_sha256']['source.py'] = 'a'*64
            elif field == 'model': plan['identity']['initial_model_sha256']['config.json'] = 'a'*64
            elif field == 'data': plan['identity']['input_sha256']['validation.parquet'] = 'a'*64
            else: plan['identity']['mathematical_config_sha256'] = 'a'*64
            with self.subTest(field=field), self.assertRaises(PlanError): self.verify(plan)

    def test_reject_wrong_parent_revision_split_or_manifest(self):
        for field in ('revision', 'count', 'manifest', 'completed', 'missingbudget'):
            parent = copy.deepcopy(self.parent)
            if field == 'revision': parent['model_revision'] = 'other'
            elif field == 'count': parent['budget']['split_counts']['validation'] = 64
            elif field == 'manifest': parent['input_sha256']['manifest.json'] = 'a'*64
            elif field == 'completed': parent['completed_updates'] = 1
            else: parent['budget'].pop('ppo_micro_batch_size')
            with self.subTest(field=field), self.assertRaises(PlanError):
                verify_plan(self.discovery, parent, expected_plan_sha256=document_sha256(self.discovery))

    def test_reject_fixture_roles_child_parent_collision_and_schema_extras(self):
        for change in ('fixture', 'sameparent', 'schema', 'extra'):
            plan = copy.deepcopy(self.discovery)
            if change == 'fixture': plan['role'] = 'synthetic_native_fixture'
            elif change == 'sameparent': plan['run_id'] = 'production'
            elif change == 'schema': plan['schema_version'] = True
            else: plan['accepted'] = True
            with self.subTest(change=change), self.assertRaises(PlanError): self.verify(plan)

    def test_approved_budget_and_fresh_initialization_cannot_change(self):
        for field, value in (('outer_count', 5), ('fixed_validation_count', 64),
                             ('optimizer_writes_per_rank', 19), ('monitors_per_rank', 39)):
            plan = copy.deepcopy(self.discovery); plan['phase_budget'][field] = value
            with self.subTest(field=field), self.assertRaises(PlanError): self.verify(plan)
        for field, value in (('adam_state', 'inherited'), ('logical_update', 1),
                             ('checkpoint', '/discovery/step1'), ('inherited_policy_run_id', 'discovery')):
            plan = copy.deepcopy(self.discovery); plan['initialization'][field] = value
            with self.subTest(field=field), self.assertRaises(PlanError): self.verify(plan)
        plan = copy.deepcopy(self.discovery); plan['mathematics']['ppo_mini_batch_size'] = 256
        with self.assertRaises(PlanError): self.verify(plan)

    def test_original_probability_gate_and_candidates_cannot_be_relaxed(self):
        for field, value in (('dual_clip', 4), ('epsilon', .3), ('target_old_policy_kl', .1),
                             ('kl_log_bound', 800), ('negative_tail_limit', 1), ('dual_clip', True)):
            plan = copy.deepcopy(self.discovery); plan['candidate_thresholds'][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(PlanError): self.verify(plan)
        for field, value in (('tolerance', .1), ('required_actual_optimizer_writes', 1)):
            plan = copy.deepcopy(self.discovery); plan['initial_probability_gate'][field] = value
            with self.subTest(field=field), self.assertRaises(PlanError): self.verify(plan)

    def test_actual_artifact_hash_slots_are_required_but_do_not_certify_gpu(self):
        for field in ('gate', 'config', 'encoder', 'head', 'reward_math', 'round'):
            plan = copy.deepcopy(self.discovery)
            if field == 'gate': plan['initial_probability_gate']['proof_sha256'] = 'not a hash'
            elif field == 'config': plan['resolved_config_sha256'] = None
            elif field == 'encoder': plan['initial_reward_fingerprint']['encoder_sha256']['config.json'] = 'a'*64
            elif field == 'head': plan['initial_reward_fingerprint']['head_sha256'].pop('commit_probe.pkl')
            elif field == 'round': plan['initial_reward_fingerprint']['probe_round'] = True
            else: plan['initial_reward_fingerprint']['reward_math_sha256'] = 'bad'
            with self.subTest(field=field), self.assertRaises(PlanError): self.verify(plan)
        result = self.verify(self.discovery)
        self.assertFalse(result['actual_recipe_verified'])
        self.assertFalse(result['actual_probe_identity_verified'])

    def test_actual_six_file_encoder_fingerprint_rejects_missing_wrong_or_extra(self):
        self.assertEqual(len(self.discovery['initial_reward_fingerprint']['encoder_sha256']), 6)
        for change in ('missing', 'wrong', 'extra'):
            plan = copy.deepcopy(self.discovery)
            encoder = plan['initial_reward_fingerprint']['encoder_sha256']
            if change == 'missing': encoder.pop('config.json')
            elif change == 'wrong': encoder['config.json'] = 'a'*64
            else: encoder['vocab.json'] = self.parent['initial_model_sha256']['vocab.json']
            with self.subTest(change=change), self.assertRaises(PlanError): self.verify(plan)

    def test_confirmation_tail_and_provenance_required(self):
        original, raw = self.confirmation()
        for tail in (None, -1, 1.01, float('nan'), float('inf'), True):
            plan = copy.deepcopy(original); plan['candidate_thresholds']['negative_tail_limit'] = tail
            with self.subTest(tail=tail), self.assertRaises(PlanError):
                self.verify(plan, discovery_plan=self.discovery, discovery_evidence_bytes=raw)
        for name in ('threshold_selection', 'evidence_sha256', 'plan_sha256'):
            plan = copy.deepcopy(original); plan['discovery'].pop(name)
            with self.subTest(name=name), self.assertRaises(PlanError):
                self.verify(plan, discovery_plan=self.discovery, discovery_evidence_bytes=raw)

    def test_confirmation_rejects_changed_bytes_wrong_role_run_budget_or_identity(self):
        plan, raw = self.confirmation()
        with self.assertRaises(PlanError):
            self.verify(plan, discovery_plan=self.discovery, discovery_evidence_bytes=raw + b' ')
        for field, value in (('role', 'policy_safety_calibration_structural_verification'),
                             ('origin', 'CPU_fixture'), ('run_id', 'other'),
                             ('completed_outer_updates', 0), ('optimizer_writes_per_rank', 19),
                             ('ranks', list(range(7))), ('ranks', [False, True, 2, 3, 4, 5, 6, 7]),
                             ('ranks', [float(r) for r in range(8)])):
            bad = json.loads(raw); bad[field] = value
            changed = canonical_bytes(bad); changed_plan = copy.deepcopy(plan)
            changed_plan['discovery']['evidence_sha256'] = hashlib.sha256(changed).hexdigest()
            with self.subTest(field=field), self.assertRaises(PlanError):
                self.verify(changed_plan, discovery_plan=self.discovery, discovery_evidence_bytes=changed)

    def test_confirmation_cannot_inherit_discovery_gate_or_change_health_rule(self):
        plan, raw = self.confirmation()
        for field in ('run', 'gate', 'rule', 'selection', 'reward'):
            bad = copy.deepcopy(plan)
            if field == 'run': bad['run_id'] = 'discovery'
            elif field == 'gate': bad['slurm_job_id'] = '3265'
            elif field == 'rule': bad['health_rule']['document']['criteria']['definition'] = 'changed'
            elif field == 'reward': bad['initial_reward_fingerprint']['head_sha256']['commit_probe.pkl'] = 'a'*64
            else: bad['discovery']['threshold_selection']['document']['criteria']['candidate_thresholds']['negative_tail_limit'] = .3
            with self.subTest(field=field), self.assertRaises(PlanError):
                self.verify(bad, discovery_plan=self.discovery, discovery_evidence_bytes=raw)


if __name__ == '__main__':
    unittest.main()
