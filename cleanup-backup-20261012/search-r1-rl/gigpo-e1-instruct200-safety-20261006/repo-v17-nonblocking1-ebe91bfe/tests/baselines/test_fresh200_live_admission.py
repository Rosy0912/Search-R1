"""UNEXECUTED isolated CPU fixtures, never live approval or GPU evidence.

Only selected source functions/constants are AST-extracted. Project modules are
never imported, and no fixture dispatches Ray, CUDA, a job, or an external RPC.
Real CPU parameter/Adam/scheduler/RNG objects check policy-only preservation.
"""
import ast
import copy
import hashlib
import json
import math
import os
import random
import re
import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[2]
LIVE_FUNCTIONS = {
    'require', 'digest', 'core', 'read_bound_json', 'existing_record',
    'live_initialization', 'validate_live_initialization', 'validate_live_binding',
    '_initial_payload', 'validate_admission', 'validate_config_delta',
    'validate_production_review', 'read_bound_json_lines', 'driver_admission_state',
    'worker_admission_state', 'install_worker_admission',
    'restore_worker_admission_state', 'restore_driver_admission_state',
    'validate_live_safety_config', 'validate_live_update_boundary',
}


class WithoutImports(ast.NodeTransformer):
    """Replace dependency imports with the explicit isolated fixture namespace."""
    def visit_Import(self, node):
        return None

    def visit_ImportFrom(self, node):
        return None


def extract(path, names, namespace, constants=False, class_name=None):
    tree = ast.parse(path.read_text(), filename=str(path))
    nodes = tree.body
    if class_name:
        nodes = next(node.body for node in nodes
                     if isinstance(node, ast.ClassDef) and node.name == class_name)
    selected, found = [], set()
    for node in nodes:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node = copy.deepcopy(node)
            node.decorator_list = []
            selected.append(WithoutImports().visit(node))
            found.add(node.name)
        elif constants and isinstance(node, ast.Assign):
            selected.append(copy.deepcopy(node))
    if found != names:
        raise AssertionError('Missing extracted source functions: ' + repr(names - found))
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(module, str(path), 'exec'), namespace)


def source_namespace():
    namespace = dict(copy=copy, hashlib=hashlib, json=json, math=math, os=os, dist=torch.distributed,
                     stat=stat, Path=Path, OmegaConf=OmegaConf, torch=torch, re=re)
    extract(ROOT / 'search_r1/baselines/fresh200.py',
            {'calibration_initialization', 'validate_attempt_ledger',
             'validate_worker_attempt_ledger', 'record_worker_attempt'}, namespace, constants=True)
    extract(ROOT / 'search_r1/baselines/fresh200_live.py', LIVE_FUNCTIONS,
            namespace, constants=True)
    extract(ROOT / 'search_r1/baselines/policy_safety.py',
            {'rejection_reason', 'calibrated_or_candidate_rejection', 'local_statistics', 'global_statistics'}, namespace)
    extract(ROOT / 'search_r1/baselines/partial_update.py',
            {'_adam_steps', 'rejection_disposition'}, namespace)
    # These placement helpers are CPU boundary substitutes, not FSDP evidence.
    namespace.update(cpu_snapshot=copy.deepcopy,
        rng_state=lambda: dict(python=random.getstate(), torch_cpu=torch.get_rng_state().clone()),
        offload_fsdp_param_and_grad=lambda *args, **kwargs: None,
        offload_fsdp_optimizer=lambda *args, **kwargs: None)
    extract(ROOT / 'verl/workers/fsdp_workers.py',
            {'_validate_checkpoint_runtime_placement', '_restore_checkpoint_runtime_metadata'}, namespace)
    extract(ROOT / 'verl/workers/fsdp_workers.py', {'checkpoint_runtime_metadata'},
            namespace, class_name='ActorRolloutRefWorker')
    return namespace


def fixture_ledger(schema):
    history, total = [], 0
    # One abandoned partial retains writes; five separate countable outers follow.
    attempts = [(7, 3, True)] + [(20, writes, False) for writes in (1, 19, 7, 20, 2)]
    completed = 0
    for index, (processed, writes, abandoned) in enumerate(attempts, 1):
        counted = processed == 20 and writes > 0 and not abandoned
        history.append(dict(attempt_id=index, completed_before=completed,
            actual_adam_total_before=total, opportunities_processed=processed,
            actual_successful_adam_writes=writes, skipped_opportunities=processed - writes,
            gradient_micro_batches=writes * 8, actual_adam_total=total + writes,
            scheduler_step=completed + int(counted), abandoned=abandoned,
            counted_complete=counted, safe_to_continue=True))
        total += writes
        completed += int(counted)
    return dict(schema=schema, completed_outer_updates=completed, attempts=len(history),
        skipped_attempts=1, actual_adam_total=total,
        successful_adam_writes_completed_outers=total - 3,
        successful_adam_writes_abandoned_attempts=3,
        skipped_opportunities=sum(row['skipped_opportunities'] for row in history),
        gradient_micro_batches=total * 8, consumed_training_questions=len(history) * 512,
        history=history, active_attempt=None)


class SyntheticGroup:
    """Deterministic RPC return seam; these rows are explicitly synthetic."""
    def __init__(self, rows):
        self.rows = copy.deepcopy(rows)
        self.calls = []
        self.restore_corruption = None

    def calibration_live_admission_identity(self):
        self.calls.append('read')
        return copy.deepcopy(self.rows)

    def calibration_restore_live_admission(self, saved):
        self.calls.append('restore')
        self.rows = copy.deepcopy(saved)
        if self.restore_corruption:
            self.restore_corruption(self.rows)
        return copy.deepcopy(self.rows)


class LiveAdmission(unittest.TestCase):
    def setUp(self):
        self.ns = source_namespace()
        self.temp = tempfile.TemporaryDirectory(prefix='synthetic-live-admission-')
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name).resolve()
        self.env = patch.dict(os.environ, dict(SLURM_JOB_ID='SYNTHETIC_CPU_JOB',
            SEARCH_R1_BASELINE_RUN_ID='SYNTHETIC_CPU_RUN', SEARCH_R1_BASELINE_COMMIT='a' * 40,
            SEARCH_R1_BASELINE_OUTPUT=str(self.run)))
        self.env.start()
        self.addCleanup(self.env.stop)
        self.binding = self.valid_binding()
        binding = self.write('synthetic-binding.json', self.binding)
        os.environ.update(GIGPO_POLICY_NATIVE_BINDING=binding['path'],
                          GIGPO_POLICY_NATIVE_BINDING_SHA256=binding['sha256'])
        self.c0_config = self.config()
        config = self.write('resolved-config-C0.json', self.c0_config)
        os.environ['SEARCH_R1_BASELINE_CONFIG_SHA256'] = config['sha256']
        self.c0 = self.ns['_initial_payload']()
        self.c1, self.c1_config = self.production_payload()

    def write(self, name, value, jsonl=False):
        path = self.run / name
        raw = (b'\n'.join(json.dumps(row, sort_keys=True, allow_nan=False).encode() for row in value) + b'\n'
               if jsonl else json.dumps(value, sort_keys=True, allow_nan=False).encode())
        path.write_bytes(raw)
        return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())

    def valid_binding(self):
        return dict(self.ns['live_initialization'](),
            status='REVIEWED_FOR_FRESH200_LIVE_TRAINING_ONLY', validation_scope=self.ns['LIVE_SCOPE'],
            calibration_scope=self.ns['CALIBRATION_SCOPE'], live_goal_updated_at=self.ns['LIVE_GOAL_UPDATED_AT'],
            live_authorization_goal_updated_at=self.ns['LIVE_AUTHORIZATION_GOAL_UPDATED_AT'],
            official_training_outer_target=200, initial_calibration_counted_outer_limit=5,
            first_full_acceptance_step=5, first5_count_toward_goal=True,
            first5_known_token_kl_enforced=True, first5_tail_observation_only=True,
            restart_after_calibration=False, production_tail_required_before_outer6=True,
            retained_training_updates_contributed=0, candidate_old_policy_kl=.2, target_old_policy_kl=.2,
            outer_counting_authorization_source='human', outer_counting_human_goal_updated_at=1791283983,
            allowed_safety_modes=['calibration', 'production'], code_commit='a' * 40,
            source_commit='a' * 40, source_native_acceptance=None, negative_tail_limit=None,
            actual_gpu_acceptance=None, actual_probability_acceptance=None)

    def config(self):
        return dict(actor_rollout_ref=dict(actor=dict(fresh200_live_protocol=True,
            fresh200_protocol=True, ppo_mini_batch_size=128, ppo_micro_batch_size=16,
            optim=dict(lr=1e-6, lr_warmup_steps=57, total_training_steps=200),
            policy_safety=dict(enabled=True, live_training_protocol=True, mode='calibration',
                first5_known_token_kl_enforced=True, first5_tail_observation_only=True,
                calibration_scope=self.ns['CALIBRATION_SCOPE'], candidate_old_policy_kl=.2,
                target_old_policy_kl=.2, negative_tail_limit=None, dual_clip=3., kl_log_bound=30.,
                monitor_first_global_rows=256, calibration_evidence_path=None,
                calibration_evidence_sha256=None)), model=dict(path='/SYNTHETIC/official'),
            rollout=dict(n_agent=5, log_prob_micro_batch_size=32),
            ref=dict(log_prob_micro_batch_size=32)),
            trainer=dict(total_training_steps=200, default_hdfs_dir=None,
                default_local_dir=str(self.run / 'original-checkpoints')),
            algorithm=dict(kl_ctrl=dict(kl_coef=0)), baselines=dict(method='gigpo'))

    def production_payload(self):
        ledger = fixture_ledger(self.ns['ATTEMPT_SCHEMA'])
        self.ns['validate_attempt_ledger'](ledger)
        manifest_value = dict(job_id=self.c0['job_id'], code_commit=self.c0['source_commit'],
            logical_step=5, actual_new_updates=5, config_sha256=self.c0['config_record']['sha256'])
        manifest = self.write('synthetic-first5-manifest.json', manifest_value)
        probability = self.write('synthetic-probability-approval.json', dict(status='PASS',
            scope='GIGPO_OFFICIAL_INSTRUCT_FRESH200_RAW_ACTION_PROBABILITY_IDENTITY_V1',
            actual_adam_writes=0, job_id=self.c0['job_id'], run_id=self.c0['run_id'],
            code_commit=self.c0['source_commit'],
            actual_hf_role_micro_batch_sizes={'old': 4, 'current': 2, 'reference': 4},
            actual_dropout_neutral_train_eval_verified=True))
        guards = []
        for rank in range(8):
            rows = [dict(rank=rank, execution_mode='calibration', commit=self.c0['source_commit'],
                run_id=self.c0['run_id'], config_sha256=self.c0['config_record']['sha256'],
                native_binding_sha256=self.c0['initial_binding_record']['sha256'], attempt_id=index,
                negative_tail_fraction_count=.01 * (rank + index)) for index in range(1, 7)]
            guards.append(dict(self.write(f'synthetic-rank-{rank}.jsonl', rows, jsonl=True), rank=rank))
        worker_config = copy.deepcopy(self.c0_config['actor_rollout_ref'])
        worker_config['actor'].update(ppo_mini_batch_size=16, ppo_micro_batch_size=2)
        roles = {role: [dict(rank=rank, pid=10000 + rank, start_ticks=str(20000 + rank), role=role,
            state=dict(payload=copy.deepcopy(self.c0), worker_config=copy.deepcopy(worker_config),
                audit_environment={key: os.environ[key] for key in self.ns['AUDIT_ENV_KEYS']}))
            for rank in range(8)] for role in ('actor', 'reference')}
        role_digest = self.ns['digest'](roles)
        tail = self.write('synthetic-tail-evidence.json', dict(completed_outer_updates=5,
            actual_attempt_ledger_digest=self.ns['digest'](ledger), job_id=self.c0['job_id'],
            run_id=self.c0['run_id'], source_commit=self.c0['source_commit'],
            config_sha256=self.c0['config_record']['sha256'], rank_guard_records=guards,
            role_admissions=roles, role_admissions_digest=role_digest))
        pauses = [dict(rank=rank, pid=10000 + rank, start_ticks=str(20000 + rank), scheduler_step=5)
                  for rank in range(8)]
        ready = self.write('synthetic-first5-ready.json', dict(
            manifest=dict(sha256=manifest['sha256'], identity=manifest_value),
            checkpoint_pause_identity=pauses,
            actual_state_roundtrip=[dict(rank=rank) for rank in range(8)],
            actual_driver_state_roundtrip=dict(actual_live_admission_reload=True),
            actual_worker_runtime_roundtrip=dict(complete_live_before_after_equal=True)))
        review = self.write('synthetic-independent-policy-review.json', dict(status='PASS',
            scope=self.ns['POLICY_SCOPE'], independent_review=True, implementation_authorship=False,
            root_conversation_inherited=False, job_id=self.c0['job_id'], run_id=self.c0['run_id'],
            source_commit=self.c0['source_commit'], live_goal_updated_at=self.ns['LIVE_GOAL_UPDATED_AT'],
            live_authorization_goal_updated_at=self.ns['LIVE_AUTHORIZATION_GOAL_UPDATED_AT'],
            first5_count_toward_goal=True, logical_step=5, actual_new_updates=5,
            manifest_sha256=manifest['sha256'], config_sha256=self.c0['config_record']['sha256'],
            initial_binding_sha256=self.c0['initial_binding_record']['sha256'],
            probability_approval_sha256=probability['sha256'], fresh200_attempt_ledger=ledger,
            actual_adam_total=ledger['actual_adam_total'], scheduler_step=5, rank_count=8,
            actual_tail_evidence_record=tail, first5_state_ready_record=ready,
            role_admissions_digest=role_digest,
            checkpoint_pause_identity=pauses, production_policy=dict(target_old_policy_kl=.2,
                negative_tail_limit=.17, dual_clip=3., kl_log_bound=30.)))
        config = copy.deepcopy(self.c0_config)
        policy = config['actor_rollout_ref']['actor']['policy_safety']
        policy.update(mode='production', negative_tail_limit=.17,
                      calibration_evidence_path=review['path'], calibration_evidence_sha256=review['sha256'])
        record = self.write('resolved-config-C1.json', config)
        payload = dict(copy.deepcopy(self.c0), phase='C1_PRODUCTION', config_epoch=1,
            config_record=record, policy=copy.deepcopy(policy), policy_review_record=review,
            first5_manifest_record=manifest, probability_approval_record=probability,
            parent_admission_sha256=self.ns['digest'](self.c0))
        return payload, config

    def worker(self, role='actor', rank=0):
        config = OmegaConf.create(copy.deepcopy(self.c0_config['actor_rollout_ref']))
        config.actor.ppo_mini_batch_size = 16
        config.actor.ppo_micro_batch_size = 2
        module = torch.nn.Linear(2, 1, bias=False)
        optimizer = torch.optim.AdamW(module.parameters(), lr=1e-6)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: step / 57.)
        ledger = fixture_ledger(self.ns['ATTEMPT_SCHEMA'])
        for attempt in ledger['history']:
            for _ in range(attempt['actual_successful_adam_writes']):
                optimizer.zero_grad(set_to_none=True)
                module.weight.grad = torch.ones_like(module.weight)
                optimizer.step()
            if attempt['counted_complete']:
                scheduler.step()
        return SimpleNamespace(config=config, rank=rank, _is_actor=role == 'actor',
            actor=SimpleNamespace(config=config.actor), actor_module_fsdp=module,
            actor_optimizer=optimizer, actor_lr_scheduler=scheduler,
            _is_offload_param=True, _is_offload_optimizer=True, _is_offload_grad=False,
            rollout_sharding_manager=SimpleNamespace(gen_random_states=None, torch_random_states=None),
            fresh200_live_admission=copy.deepcopy(self.c0),
            fresh200_worker_ledger=dict(schema='FRESH200_WORKER_ACTUAL_ATTEMPTS_V1',
                attempts=ledger['attempts'], completed_outer_updates=5,
                actual_adam_total=ledger['actual_adam_total'], history=[]))

    def math_state(self, worker):
        return dict(module_id=id(worker.actor_module_fsdp), optimizer_id=id(worker.actor_optimizer),
            scheduler_id=id(worker.actor_lr_scheduler), parameters=copy.deepcopy(worker.actor_module_fsdp.state_dict()),
            adam=copy.deepcopy(worker.actor_optimizer.state_dict()),
            scheduler=copy.deepcopy(worker.actor_lr_scheduler.state_dict()),
            python_rng=random.getstate(), torch_rng=torch.get_rng_state().clone(),
            ledger=copy.deepcopy(worker.fresh200_worker_ledger))

    def assert_state_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_state_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for before, after in zip(left, right):
                self.assert_state_equal(before, after)
        else:
            self.assertEqual(left, right)

    def trainer(self):
        state = self.ns['worker_admission_state'](self.worker())
        groups = []
        for role in ('actor', 'reference'):
            groups.append(SyntheticGroup([dict(rank=rank, role=role, pid=10000 + rank,
                start_ticks=str(20000 + rank),
                state=copy.deepcopy(state)) for rank in range(8)]))
        return SimpleNamespace(config=OmegaConf.create(self.c0_config),
            actor_rollout_wg=groups[0], ref_policy_wg=groups[1],
            fresh200_live_admission=copy.deepcopy(self.c0))

    def test_old_isolated_two_binding_is_not_live_authority(self):
        self.ns['validate_live_binding'](self.binding, 'a' * 40)
        old = dict(self.binding, status='REVIEWED_FOR_CALIBRATION_ONLY',
            validation_scope=self.ns['CALIBRATION_SCOPE'], allowed_safety_modes=['calibration'],
            calibration_actual_outer_budget=2, diagnostic_total_optimizer_opportunities=40)
        with self.assertRaises(ValueError):
            self.ns['validate_live_binding'](old, 'a' * 40)

    def test_live_binding_requires_exact_same_job_first5_to200_intent(self):
        changes = dict(first5_count_toward_goal=False, restart_after_calibration=True,
            production_tail_required_before_outer6=False, initial_calibration_counted_outer_limit=2,
            official_training_outer_target=300, first_full_acceptance_step=True,
            live_goal_updated_at=1791283983, target_old_policy_kl=.01,
            source_commit='b' * 40, actual_gpu_acceptance={'status': 'PASS'})
        for key, value in changes.items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.ns['validate_live_binding'](dict(self.binding, **{key: value}), 'a' * 40)
        with self.assertRaises(ValueError):
            self.ns['validate_live_binding'](dict(self.binding, code_commit='g' * 40,
                source_commit='g' * 40), 'g' * 40)

    def test_first5_known_kl_and_observational_tail_flags_are_required_in_binding_and_policy(self):
        for flag in ('first5_known_token_kl_enforced', 'first5_tail_observation_only'):
            for missing in (False, True):
                binding = copy.deepcopy(self.binding)
                policy = copy.deepcopy(self.c0['policy'])
                if missing:
                    binding.pop(flag)
                    policy.pop(flag)
                else:
                    binding[flag] = False
                    policy[flag] = False
                with self.subTest(flag=flag, missing=missing):
                    with self.assertRaises(ValueError):
                        self.ns['validate_live_binding'](binding, 'a' * 40)
                    with self.assertRaises(ValueError):
                        self.ns['validate_live_safety_config'](policy)

    def test_actual_wrapper_enforces_known_point_two_from_first_outer_but_observes_null_tail(self):
        policy = self.c0['policy']
        stats = dict(rejected_numeric_count=0, weight_mass=1., sampled_old_policy_kl=.2,
                     negative_tail_fraction_count=1.)
        self.assertIsNone(self.ns['calibrated_or_candidate_rejection'](stats, policy))
        rejected = dict(stats, sampled_old_policy_kl=.201)
        self.assertEqual(self.ns['calibrated_or_candidate_rejection'](rejected, policy), 'OLD_POLICY_TOKEN_KL')
        numeric = dict(stats, rejected_numeric_count=1, sampled_old_policy_kl=None)
        self.assertEqual(self.ns['calibrated_or_candidate_rejection'](numeric,
            dict(policy, first5_known_token_kl_enforced=False)),
            'NONFINITE_OR_EXP_RANGE_OR_EMPTY_EFFECTIVE_MONITOR')
        legacy = dict(policy, live_training_protocol=False)
        legacy.pop('first5_known_token_kl_enforced')
        legacy.pop('first5_tail_observation_only')
        self.assertEqual(self.ns['calibrated_or_candidate_rejection'](rejected, legacy),
                         'CALIBRATION_CANDIDATE_OLD_POLICY_TOKEN_KL')
        self.assertIsNone(self.ns['calibrated_or_candidate_rejection'](stats, legacy))

    def test_actual_worker_post_kl_path_retains_three_adam_writes_and_abandons_rollout(self):
        policy = self.c0['policy']
        reason = self.ns['calibrated_or_candidate_rejection'](dict(rejected_numeric_count=0,
            weight_mass=1., sampled_old_policy_kl=.201, negative_tail_fraction_count=1.), policy)
        self.assertEqual(reason, 'OLD_POLICY_TOKEN_KL')
        self.assertEqual(self.ns['rejection_disposition'](dict(rejection_reason=reason,
            phase='after', mini_index=2)), 'fail_closed')
        self.assertEqual(self.ns['rejection_disposition'](dict(rejection_reason=reason,
            phase='after_unknown', mini_index=2)), 'fail_closed')
        # Reuse only the legacy fixture's AST helper/fixture definitions. Its
        # project imports and test bodies are never imported or evaluated.
        class KnownRefusal(Exception):
            def __init__(self, reason, record):
                super().__init__(reason)
                self.record = record
        namespace = dict(ast=ast, copy=copy, hashlib=hashlib, json=json, math=math, os=os, Path=Path,
            SimpleNamespace=SimpleNamespace, torch=torch, ROOT=ROOT, KnownRefusal=KnownRefusal,
            _adam_steps=self.ns['_adam_steps'], rejection_disposition=self.ns['rejection_disposition'],
            local_statistics=self.ns['local_statistics'], global_statistics=self.ns['global_statistics'],
            CALIBRATION_SCOPE=self.ns['CALIBRATION_SCOPE'],
            calibrated_or_candidate_rejection=self.ns['calibrated_or_candidate_rejection'])
        extract(ROOT / 'tests/baselines/test_fresh200_worker_skip_static.py',
                {'source_function', 'source_monitor_tail', 'actual_worker_rejection', 'worker_fixture'}, namespace)
        worker, data, method, calls, stages, saved = namespace['worker_fixture'](self.run / 'synthetic-partial', {2: 'kl'})
        metrics, receipt = method(worker, data)
        self.assertEqual(calls, [0, 1, 2])
        self.assertEqual(receipt['actual_successful_adam_writes'], 3)
        self.assertEqual(receipt['actual_adam_total'], 3)
        self.assertEqual(receipt['opportunities_processed'], 3)
        self.assertTrue(receipt['abandoned'])
        self.assertFalse(receipt['counted_complete'])
        self.assertEqual(receipt['refusals'][0]['partial_update']['rejection']['rejection_reason'], reason)
        actual_rejection = receipt['refusals'][0]['partial_update']['rejection']
        self.assertEqual(actual_rejection['refusal_kind'], 'known_policy_limit')
        self.assertEqual(actual_rejection['matched_guard'], 'ORIGINAL_GLOBAL_STATISTICS_POLICY_REFUSAL')
        self.assertEqual(self.ns['rejection_disposition'](actual_rejection), 'abandon')
        self.assertEqual(self.ns['_adam_steps']({'optimizer': worker.actor_optimizer.state_dict()}), 3)
        self.assertEqual(metrics['actor/optimizer_steps'], 3)
        self.assertTrue(saved)
        self.assertTrue(all(parameter.grad is None for parameter in worker.actor_module_fsdp.parameters()))
        self.ns['record_worker_attempt'](worker, receipt)
        self.assertEqual(worker.actor_lr_scheduler.last_epoch, 0)
        self.assertEqual(worker.actor_lr_scheduler.writes, 0)
        self.assertEqual(worker.fresh200_worker_ledger['actual_adam_total'], 3)
        self.assertEqual(worker.fresh200_worker_ledger['completed_outer_updates'], 0)

    def test_c1_fullconfig_delta_allows_only_approved_policy_fields(self):
        self.ns['validate_admission'](self.c0)
        self.ns['validate_admission'](self.c1)
        for path, wrong in [('actor_rollout_ref.actor.ppo_micro_batch_size', 8),
                ('actor_rollout_ref.actor.optim.lr', 2e-6), ('trainer.total_training_steps', 300),
                ('actor_rollout_ref.actor.policy_safety.target_old_policy_kl', .01)]:
            changed = OmegaConf.create(self.c1_config)
            OmegaConf.update(changed, path, wrong)
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.ns['validate_config_delta'](self.c0_config, OmegaConf.to_container(changed, resolve=True))
        for wrong in (None, True, float('nan'), -0.1, 1.1):
            changed = copy.deepcopy(self.c1_config)
            changed['actor_rollout_ref']['actor']['policy_safety']['negative_tail_limit'] = wrong
            with self.subTest(tail=wrong), self.assertRaises(ValueError):
                self.ns['validate_config_delta'](self.c0_config, changed)

    def test_real_cpu_install_preserves_model_adam_rng_and_normalized_micro(self):
        for role in ('actor', 'reference'):
            os.environ['SEARCH_R1_BASELINE_CONFIG_SHA256'] = self.c0['config_record']['sha256']
            worker = self.worker(role)
            before = self.math_state(worker)
            observed_c0 = self.ns['worker_admission_state'](worker)
            installed = self.ns['install_worker_admission'](worker, self.c1)
            self.assertEqual(installed['payload'], self.c1)
            self.assertEqual(worker.config.actor.ppo_mini_batch_size, 16)
            self.assertEqual(worker.config.actor.ppo_micro_batch_size, 2)
            self.assertEqual(observed_c0['payload'], self.c0)
            self.assert_state_equal(before, self.math_state(worker))

    def test_review_requires_bound_original_all8_evidence_and_actual_readback(self):
        review = self.ns['read_bound_json'](self.c1['policy_review_record'])
        self.ns['validate_production_review'](review, self.c1)
        wrong_digest = dict(review, role_admissions_digest='b' * 64)
        with self.assertRaises(ValueError):
            self.ns['validate_production_review'](wrong_digest, self.c1)
        evidence = self.ns['read_bound_json'](review['actual_tail_evidence_record'])
        evidence['rank_guard_records'][7]['rank'] = 6
        changed = dict(review, actual_tail_evidence_record=self.write('synthetic-wrong-all8.json', evidence))
        with self.assertRaises(ValueError):
            self.ns['validate_production_review'](changed, self.c1)
        ready = self.ns['read_bound_json'](review['first5_state_ready_record'])
        ready['actual_driver_state_roundtrip']['actual_live_admission_reload'] = False
        changed = dict(review, first5_state_ready_record=self.write('synthetic-wrong-readback.json', ready))
        with self.assertRaises(ValueError):
            self.ns['validate_production_review'](changed, self.c1)

    def test_realized_admission_rejects_wrong_launcher_path_sha_or_parent_chain(self):
        for phase in (self.c0, self.c1):
            wrong_path = copy.deepcopy(phase)
            wrong_path['config_record'] = self.write('synthetic-unadmitted-full-config.json',
                self.c0_config if phase['config_epoch'] == 0 else self.c1_config)
            with self.assertRaises(ValueError):
                self.ns['validate_admission'](wrong_path)
            wrong_binding = copy.deepcopy(phase)
            wrong_binding['initial_binding_record']['sha256'] = 'b' * 64
            with self.assertRaises(ValueError):
                self.ns['validate_admission'](wrong_binding)
        wrong_parent = dict(self.c1, parent_admission_sha256='b' * 64)
        with self.assertRaises(ValueError):
            self.ns['validate_admission'](wrong_parent)
        invented_initial = dict(self.c0, first5_manifest_record=self.c1['first5_manifest_record'])
        with self.assertRaises(ValueError):
            self.ns['validate_admission'](invented_initial)

    def test_production_review_preserves_before_adam_micro2_and_dropout_authority(self):
        review = self.ns['read_bound_json'](self.c1['policy_review_record'])
        probability = self.ns['read_bound_json'](self.c1['probability_approval_record'])
        for key, value in [('actual_hf_role_micro_batch_sizes', {'old': 4, 'current': 4, 'reference': 4}),
                ('actual_dropout_neutral_train_eval_verified', False), ('actual_adam_writes', 1)]:
            changed = dict(probability, **{key: value})
            record = self.write('synthetic-unadmitted-probability.json', changed)
            payload = dict(self.c1, probability_approval_record=record)
            changed_review = dict(review, probability_approval_sha256=record['sha256'])
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.ns['validate_production_review'](changed_review, payload)

    def test_real_worker_saved_admission_restore_and_metadata_readback(self):
        worker = self.worker()
        self.ns['install_worker_admission'](worker, self.c1)
        metadata = self.ns['checkpoint_runtime_metadata'](worker)
        before = self.math_state(worker)
        OmegaConf.update(worker.config, 'actor.policy_safety', copy.deepcopy(self.c0['policy']), merge=False)
        worker.fresh200_live_admission = copy.deepcopy(self.c0)
        os.environ['SEARCH_R1_BASELINE_CONFIG_SHA256'] = self.c0['config_record']['sha256']
        with patch.object(torch.cuda, 'current_device', return_value=0):
            self.ns['_restore_checkpoint_runtime_metadata'](worker, metadata)
        self.assert_state_equal(metadata, self.ns['checkpoint_runtime_metadata'](worker))
        self.assert_state_equal(before, self.math_state(worker))

    def test_worker_restore_rejects_wrong_math_or_audit_metadata_before_mutation(self):
        worker = self.worker()
        state = self.ns['worker_admission_state'](worker)
        changes = [lambda row: row['worker_config']['actor'].update(ppo_micro_batch_size=16),
                   lambda row: row['audit_environment'].update(SEARCH_R1_BASELINE_CONFIG_SHA256='b' * 64),
                   lambda row: row['audit_environment'].update(GIGPO_POLICY_NATIVE_BINDING='/wrong/path'),
                   lambda row: row['audit_environment'].update(SYNTHETIC_UNAUTHORIZED_ENV='injected')]
        for change in changes:
            bad = copy.deepcopy(state)
            change(bad)
            before = self.math_state(worker)
            with self.assertRaises(ValueError):
                self.ns['restore_worker_admission_state'](worker, bad)
            self.assert_state_equal(before, self.math_state(worker))
            self.assertEqual(self.ns['worker_admission_state'](worker), state)

    def test_saved_live_protocol_mismatch_rejects_before_placement(self):
        worker = self.worker()
        with self.assertRaises(ValueError):
            self.ns['_restore_checkpoint_runtime_metadata'](worker, {})
        worker.config.actor.fresh200_live_protocol = False
        with self.assertRaises(ValueError):
            self.ns['_restore_checkpoint_runtime_metadata'](worker, {'fresh200_live_admission': {}})

    def test_driver_all8_role_payload_and_coverage_refusal(self):
        for role, change in [('actor', lambda rows: rows[7]['state']['payload'].update(config_epoch=1)),
                ('reference', lambda rows: rows.pop()),
                ('actor', lambda rows: rows[7].update(rank=6)),
                ('reference', lambda rows: rows[7].update(role='actor')),
                ('actor', lambda rows: rows[7].update(pid=rows[6]['pid'])),
                ('reference', lambda rows: rows[7].update(pid=0)),
                ('actor', lambda rows: rows[7].update(start_ticks='SYNTHETIC_INVALID')),
                ('reference', lambda rows: rows[1].update(rank=True))]:
            trainer = self.trainer()
            group = trainer.actor_rollout_wg if role == 'actor' else trainer.ref_policy_wg
            change(group.rows)
            with self.subTest(role=role), self.assertRaises(ValueError):
                self.ns['driver_admission_state'](trainer)

    def test_driver_actual_restore_calls_both_roles_and_requires_exact_readback(self):
        trainer = self.trainer()
        saved = self.ns['driver_admission_state'](trainer)
        self.ns['restore_driver_admission_state'](trainer, saved)
        self.assertEqual(self.ns['driver_admission_state'](trainer), saved)
        self.assertEqual(trainer.actor_rollout_wg.calls.count('restore'), 1)
        self.assertEqual(trainer.ref_policy_wg.calls.count('restore'), 1)
        trainer.ref_policy_wg.restore_corruption = lambda rows: rows[7]['state']['audit_environment'].update(
            SEARCH_R1_BASELINE_CONFIG_SHA256='b' * 64)
        with self.assertRaises(ValueError):
            self.ns['restore_driver_admission_state'](trainer, saved)

    def test_completed_boundary_rejects_sixth_under_c0_and_caps_c1_at200(self):
        worker = self.worker()
        def check(completed, payload, **metadata):
            worker.fresh200_worker_ledger['completed_outer_updates'] = completed
            data = SimpleNamespace(meta_info=dict(fresh200_completed_before=completed,
                fresh200_config_epoch=payload['config_epoch'],
                fresh200_config_sha256=payload['config_record']['sha256'], **metadata))
            self.ns['validate_live_update_boundary'](worker, data)
        check(4, self.c0)
        with self.assertRaises(ValueError):
            check(5, self.c0)
        self.ns['install_worker_admission'](worker, self.c1)
        check(5, self.c1)
        check(199, self.c1)
        for completed in (4, 200, True):
            with self.subTest(completed=completed), self.assertRaises(ValueError):
                check(completed, self.c1)

    def test_hash_bound_evidence_rejects_changed_bytes_and_duplicate_json(self):
        record = self.write('synthetic-bound.json', {'value': 1})
        self.assertEqual(self.ns['read_bound_json'](record), {'value': 1})
        Path(record['path']).write_bytes(b'{"value": 2}')
        with self.assertRaises(ValueError):
            self.ns['read_bound_json'](record)
        path = self.run / 'synthetic-duplicate.json'
        raw = b'{"value": 1, "value": 2}'
        path.write_bytes(raw)
        with self.assertRaises(ValueError):
            self.ns['read_bound_json'](dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))


if __name__ == '__main__':
    unittest.main()
