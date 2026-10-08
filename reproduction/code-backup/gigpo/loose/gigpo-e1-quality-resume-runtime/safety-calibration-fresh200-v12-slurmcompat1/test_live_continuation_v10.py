"""UNEXECUTED AST-extracted CPU controls, never actual all8/GPU approval.

No runtime/project module is imported. Selected production methods run only in
synthetic rank/RPC fixtures after separate execution authorization. The fake
matched stage records local calls; it never claims to perform a distributed MAX.
"""
import ast
import copy
import hashlib
import json
import math
import numbers
import os
import re
import tempfile
import unittest
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping
from unittest.mock import patch

import torch


RUNTIME = Path(__file__).resolve().parent


def source_fixture_root():
    candidates = {Path(entry).resolve() for entry in os.environ.get('PYTHONPATH', '').split(os.pathsep)
                  if entry and (Path(entry) / 'search_r1/baselines/fresh200_live.py').is_file()}
    if len(candidates) == 1:
        return candidates.pop()
    if candidates:
        raise AssertionError('Ambiguous actual source entries in fixture PYTHONPATH')
    local = RUNTIME.parent.parent / 'gigpo-e1-fresh200-compat11-20261006'
    if (local / 'search_r1/baselines/fresh200_live.py').is_file():
        return local.resolve()
    raise AssertionError('Missing actual source entry in fixture PYTHONPATH')


SOURCE = source_fixture_root()


class StripImports(ast.NodeTransformer):
    def visit_Import(self, node):
        return None

    def visit_ImportFrom(self, node):
        return None


def selected_functions(path, names, namespace, constants=False):
    tree = ast.parse(path.read_text(), filename=str(path))
    selected, found = [], set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            value = copy.deepcopy(node)
            value.decorator_list = []
            selected.append(StripImports().visit(value))
            found.add(node.name)
        elif constants and isinstance(node, ast.Assign):
            selected.append(copy.deepcopy(node))
    if found != names:
        raise AssertionError('Missing exact production fixture seams: ' + repr(names - found))
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(module, str(path), 'exec'), namespace)


def require(value, reason):
    if not value:
        raise ValueError(reason)


def state_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class SyntheticMatchedFailure(ValueError):
    pass


class SyntheticOriginalWorker:
    """No tensor math or remote calls: exact original-call/error identity seam."""
    def __init__(self, rank=0, role='actor'):
        self.rank = rank
        self._is_actor = role == 'actor'
        self._is_ref = role == 'reference'
        self.role = role
        self.pid = 10000 + rank
        self.start_ticks = str(20000 + rank)
        self.fresh200_worker_ledger = dict(completed_outer_updates=5, actual_adam_total=52)
        self.actor_lr_scheduler = SimpleNamespace(last_epoch=5)
        self.payload = dict(config_epoch=0, phase='C0_CALIBRATION', fixture_only=True)
        self.worker_config = dict(actor=dict(ppo_mini_batch_size=16, ppo_micro_batch_size=2))
        self.protected = dict(model_values=[1., 2.], adam_moments=[.5, .7], scheduler=5,
            python_rng='synthetic-python-state', torch_rng='synthetic-torch-state',
            sampler_cursor=3072, attempts=6, actual_adam_total=52, completed_outer_updates=5)
        self.original_calls = []
        self.original_error = None
        self.identity_error = None
        self.restore_calls = []
        self.install_calls = []
        self.actor = SimpleNamespace(config=SimpleNamespace(ppo_micro_batch_size=2))
        self.hf_calls = []
        self.capture_calls = []

    def update_actor(self, data):
        self.original_calls.append(data)
        if self.original_error:
            raise self.original_error
        return data

    def _admission_protected_state(self):
        return copy.deepcopy(self.protected)

    def compute_log_prob(self, data):
        self.hf_calls.append(('actor', data))
        if self.original_error:
            raise self.original_error
        return data

    def compute_ref_log_prob(self, data):
        self.hf_calls.append(('reference', data))
        if self.original_error:
            raise self.original_error
        return data

    def _complete_pause_state(self):
        return self._admission_protected_state()

    def calibration_reference_identity(self):
        return self._admission_protected_state()

    def _capture_raw_complete_hf(self, role, data, original, **callbacks):
        self.capture_calls.append((role, data, tuple(sorted(callbacks))))
        return original()

    def _live_admission_identity(self):
        if self.identity_error:
            raise self.identity_error
        return dict(rank=self.rank, pid=self.pid, start_ticks=self.start_ticks, role=self.role,
            state=dict(payload=copy.deepcopy(self.payload), worker_config=copy.deepcopy(self.worker_config)))


def worker_class_fixture():
    events = []
    def matched_stage(phase, errors):
        events.append((phase, list(errors)))
        if errors:
            raise SyntheticMatchedFailure(phase + ': ' + '; '.join(errors))
    def validate_admission(payload):
        require(payload.get('fixture_only') is True, 'Synthetic payload only')
    def install_worker_admission(worker, payload):
        worker.install_calls.append(copy.deepcopy(payload))
        worker.payload = copy.deepcopy(payload)
    def restore_worker_admission_state(worker, state):
        worker.restore_calls.append(copy.deepcopy(state))
        worker.payload = copy.deepcopy(state['payload'])
        worker.worker_config = copy.deepcopy(state['worker_config'])
    namespace = dict(SyntheticOriginalWorker=SyntheticOriginalWorker, copy=copy,
        require=require, equal_state=lambda left, right: left == right, state_digest=state_digest,
        matched_stage=matched_stage, validate_admission=validate_admission,
        install_worker_admission=install_worker_admission,
        restore_worker_admission_state=restore_worker_admission_state,
        validate_live_update_boundary=lambda worker, data: None)
    path = RUNTIME / 'calibration_worker_v1.py'
    tree = ast.parse(path.read_text(), filename=str(path))
    original = next(node for node in tree.body
                    if isinstance(node, ast.ClassDef) and node.name == 'CalibrationWorker')
    wanted = {'calibration_live_admission_identity', 'calibration_install_live_admission',
              'calibration_restore_live_admission', 'update_actor',
              'calibration_raw_current_log_prob', 'compute_log_prob', 'compute_ref_log_prob'}
    methods, found = [], set()
    for method in original.body:
        if isinstance(method, ast.FunctionDef) and method.name in wanted:
            method = copy.deepcopy(method)
            method.decorator_list = []
            methods.append(StripImports().visit(method))
            found.add(method.name)
    if found != wanted:
        raise AssertionError('Incomplete extracted live worker method coverage')
    # Compile as a class so the source's zero-argument super() keeps its real cell.
    selected = ast.ClassDef(name='CalibrationWorker',
        bases=[ast.Name(id='SyntheticOriginalWorker', ctx=ast.Load())], keywords=[],
        body=methods, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[selected], type_ignores=[]))
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace, events


class WorkerLocalAdmissionBoundary(unittest.TestCase):
    def setUp(self):
        self.ns, self.events = worker_class_fixture()
        self.worker = self.ns['CalibrationWorker']()
        self.c1 = dict(config_epoch=1, phase='C1_PRODUCTION', fixture_only=True)

    def rows(self):
        return [self.ns['CalibrationWorker'](rank)._live_admission_identity() for rank in range(8)]

    def test_all8_local_refusal_never_dispatches_original_actor_update(self):
        def reject(worker, data):
            raise ValueError('synthetic sixth-under-C0 refusal')
        self.ns['validate_live_update_boundary'] = reject
        workers = [self.ns['CalibrationWorker'](rank) for rank in range(8)]
        for worker in workers:
            with self.assertRaises(SyntheticMatchedFailure):
                worker.update_actor(SimpleNamespace(meta_info={'fixture_only': True}))
            self.assertEqual(worker.original_calls, [])
        self.assertEqual(len(self.events), 8)
        self.assertTrue(all(phase == 'live_admission/pre_update_boundary' and len(errors) == 1
                            for phase, errors in self.events))

    def test_unknown_original_update_failure_propagates_without_extra_matched_stage(self):
        failure = RuntimeError('synthetic unknown distributed update failure')
        self.worker.original_error = failure
        data = object()
        with self.assertRaises(RuntimeError) as caught:
            self.worker.update_actor(data)
        self.assertIs(caught.exception, failure)
        self.assertEqual(self.worker.original_calls, [data])
        self.assertEqual(self.events, [('live_admission/pre_update_boundary', [])])
        self.assertEqual(self.worker.install_calls + self.worker.restore_calls, [])

    def test_identity_local_error_is_matched_once(self):
        self.worker.identity_error = ValueError('synthetic invalid local admission bytes')
        with self.assertRaises(SyntheticMatchedFailure):
            self.worker.calibration_live_admission_identity()
        self.assertEqual(len(self.events), 1)
        self.assertEqual(self.events[0][0], 'live_admission/local_identity')
        self.assertEqual(len(self.events[0][1]), 1)

    def test_install_requires_known_idle_completed5_and_exact_scheduler(self):
        changes = [lambda worker: worker.fresh200_worker_ledger.update(completed_outer_updates=4),
                   lambda worker: setattr(worker.actor_lr_scheduler, 'last_epoch', 4)]
        for change in changes:
            worker = self.ns['CalibrationWorker']()
            before = worker._admission_protected_state()
            change(worker)
            self.events.clear()
            with self.assertRaises(SyntheticMatchedFailure):
                worker.calibration_install_live_admission(self.c1)
            self.assertEqual(worker.install_calls, [])
            self.assertEqual(worker._admission_protected_state(), before)
            self.assertEqual(len(self.events), 1)
            self.assertEqual(self.events[0][0], 'live_admission/prevalidate_idle_completed5')
        with self.assertRaises(SyntheticMatchedFailure):
            self.worker.calibration_install_live_admission(dict(self.c1, config_epoch=0))
        self.assertEqual(self.worker.install_calls, [])

    def test_install_actor_and_reference_policy_reads_same_protected_state(self):
        for role in ('actor', 'reference'):
            worker = self.ns['CalibrationWorker'](role=role)
            before = worker._admission_protected_state()
            result = worker.calibration_install_live_admission(self.c1)
            self.assertEqual(result['state']['payload'], self.c1)
            self.assertEqual(result['state']['worker_config']['actor']['ppo_micro_batch_size'], 2)
            self.assertEqual(result['protected_state_before_sha256'], state_digest(before))
            self.assertEqual(result['protected_state_after_sha256'], state_digest(before))
            self.assertEqual(worker._admission_protected_state(), before)
            self.assertEqual(len(worker.install_calls), 1)

    def test_install_protected_mutation_is_refused_without_claiming_rollback(self):
        def corrupt(worker, payload):
            worker.install_calls.append(copy.deepcopy(payload))
            worker.payload = copy.deepcopy(payload)
            worker.protected['actual_adam_total'] += 1
        self.ns['install_worker_admission'] = corrupt
        with self.assertRaises(SyntheticMatchedFailure):
            self.worker.calibration_install_live_admission(self.c1)
        self.assertEqual(self.worker.protected['actual_adam_total'], 53)
        self.assertEqual([phase for phase, _ in self.events], [
            'live_admission/prevalidate_idle_completed5', 'live_admission/install_and_exact_readback'])
        self.assertEqual(len(self.events[-1][1]), 1)
        self.assertEqual(self.worker.restore_calls, [])

    def test_restore_one_rank_pid_start_role_or_coverage_mismatch_refuses_before_restore(self):
        changes = [lambda rows: rows[0].update(pid=99999),
                   lambda rows: rows[0].update(start_ticks='99999'),
                   lambda rows: rows[0].update(role='reference'),
                   lambda rows: rows[7].update(rank=6), lambda rows: rows.pop()]
        for change in changes:
            rows = self.rows()
            change(rows)
            self.events.clear()
            with self.assertRaises(SyntheticMatchedFailure):
                self.worker.calibration_restore_live_admission(rows)
            self.assertEqual(self.worker.restore_calls, [])
            self.assertEqual(len(self.events), 1)
            self.assertEqual(self.events[0][0], 'live_admission/restore_and_exact_readback')

    def test_restore_requires_actual_same_metadata_readback(self):
        rows = self.rows()
        rows[0]['state']['payload'] = copy.deepcopy(self.c1)
        result = self.worker.calibration_restore_live_admission(rows)
        self.assertEqual(result, rows[0])
        before = self.worker._admission_protected_state()
        def incomplete_restore(worker, state):
            worker.restore_calls.append(copy.deepcopy(state))
            # Deliberately do not install the saved payload.
        self.ns['restore_worker_admission_state'] = incomplete_restore
        rows[0]['state']['payload'] = dict(self.c1, phase='SYNTHETIC_DIFFERENT_PHASE')
        self.events.clear()
        with self.assertRaises(SyntheticMatchedFailure):
            self.worker.calibration_restore_live_admission(rows)
        self.assertEqual(self.worker._admission_protected_state(), before)
        self.assertEqual(len(self.events), 1)

    def test_restore_protected_mutation_is_refused(self):
        def corrupt(worker, state):
            worker.restore_calls.append(copy.deepcopy(state))
            worker.payload = copy.deepcopy(state['payload'])
            worker.protected['torch_rng'] = 'synthetic-mutated-rng'
        self.ns['restore_worker_admission_state'] = corrupt
        with self.assertRaises(SyntheticMatchedFailure):
            self.worker.calibration_restore_live_admission(self.rows())
        self.assertEqual(self.worker.protected['torch_rng'], 'synthetic-mutated-rng')
        self.assertEqual(len(self.events), 1)

    def test_current_probe_copies_input_and_forwards_actual_micro2_only(self):
        data = SimpleNamespace(meta_info=dict(raw_calibration_observation=True,
            actor_token_identity='sampled_action_prefix_v1', micro_batch_size=4),
            tensors={'original_values': [1, 2]}, non_tensors={'source': [13, 6]})
        before = copy.deepcopy(vars(data))
        current = self.worker.calibration_raw_current_log_prob(data)
        self.assertIsNot(current, data)
        self.assertEqual(vars(data), before)
        self.assertEqual(current.meta_info['micro_batch_size'], 2)
        self.assertEqual(current.tensors, data.tensors)
        self.assertIsNot(current.tensors, data.tensors)
        self.assertEqual(self.worker.capture_calls[0][0], 'current')
        self.assertEqual(self.worker.hf_calls, [('actor', current)])
        self.assertIs(self.worker.compute_log_prob(data), data)
        self.assertIs(self.worker.compute_ref_log_prob(data), data)
        self.assertEqual([role for role, _, _ in self.worker.capture_calls], ['current', 'old', 'reference'])
        self.assertEqual([row.meta_info['micro_batch_size'] for _, row in self.worker.hf_calls], [2, 4, 4])

    def test_current_probe_wrong_micro_refuses_before_original_and_unknown_original_propagates(self):
        data = SimpleNamespace(meta_info=dict(raw_calibration_observation=True, micro_batch_size=4))
        for wrong in (4, True, None):
            self.worker.actor.config.ppo_micro_batch_size = wrong
            self.events.clear()
            with self.subTest(micro=wrong), self.assertRaises(SyntheticMatchedFailure):
                self.worker.calibration_raw_current_log_prob(data)
            self.assertEqual(self.worker.hf_calls, [])
            self.assertEqual(data.meta_info['micro_batch_size'], 4)
            self.assertEqual(len(self.events), 1)
        self.worker.actor.config.ppo_micro_batch_size = 2
        self.worker.original_error = RuntimeError('synthetic unknown original current HF failure')
        self.events.clear()
        with self.assertRaises(RuntimeError) as caught:
            self.worker.calibration_raw_current_log_prob(data)
        self.assertIs(caught.exception, self.worker.original_error)
        self.assertEqual(self.events, [('complete_hf/current/actual_training_micro_metadata', [])])
        self.assertEqual(data.meta_info['micro_batch_size'], 4)


def synthetic_ledger(schema, completed):
    history, total, counted = [], 0, 0
    # The abandoned writes remain in actual totals while completing no outer.
    for index, (processed, writes, abandoned) in enumerate(
            [(7, 3, True)] + [(20, 1, False)] * completed, 1):
        countable = processed == 20 and writes > 0 and not abandoned
        history.append(dict(attempt_id=index, completed_before=counted,
            actual_adam_total_before=total, opportunities_processed=processed,
            actual_successful_adam_writes=writes, skipped_opportunities=processed - writes,
            gradient_micro_batches=writes * 8, actual_adam_total=total + writes,
            scheduler_step=counted + int(countable), abandoned=abandoned,
            counted_complete=countable, safe_to_continue=True))
        total += writes
        counted += int(countable)
    return dict(schema=schema, completed_outer_updates=counted, attempts=len(history),
        skipped_attempts=1, actual_adam_total=total,
        successful_adam_writes_completed_outers=completed,
        successful_adam_writes_abandoned_attempts=3, skipped_opportunities=4 + 19 * completed,
        gradient_micro_batches=total * 8, consumed_training_questions=512 * len(history),
        history=history, active_attempt=None)


def control_namespace():
    namespace = dict(copy=copy, json=json, hashlib=hashlib, os=os, Path=Path,
                     math=math, re=re, require=require)
    selected_functions(SOURCE / 'search_r1/baselines/fresh200.py',
                       {'calibration_initialization', 'validate_attempt_ledger'}, namespace, constants=True)
    selected_functions(SOURCE / 'search_r1/baselines/fresh200_live.py',
                       {'require', 'live_initialization', 'validate_live_initialization', 'validate_live_binding'},
                       namespace, constants=True)
    selected_functions(RUNTIME / 'calibration_contract_v1.py',
                       {'initial_fields', 'validate_initial_fields', 'validate_binding'}, namespace, constants=True)
    namespace['validate_guard_binding'] = lambda value: require(
        value.get('fixture_only') is True, 'Isolated fixture guard seam only')
    # Full cryptographic admission validation is exercised in the source fixture.
    # Here this observer isolates the driver's real completed5 transition checks.
    namespace['validate_admission'] = lambda value: require(
        value.get('fixture_only') is True, 'Isolated fixture admission observer only')
    selected_functions(RUNTIME / 'live_continuation_v10.py',
        {'next_counted_outer', 'validate_completed5_transition', 'quality_report',
         'driver_training_projection', 'immutable_bytes', 'immutable_json'}, namespace)
    return namespace


class CountedControl(unittest.TestCase):
    def setUp(self):
        self.ns = control_namespace()
        self.c0 = dict(config_epoch=0, phase='C0_CALIBRATION', fixture_only=True,
                       live_goal_updated_at=self.ns['LIVE_GOAL_UPDATED_AT'])
        self.c1 = dict(self.c0, config_epoch=1, phase='C1_PRODUCTION')

    def ledger(self, completed):
        value = synthetic_ledger(self.ns['ATTEMPT_SCHEMA'], completed)
        self.ns['validate_attempt_ledger'](value)
        return value

    def test_counted_loop_keeps_partial_writes_and_requires_c1_before_sixth(self):
        ledger = self.ledger(4)
        self.assertEqual(ledger['actual_adam_total'], 7)
        before = copy.deepcopy(ledger)
        self.assertEqual(self.ns['next_counted_outer'](ledger, self.c0), 5)
        self.assertEqual(ledger, before)
        with self.assertRaises(ValueError):
            self.ns['next_counted_outer'](self.ledger(5), self.c0)
        self.assertEqual(self.ns['next_counted_outer'](self.ledger(5), self.c1), 6)
        with self.assertRaises(ValueError):
            self.ns['next_counted_outer'](self.ledger(2), self.c1)

    def test_exact200_stops_without_another_fetch_and_invalid_active_attempt_refuses(self):
        self.assertEqual(self.ns['next_counted_outer'](self.ledger(199), self.c1), 200)
        self.assertIsNone(self.ns['next_counted_outer'](self.ledger(200), self.c1))
        for completed, payload in [(200, self.c0), (201, self.c1),
                (5, dict(self.c1, live_goal_updated_at=1791283983))]:
            with self.subTest(completed=completed), self.assertRaises(ValueError):
                self.ns['next_counted_outer'](self.ledger(completed), payload)
        ledger = self.ledger(5)
        ledger['attempts'] += 1
        ledger['consumed_training_questions'] += 512
        ledger['active_attempt'] = dict(attempt_id=ledger['attempts'],
            completed_before=5, actual_adam_total_before=ledger['actual_adam_total'])
        with self.assertRaises(ValueError):
            self.ns['next_counted_outer'](ledger, self.c1)

    def test_completed5_transition_binds_exact_partial_ledger_and_all8_pause(self):
        ledger = self.ledger(5)
        rows = [dict(rank=rank, pid=10000 + rank, start_ticks=str(20000 + rank),
            model_shards_sha256=str(rank) * 64, optimizer_tensors_sha256='a' * 64,
            rng_tensors_sha256='b' * 64, scheduler_step=5) for rank in range(8)]
        review = dict(fresh200_attempt_ledger=copy.deepcopy(ledger), checkpoint_pause_identity=copy.deepcopy(rows))
        self.ns['validate_completed5_transition'](ledger, review, self.c1, list(reversed(rows)))
        for key, wrong in [('pid', 99999), ('start_ticks', '99999'),
                ('model_shards_sha256', 'c' * 64), ('optimizer_tensors_sha256', 'c' * 64),
                ('rng_tensors_sha256', 'c' * 64), ('scheduler_step', 6)]:
            changed = copy.deepcopy(rows)
            changed[7][key] = wrong
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.ns['validate_completed5_transition'](ledger, review, self.c1, changed)
        for changed in (rows[:-1], [dict(row, rank=0) for row in rows]):
            with self.assertRaises(ValueError):
                self.ns['validate_completed5_transition'](ledger, review, self.c1, changed)
        other = dict(review, fresh200_attempt_ledger=self.ledger(4))
        with self.assertRaises(ValueError):
            self.ns['validate_completed5_transition'](ledger, other, self.c1, rows)
        with self.assertRaises(ValueError):
            self.ns['validate_completed5_transition'](ledger, review, self.c0, rows)

    def test_driver_projection_excludes_policy_only_and_retains_math_rng_data(self):
        before = dict(fresh200_live_admission=self.c0,
            config=dict(actor_rollout_ref=dict(actor=dict(policy_safety={'mode': 'calibration'},
                ppo_micro_batch_size=16)), trainer=dict(total_training_steps=200)),
            rng={'fixture_state': [1, 2]}, sampler={'cursor': 3072},
            actual_attempt_ledger=self.ledger(5), global_steps=5)
        untouched = copy.deepcopy(before)
        after = copy.deepcopy(before)
        after['fresh200_live_admission'] = self.c1
        after['config']['actor_rollout_ref']['actor']['policy_safety'] = {'mode': 'production'}
        project = self.ns['driver_training_projection']
        self.assertEqual(project(before), project(after))
        self.assertEqual(before, untouched)
        for key, value in [('rng', {'fixture_state': [2, 3]}), ('sampler', {'cursor': 3584}),
                ('actual_attempt_ledger', self.ledger(4)), ('global_steps', 6)]:
            changed = dict(after, **{key: value})
            self.assertNotEqual(project(before), project(changed))
        after['config']['actor_rollout_ref']['actor']['ppo_micro_batch_size'] = 8
        self.assertNotEqual(project(before), project(after))

    def test_quality_degradation_reports_without_blocking_safe200(self):
        value = self.ns['quality_report']({'val/f1/all': .1}, {'val/f1/all': .8})
        self.assertTrue(value['below_half_baseline'])
        self.assertTrue(value['report_only'])
        self.assertFalse(value['optimizer_or_outer_rejected_by_quality'])
        self.assertFalse(self.ns['quality_report']({'val/f1/all': .4}, {'val/f1/all': .8})['below_half_baseline'])

    def test_c0_evidence_files_are_exclusive_and_c1_has_distinct_bytes(self):
        with tempfile.TemporaryDirectory(prefix='synthetic-c0-immutable-') as directory:
            c0 = Path(directory) / 'first5-C0.json'
            first = self.ns['immutable_json'](c0, {'phase': 'C0', 'logical_step': 5})
            frozen = c0.read_bytes()
            with self.assertRaises(FileExistsError):
                self.ns['immutable_json'](c0, {'phase': 'C1', 'logical_step': 5})
            c1 = self.ns['immutable_json'](Path(directory) / 'admitted-C1.json', {'phase': 'C1', 'logical_step': 5})
            self.assertEqual(c0.read_bytes(), frozen)
            self.assertEqual(hashlib.sha256(frozen).hexdigest(), first['sha256'])
            self.assertNotEqual(c1['path'], first['path'])

    def test_runtime_binding_calls_real_live_validator_and_rejects_old_two_ready(self):
        binding = dict(self.ns['live_initialization'](), fixture_only=True,
            status='REVIEWED_FOR_FRESH200_LIVE_TRAINING_ONLY', validation_scope=self.ns['LIVE_SCOPE'],
            calibration_scope=self.ns['SCOPE'], live_goal_updated_at=self.ns['LIVE_GOAL_UPDATED_AT'],
            live_authorization_goal_updated_at=self.ns['LIVE_AUTHORIZATION_GOAL_UPDATED_AT'],
            official_training_outer_target=200, initial_calibration_counted_outer_limit=5,
            first_full_acceptance_step=5, first5_count_toward_goal=True, restart_after_calibration=False,
            first5_known_token_kl_enforced=True, first5_tail_observation_only=True,
            production_tail_required_before_outer6=True, retained_training_updates_contributed=0,
            candidate_old_policy_kl=.2, target_old_policy_kl=.2, negative_tail_limit=None,
            outer_counting_authorization_source='human', outer_counting_human_goal_updated_at=1791283983,
            allowed_safety_modes=['calibration', 'production'], source_commit='a' * 40,
            code_commit='a' * 40, source_native_acceptance=None, actual_gpu_acceptance=None,
            actual_probability_acceptance=None, actor_token_identity='sampled_action_prefix_v1',
            capture_actual_runner_inputs=True, capture_actual_normalizer_inputs=True,
            calibration_phase='probability_probe_then_wait', debug_evaluation_rows=512,
            extra_stage64_evaluation=False, prefix_mismatch_limit=0, action_token_mismatch_limit=0,
            actor_recompute_abs_tolerance=None, actor_vllm_abs_tolerance=None,
            actor_recompute_joint_abs_tolerance=None, actor_vllm_joint_abs_tolerance=None,
            probability_tolerance_evidence=[], probability_review_wait_seconds=60)
        self.ns['validate_binding'](binding, 'a' * 40)
        old = dict(binding, status='REVIEWED_FOR_CALIBRATION_ONLY',
            validation_scope=self.ns['SCOPE'], allowed_safety_modes=['calibration'],
            diagnostic_total_optimizer_opportunities=40, calibration_actual_outer_budget=2)
        with self.assertRaises(ValueError):
            self.ns['validate_binding'](old, 'a' * 40)


def actual_lr_boundary_namespace():
    @contextmanager
    def observe_time(*args, **kwargs):
        yield
    namespace = dict(require=require, observe_time=observe_time)
    selected_functions(RUNTIME / 'calibration_driver_v1.py', {'worker_call'}, namespace)
    path = RUNTIME / 'calibration_driver_v1.py'
    tree = ast.parse(path.read_text(), filename=str(path))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_one_outer_body')
    start = next(index for index, node in enumerate(function.body)
                 if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                 and node.targets[0].id == 'before_update_states')
    end = next(index for index in range(start, len(function.body))
               if any(isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                      and node.targets[0].id == 'result' for node in ast.walk(function.body[index])))
    # Exact contiguous original statements through dispatch; no duplicate LR implementation.
    selected = ast.FunctionDef(name='actual_preupdate_boundary',
        args=ast.arguments(posonlyargs=[], args=[ast.arg(arg=name) for name in ('trainer', 'outer', 'progress', 'batch')],
                           vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
        body=copy.deepcopy(function.body[start:end + 1]) + [ast.Return(value=ast.Name(id='result', ctx=ast.Load()))],
        decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[selected], type_ignores=[]))
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace


class OriginalPreupdateLearningRate(unittest.TestCase):
    def test_actual_boundary_caps_after_warmup_and_stops_at_original_dispatch(self):
        function = actual_lr_boundary_namespace()['actual_preupdate_boundary']
        for outer, rate in [(1, 0.), (2, 1e-6 * (1. / 57.)), (57, 1e-6 * (56. / 57.)),
                (58, 1e-6), (59, 1e-6), (200, 1e-6)]:
            events, progress = [], {}
            failure = RuntimeError('synthetic stop before original actor execution')
            rows = [dict(identity=dict(rank=rank, pid=10000 + rank,
                scheduler_step=outer - 1, learning_rates=[rate])) for rank in range(8)]
            def read():
                events.append('actual_scheduler_identity')
                return copy.deepcopy(rows)
            def dispatch(batch):
                events.append(('original_dispatch', progress['stage'], batch))
                raise failure
            trainer = SimpleNamespace(_calibration_progress=progress,
                actor_rollout_wg=SimpleNamespace(calibration_complete_pause_state=read, update_actor=dispatch))
            batch = object()
            with self.subTest(outer=outer), self.assertRaises(RuntimeError) as caught:
                function(trainer, outer, progress, batch)
            self.assertIs(caught.exception, failure)
            self.assertEqual(events, ['actual_scheduler_identity', ('original_dispatch', 'update_actor_driver_dispatch', batch)])

    def test_wrong_single_rank_schedule_refuses_before_original_dispatch(self):
        function = actual_lr_boundary_namespace()['actual_preupdate_boundary']
        for changed in ('scheduler', 'uncapped_rate', 'missing', 'duplicate_rank', 'duplicate_pid', 'empty'):
            rows = [dict(identity=dict(rank=rank, pid=10000 + rank,
                scheduler_step=199, learning_rates=[1e-6])) for rank in range(8)]
            if changed == 'scheduler':
                rows[7]['identity']['scheduler_step'] = 198
            elif changed == 'uncapped_rate':
                rows[7]['identity']['learning_rates'] = [1e-6 * (199. / 57.)]
            elif changed == 'missing':
                rows.pop()
            elif changed == 'duplicate_rank':
                rows[7]['identity']['rank'] = 6
            elif changed == 'duplicate_pid':
                rows[7]['identity']['pid'] = rows[6]['identity']['pid']
            else:
                rows.clear()
            calls = []
            trainer = SimpleNamespace(actor_rollout_wg=SimpleNamespace(
                calibration_complete_pause_state=lambda: rows,
                update_actor=lambda batch: calls.append(batch)))
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                function(trainer, 200, {}, object())
            self.assertEqual(calls, [])


def action_mapping_namespace():
    namespace = dict(torch=torch, copy=copy, numbers=numbers, Mapping=Mapping,
                     dataclass=dataclass, math=math, require=require)
    path = SOURCE / 'search_r1/baselines/action_prefix_forward.py'
    tree = ast.parse(path.read_text(), filename=str(path))
    wanted = {'_require', '_integer_tensor', '_support_tensor', 'prepare_action_prefix_micro'}
    nodes, found = [], set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            nodes.append(copy.deepcopy(node))
        elif isinstance(node, ast.ClassDef) and node.name == 'PreparedActionPrefixMicro':
            nodes.append(copy.deepcopy(node))
        elif isinstance(node, ast.FunctionDef) and node.name in wanted:
            nodes.append(StripImports().visit(copy.deepcopy(node)))
            found.add(node.name)
    if found != wanted:
        raise AssertionError('Missing original action preparation source')
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), 'exec'), namespace)
    selected_functions(RUNTIME / 'raw_hf_observation_v1.py',
        {'require', 'validate_actual_model_inputs', 'unpack_complete_action_calls'}, namespace)
    return namespace


class DifferentPackingRowBridge(unittest.TestCase):
    def raw(self, namespace, role, micro_size):
        sources = [13, 6, 21, 4]
        count = len(sources)
        data = dict(responses=torch.zeros((count, 6), dtype=torch.long),
            actor_action_prefix_ids=torch.zeros((count, 4, 4096), dtype=torch.long),
            actor_action_tokens=torch.zeros((count, 4, 500), dtype=torch.long),
            actor_action_lengths=torch.zeros((count, 4), dtype=torch.long),
            actor_action_prefix_lengths=torch.zeros((count, 4), dtype=torch.long),
            actor_action_output_offsets=torch.full((count, 4), -1, dtype=torch.long),
            actor_action_count=torch.full((count,), 2, dtype=torch.long))
        records, output = [], torch.zeros((count, 6), dtype=torch.float32)
        for row, source in enumerate(sources):
            provenance = []
            for slot, offset, tokens in [(0, 0, [0, source + 30]), (1, 4, [2])]:
                # Valid PAD/special IDs remain actions; prefixes vary across packing groups.
                prefix = [source + 10, 0] + [source + 20] * (row + slot + 1)
                length = len(tokens)
                data['actor_action_prefix_ids'][row, slot, :len(prefix)] = torch.tensor(prefix)
                data['actor_action_prefix_lengths'][row, slot] = len(prefix)
                data['actor_action_tokens'][row, slot, :length] = torch.tensor(tokens)
                data['actor_action_lengths'][row, slot] = length
                data['actor_action_output_offsets'][row, slot] = offset
                data['responses'][row, offset:offset + length] = torch.tensor(tokens)
                output[row, offset:offset + length] = torch.tensor(
                    [-.5 - source * .001 - slot * .01 - index * .001 for index in range(length)])
                provenance.append(dict(step_id=slot, generation_prefix=prefix,
                                       actor_action_tokens=tokens, actor_output_offset=offset))
            records.append(provenance)
        prepare, calls = namespace['prepare_action_prefix_micro'], []
        for start in range(0, count, micro_size):
            micro = {key: value[start:start + micro_size] for key, value in data.items()}
            prepared = prepare(micro, 0, 2)
            flat = prepared.flat_micro
            log_probs = torch.zeros_like(flat['responses'], dtype=torch.float32)
            for local in range(micro_size):
                for slot in range(2):
                    length = int(data['actor_action_lengths'][start + local, slot])
                    offset = int(data['actor_action_output_offsets'][start + local, slot])
                    log_probs[local * 4 + slot, :length] = output[start + local, offset:offset + length]
            support = flat['attention_mask'].bool()
            calls.append(dict(flat_micro=copy.deepcopy(flat), flat_log_probs=log_probs,
                temperature=1., ulysses_sequence_parallel_size=1, use_remove_padding=True,
                observer_errors=[], actual_normalizer_calls=[dict(observer_errors=[], fixture_only=True)],
                model_calls=[dict(args=(), kwargs=dict(input_ids=flat['input_ids'][support].reshape(1, -1),
                    position_ids=flat['position_ids'][support].reshape(1, -1), attention_mask=None, use_cache=False))]))
        return dict(schema='RAW_ACTION_HF_FORWARD_OBSERVATION_V1', role=role, rank=0, pid=10000,
            actor_token_identity='sampled_action_prefix_v1', actual_pad_token_id=0,
            actual_eos_token_id=2, temperature=1., observer_errors=[], input_tensors=data,
            action_prefix_records=records, trajectory_sources=sources,
            trajectory_question_ids=['SYNTHETIC_Q_' + str(value) for value in sources],
            micro_batch_size=micro_size, flat_calls=calls, output_log_probs=output)

    def test_actual_preparation_and_unpack_maps_current2_old4_reference4_by_source_slot(self):
        namespace = action_mapping_namespace()
        mapped = {}
        for role, size in [('current', 2), ('old', 4), ('reference', 4)]:
            raw = self.raw(namespace, role, size)
            mapped[role] = namespace['unpack_complete_action_calls'](raw,
                prepare=namespace['prepare_action_prefix_micro'], pad_id=0, eos_id=2)
            self.assertEqual(set(mapped[role]), {(source, slot) for source in (13, 6, 21, 4) for slot in (0, 1)})
        for key in mapped['current']:
            current = mapped['current'][key]
            for role in ('old', 'reference'):
                other = mapped[role][key]
                for field in ('source', 'step_id', 'prefix', 'tokens', 'positions', 'mask',
                              'offset', 'question_id', 'response_group', 'response_slot'):
                    self.assertEqual(current[field], other[field])
                self.assertTrue(torch.equal(current['log_probs'], other['log_probs']))
        self.assertEqual(mapped['current'][(21, 0)]['normalizer_flat_row'], 0)
        self.assertEqual(mapped['old'][(21, 0)]['normalizer_flat_row'], 8)

    def test_bridge_rejects_duplicate_sources_or_wrong_actual_packed_positions(self):
        namespace = action_mapping_namespace()
        for changed in ('source', 'position'):
            raw = self.raw(namespace, 'current', 2)
            if changed == 'source':
                raw['trajectory_sources'][3] = raw['trajectory_sources'][2]
            else:
                raw['flat_calls'][1]['model_calls'][0]['kwargs']['position_ids'][0, 1] += 1
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                namespace['unpack_complete_action_calls'](raw,
                    prepare=namespace['prepare_action_prefix_micro'], pad_id=0, eos_id=2)


class ListWithToList(list):
    def tolist(self):
        return list(self)


class ObservedDropoutBoundary(unittest.TestCase):
    def fixture(self, config, module_dropout):
        events = []
        def matched_stage(phase, errors):
            events.append((phase, list(errors)))
            if errors:
                raise SyntheticMatchedFailure(phase)
        @contextmanager
        def capture_existing_flat_calls(actor):
            yield []
        namespace = dict(torch=torch, copy=copy, require=require,
            clone_cpu=copy.deepcopy, matched_stage=matched_stage,
            capture_existing_flat_calls=capture_existing_flat_calls,
            SyntheticOriginalWorker=SyntheticOriginalWorker,
            ACTION_PREFIX_KEYS=action_mapping_namespace()['ACTION_PREFIX_KEYS'])
        path = RUNTIME / 'calibration_worker_v1.py'
        tree = ast.parse(path.read_text(), filename=str(path))
        original = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'CalibrationWorker')
        method = copy.deepcopy(next(node for node in original.body
            if isinstance(node, ast.FunctionDef) and node.name == '_capture_raw_complete_hf'))
        method.decorator_list = []
        selected = ast.ClassDef(name='CaptureFixture', bases=[ast.Name(id='SyntheticOriginalWorker', ctx=ast.Load())],
            keywords=[], body=[StripImports().visit(method)], decorator_list=[])
        exec(compile(ast.fix_missing_locations(ast.Module(body=[selected], type_ignores=[])), str(path), 'exec'), namespace)
        worker = namespace['CaptureFixture']()
        module = torch.nn.Sequential(torch.nn.Dropout(module_dropout))
        module.config = SimpleNamespace(to_dict=lambda: copy.deepcopy(config))
        worker.actor.actor_module = module
        keys = list(namespace['ACTION_PREFIX_KEYS']) + ['responses', 'input_ids', 'attention_mask', 'position_ids']
        data = SimpleNamespace(meta_info=dict(actor_token_identity='sampled_action_prefix_v1',
                raw_calibration_observation=True),
            batch={key: torch.zeros((1, 2), dtype=torch.long) for key in keys},
            non_tensor_batch=dict(audit_trajectory_row=[13], audit_question_index=['SYNTHETIC_Q'],
                                  action_prefix_records=ListWithToList([[]])))
        return worker, data, events

    def test_observed_zero_qwen2_dropout_reaches_original_and_unknown_error_propagates(self):
        worker, data, events = self.fixture({'model_type': 'qwen2', 'attention_dropout': 0.}, 0.)
        calls = []
        original_error = RuntimeError('synthetic original model failure')
        def original():
            calls.append('original')
            raise original_error
        with self.assertRaises(RuntimeError) as caught:
            worker._capture_raw_complete_hf('current', data, original)
        self.assertIs(caught.exception, original_error)
        self.assertEqual(calls, ['original'])
        self.assertEqual(events, [('complete_hf/current/before_original_local', [])])

    def test_missing_or_active_observed_dropout_refuses_before_original(self):
        cases = [({'model_type': 'qwen2'}, 0.),
                 ({'model_type': 'qwen2', 'attention_dropout': .1}, 0.),
                 ({'model_type': 'qwen2', 'attention_dropout': 0.}, .1),
                 ({'model_type': 'other', 'attention_dropout': 0.}, 0.),
                 ({'model_type': 'qwen2', 'attention_dropout': False}, 0.)]
        for config, dropout in cases:
            worker, data, events = self.fixture(config, dropout)
            calls = []
            with self.subTest(config=config, dropout=dropout), self.assertRaises(SyntheticMatchedFailure):
                worker._capture_raw_complete_hf('current', data, lambda: calls.append('original'))
            self.assertEqual(calls, [])
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0][0], 'complete_hf/current/before_original_local')


class ForbiddenWorkerRPC:
    def __init__(self):
        self.accesses = []

    def __getattr__(self, name):
        self.accesses.append(name)
        raise AssertionError('Timing fixture must never dispatch worker RPC: ' + name)


def timing_namespace(ticks, forbidden_files=False):
    values = iter(ticks)
    attempted_files = []
    def monotonic():
        value = next(values)
        if isinstance(value, BaseException):
            raise value
        return value
    def forbidden_path(*args, **kwargs):
        attempted_files.append((args, kwargs))
        raise AssertionError('observe_time must not write a timing artifact')
    namespace = dict(time=SimpleNamespace(monotonic=monotonic), json=json, os=os,
                     Path=forbidden_path if forbidden_files else Path, attempted_files=attempted_files)
    selected_functions(RUNTIME / 'continuation_timing_v10.py',
                       {'observe_time', 'publish_timing'}, namespace)
    namespace['observe_time'] = contextmanager(namespace['observe_time'])
    return namespace


class PassiveTiming(unittest.TestCase):
    def trainer(self):
        return SimpleNamespace(actual_outer_updates=5, fresh200_attempt_ledger=dict(
            completed_outer_updates=5, actual_adam_total=52, attempts=6),
            _calibration_progress=dict(attempt_id=7),
            actor_rollout_wg=ForbiddenWorkerRPC(), ref_policy_wg=ForbiddenWorkerRPC())

    def test_unknown_yield_exception_preserved_and_partial_timing_stays_in_memory(self):
        namespace = timing_namespace([10., 11.5, 20., 23.], forbidden_files=True)
        trainer = self.trainer()
        before = copy.deepcopy(trainer.fresh200_attempt_ledger)
        with namespace['observe_time'](trainer, 'synthetic_generation_retrieval', 'generation_retrieval'):
            pass
        original = RuntimeError('synthetic original unknown worker error')
        with self.assertRaises(RuntimeError) as caught:
            with namespace['observe_time'](trainer, 'synthetic_actor_update'):
                raise original
        self.assertIs(caught.exception, original)
        self.assertEqual(trainer.fresh200_attempt_ledger, before)
        rows = trainer._calibration_progress['timings']
        self.assertEqual(len(rows), 2)
        self.assertEqual([row['wall_seconds'] for row in rows], [1.5, 3.])
        self.assertEqual([row['original_call_returned'] for row in rows], [True, False])
        self.assertEqual([row['attempt_id'] for row in rows], [7, 7])
        self.assertEqual(trainer.actor_rollout_wg.accesses + trainer.ref_policy_wg.accesses, [])
        self.assertEqual(namespace['attempted_files'], [])

    def test_measurement_finally_error_never_replaces_unknown_original_exception(self):
        measurement = RuntimeError('synthetic timer failed')
        namespace = timing_namespace([10., measurement], forbidden_files=True)
        trainer = self.trainer()
        prior = dict(stage='SYNTHETIC_PRIOR', wall_seconds=1.)
        trainer._calibration_progress['timings'] = [prior]
        original = ValueError('synthetic original unknown error')
        with self.assertRaises(ValueError) as caught:
            with namespace['observe_time'](trainer, 'synthetic_unknown_failure'):
                raise original
        self.assertIs(caught.exception, original)
        self.assertEqual(trainer._calibration_progress['timings'], [prior])
        self.assertEqual(trainer.actor_rollout_wg.accesses + trainer.ref_policy_wg.accesses, [])
        self.assertEqual(namespace['attempted_files'], [])

    def test_publication_records_actual_attempt_counters_without_gpu_or_server_claims(self):
        namespace = timing_namespace([1., 2.])
        trainer = self.trainer()
        trainer.continuation_timings = [dict(stage='SYNTHETIC_REVIEW', wall_seconds=2.)]
        with namespace['observe_time'](trainer, 'synthetic_training'):
            pass
        before = copy.deepcopy(trainer.fresh200_attempt_ledger)
        identity = dict(SLURM_JOB_ID='SYNTHETIC_CPU_JOB', SEARCH_R1_BASELINE_RUN_ID='SYNTHETIC_CPU_RUN',
            SEARCH_R1_BASELINE_COMMIT='a' * 40, SEARCH_R1_BASELINE_CONFIG_SHA256='c' * 64)
        with tempfile.TemporaryDirectory(prefix='synthetic-passive-timing-') as directory:
            with patch.dict(os.environ, identity):
                namespace['publish_timing'](trainer, Path(directory), outcome={'counted_complete': False})
            rows = [json.loads(line) for line in (Path(directory) / 'timing-progress.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual((row['completed_outer_updates'], row['actual_adam_total'], row['attempts']), (5, 52, 6))
        self.assertEqual(len(row['timings']), 2)
        self.assertTrue(row['generation_retrieval_is_client_combined'])
        self.assertIsNone(row['e5_server_cpu_or_network_attribution'])
        self.assertFalse(row['benchmark_or_gpu_acceptance_claim'])
        self.assertEqual(trainer.fresh200_attempt_ledger, before)
        self.assertEqual(trainer.continuation_timings, [])
        self.assertEqual(trainer._calibration_progress['timings'], [])
        self.assertEqual(trainer.actor_rollout_wg.accesses + trainer.ref_policy_wg.accesses, [])


if __name__ == '__main__':
    unittest.main()
