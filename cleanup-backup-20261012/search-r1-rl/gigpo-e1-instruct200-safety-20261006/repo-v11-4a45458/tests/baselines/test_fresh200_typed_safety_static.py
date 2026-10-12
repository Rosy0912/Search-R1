"""UNEXECUTED CPU fixtures for narrow structural and numeric refusal provenance.

The actor bodies below are extracted from the candidate source. Only imports,
decorators, and annotations are substituted; validator, collective, diagnostic,
and exception-control bodies are retained. Deterministic CPU transports are
local fixtures, not proof of an actual multi-rank or GPU training outcome.
"""
import ast
import copy
import json
import math
import numbers
import os
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace

import pytest
import torch

from search_r1.baselines.action_prefix_forward import (
    ACTION_PREFIX_KEYS, MAX_ACTION, MAX_ACTIONS, MAX_PREFIX,
    ActionPrefixStructureError, prepare_action_prefix_micro,
    scatter_action_prefix_outputs, validate_action_statistics_structure)
from search_r1.baselines.partial_update import _adam_steps, rejection_disposition
from search_r1.baselines.policy_safety import (
    PolicySafetyRejected, calibrated_or_candidate_rejection, global_statistics,
    local_statistics)
from tests.baselines.test_fresh200_worker_skip_static import ROOT, worker_fixture


class ImportsProvidedByFixture(ast.NodeTransformer):
    def visit_Import(self, node):
        return ast.copy_location(ast.Pass(), node)

    def visit_ImportFrom(self, node):
        return ast.copy_location(ast.Pass(), node)


def actual_function(relative, name, namespace):
    source = (ROOT / relative).read_text()
    node = next(value for value in ast.walk(ast.parse(source))
                if isinstance(value, ast.FunctionDef) and value.name == name)
    node = copy.deepcopy(node)
    node.decorator_list = []
    node.returns = None
    for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
        argument.annotation = None
    node = ImportsProvidedByFixture().visit(node)
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, str(ROOT / relative), 'exec'), namespace)
    return namespace[name]


def actual_gradient_admission(namespace):
    """Retain the actual post-forward structure/statistics/common decision body."""
    path = ROOT / 'verl/workers/actor/dp_actor.py'
    source = path.read_text()
    update = next(node for node in ast.walk(ast.parse(source))
                  if isinstance(node, ast.FunctionDef) and node.name == 'update_policy')
    stage = next(node for node in ast.walk(update)
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name)
        and node.test.id == 'safety_enabled' and 'probe_stats = local_statistics' in ast.get_source_segment(source, node))
    skeleton = ast.parse('def admit(self,old_log_prob,log_prob,advantages,response_mask,data,safety,'
        'outer_identity,batch_idx,micro_index,updates,safety_enabled=True):\n    pass\n').body[0]
    skeleton.body = [copy.deepcopy(stage)]
    skeleton = ImportsProvidedByFixture().visit(skeleton)
    module = ast.fix_missing_locations(ast.Module(body=[skeleton], type_ignores=[]))
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace['admit']


class TorchCPUTransport:
    def __init__(self, rank=0, decisions=(), unknown_at=None, original=None):
        self.reductions = []
        self.decisions = list(decisions)
        self.unknown_at = unknown_at
        self.original = original
        self.distributed = SimpleNamespace(is_initialized=lambda: True,
            get_rank=lambda: rank, all_reduce=self.reduce,
            ReduceOp=SimpleNamespace(MAX='MAX'))

    def __getattr__(self, name):
        return getattr(torch, name)

    def reduce(self, flag, op, group=None):
        self.reductions.append(int(flag))
        if len(self.reductions) == self.unknown_at:
            raise self.original
        if len(self.reductions) <= len(self.decisions):
            flag.fill_(self.decisions[len(self.reductions) - 1])


class CPUMicro(dict):
    def cuda(self):
        return self


class CPUMonitor(dict):
    def __init__(self, micro):
        super().__init__(micro)
        self.micro = micro

    def split(self, size):
        return [self.micro]


def action_micro():
    micro = CPUMicro(responses=torch.tensor([[11, 12, 13]], dtype=torch.long),
        old_log_probs=torch.zeros(1, 3, dtype=torch.float64),
        advantages=torch.tensor([[-1., -1., -2.]], dtype=torch.float64),
        policy_weights=torch.tensor([[.25, .25, .5]], dtype=torch.float64),
        loss_mask=torch.ones(1, 3), step_ids=torch.tensor([[0, 0, 1]]),
        attention_mask=torch.ones(1, 5, dtype=torch.long),
        actor_action_prefix_ids=torch.zeros(1, MAX_ACTIONS, MAX_PREFIX, dtype=torch.long),
        actor_action_tokens=torch.zeros(1, MAX_ACTIONS, MAX_ACTION, dtype=torch.long),
        actor_action_lengths=torch.tensor([[2, 1, 0, 0]]),
        actor_action_prefix_lengths=torch.tensor([[1, 2, 0, 0]]),
        actor_action_output_offsets=torch.tensor([[0, 2, -1, -1]]),
        actor_action_count=torch.tensor([2]))
    micro['actor_action_prefix_ids'][0, 0, 0] = 5
    micro['actor_action_prefix_ids'][0, 1, :2] = torch.tensor([5, 11])
    micro['actor_action_tokens'][0, 0, :2] = torch.tensor([11, 12])
    micro['actor_action_tokens'][0, 1, 0] = 13
    return micro


def actual_actor_fixture(tmp_path, monkeypatch, *, rank=0, decisions=(),
                         unknown_at=None, original=None, io_error=None):
    monkeypatch.setenv('SEARCH_R1_BASELINE_AUDIT_DIR', str(tmp_path / 'diagnostics'))
    transport = TorchCPUTransport(rank, decisions, unknown_at, original)
    diagnostics = []

    def save(state, path):
        diagnostics.append(copy.deepcopy(state))
        if io_error is not None:
            raise io_error
        return dict(path=str(path))

    namespace = dict(torch=transport, os=os, json=json, sys=sys, Path=Path,
        math=math, numbers=numbers, ACTION_PREFIX_KEYS=ACTION_PREFIX_KEYS,
        FSDP=type('NotCPUFSDP', (), {}), PolicySafetyRejected=PolicySafetyRejected,
        ActionPrefixStructureError=ActionPrefixStructureError,
        prepare_action_prefix_micro=prepare_action_prefix_micro,
        scatter_action_prefix_outputs=scatter_action_prefix_outputs,
        validate_action_statistics_structure=validate_action_statistics_structure,
        atomic_torch_save=save, cpu_snapshot=lambda value: value)
    model = torch.nn.Linear(1, 1, bias=False)
    actor = SimpleNamespace(actor_module=model,
        actor_optimizer=torch.optim.AdamW(model.parameters(), lr=.01),
        _action_prefix_batch_active=True, _action_prefix_token_ids=(0, 2),
        _fresh200_attempt_id=1, _fresh200_attempt_adam_writes=2,
        _fresh200_attempt_gradient_micros=16,
        _action_prefix_forward_context=dict(outer_step=1, mini_index=2,
            phase='gradient_action_prefix', writes=2))
    sync = actual_function('verl/workers/actor/dp_actor.py', '_safety_sync_invalid', namespace)
    actor._safety_sync_invalid = MethodType(sync, actor)
    forward = actual_function('verl/workers/actor/dp_actor.py', '_forward_micro_batch', namespace)
    return actor, forward, namespace, transport, diagnostics


def actual_monitor_fixture(tmp_path, monkeypatch, *, micro=None, current=None,
                           stage=None, original=None, io_error=None):
    actor, _, namespace, transport, diagnostics = actual_actor_fixture(
        tmp_path, monkeypatch, io_error=io_error)
    micro = action_micro() if micro is None else micro
    current = torch.zeros_like(micro['old_log_probs']) if current is None else current
    monitor = CPUMonitor(micro)
    events = []
    at_failure = []

    def fail(name):
        if stage == name:
            at_failure.append(len(transport.reductions))
            raise original

    class Probe:
        invalid = False

        def __init__(self, *args):
            fail('probe_construct')

        def __enter__(self):
            events.append('enter')
            fail('probe_enter')
            return self

        def __exit__(self, *args):
            events.append('exit')
            fail('probe_exit')

    def forward(*args, **kwargs):
        events.append('forward')
        fail('forward')
        return torch.zeros_like(current), current

    def statistics(*args):
        events.append('local_statistics')
        fail('local_statistics')
        return local_statistics(*args)

    def merged_statistics(*args):
        events.append('global_statistics')
        fail('global_statistics')
        return global_statistics(*args)

    def tensor_hash(value):
        return repr(value.tolist())

    namespace.update(TransparentProbe=Probe, tensor_hash=tensor_hash,
        local_statistics=statistics, global_statistics=merged_statistics,
        calibrated_or_candidate_rejection=calibrated_or_candidate_rejection,
        CALIBRATION_SCOPE='CPU_FIXTURE_UNUSED_CALIBRATION_SCOPE')
    actor._forward_micro_batch = forward
    actor.config = SimpleNamespace(ppo_micro_batch_size=2, state_masking=True)
    actor._safety_monitor_identity = {key: tensor_hash(value) for key, value in monitor.items()}
    guard = actual_function('verl/workers/actor/dp_actor.py', '_policy_safety_guard', namespace)
    control = dict(dual_clip=3., kl_log_bound=30., mode='native_fixture',
                   target_old_policy_kl=.2, negative_tail_limit=1.)
    return actor, guard, monitor, control, transport, diagnostics, events, at_failure


@pytest.mark.parametrize('changed', ['negative_identity', 'identity_shape', 'identity_dtype',
    'advantage_disagreement', 'fractional_mask', 'nan_mask', 'negative_weight'])
def test_pure_action_structure_validator_rejects_before_any_statistics(changed):
    micro = action_micro()
    if changed == 'negative_identity':
        micro['step_ids'][0, 0] = -1
    elif changed == 'identity_shape':
        micro['step_ids'] = micro['step_ids'][:, :2]
    elif changed == 'identity_dtype':
        micro['step_ids'] = micro['step_ids'].float()
    elif changed == 'advantage_disagreement':
        micro['advantages'][0, 1] = -3.
    elif changed == 'negative_weight':
        micro['policy_weights'][0, 0] = -.1
    else:
        micro['loss_mask'][0, 0] = float('nan') if changed == 'nan_mask' else .5
    with pytest.raises(ActionPrefixStructureError):
        validate_action_statistics_structure(micro['old_log_probs'], None,
            micro['advantages'], micro['loss_mask'], micro['policy_weights'], micro['step_ids'])


def test_action_prefix_token_identity_misalignment_is_explicit_structure():
    micro = action_micro()
    micro['step_ids'][0, 1] = 1
    with pytest.raises(ActionPrefixStructureError, match='step IDs'):
        prepare_action_prefix_micro(micro, 0, 2)


@pytest.mark.parametrize('field', ['old_log_probs', 'advantages', 'policy_weights'])
def test_nonfinite_numeric_inputs_are_preserved_for_original_numeric_statistics(field):
    micro = action_micro()
    micro[field][0, 0] = float('nan')
    validate_action_statistics_structure(micro['old_log_probs'], None,
        micro['advantages'], micro['loss_mask'], micro['policy_weights'], micro['step_ids'])
    part = local_statistics(micro['old_log_probs'], torch.zeros(1, 3),
        micro['advantages'], micro['loss_mask'], micro['policy_weights'], micro['step_ids'], 3., 30.)
    assert int(part[0][3]) > 0


@pytest.mark.parametrize('error_type', [ValueError, RuntimeError, torch.cuda.OutOfMemoryError])
def test_unknown_torch_error_inside_explicit_validator_remains_same_object(monkeypatch, error_type):
    micro = action_micro()
    original = error_type('unknown original finite-advantage comparison failure')

    def comparison(*args, **kwargs):
        raise original

    monkeypatch.setattr(torch, 'allclose', comparison)
    with pytest.raises(error_type) as caught:
        validate_action_statistics_structure(micro['old_log_probs'], None,
            micro['advantages'], micro['loss_mask'], micro['policy_weights'], micro['step_ids'])
    assert caught.value is original and not isinstance(caught.value, ActionPrefixStructureError)


@pytest.mark.parametrize('stage', ['prepare', 'scatter', 'flat_forward'])
@pytest.mark.parametrize('error_type', [ValueError, RuntimeError, torch.cuda.OutOfMemoryError])
def test_unknown_action_prefix_errors_escape_without_new_max_or_recovery(
        tmp_path, monkeypatch, stage, error_type):
    original = error_type('unknown original action-prefix ' + stage)
    actor, forward, namespace, transport, diagnostics = actual_actor_fixture(tmp_path, monkeypatch)
    worker, data, method, calls, stages, saved = worker_fixture(tmp_path / 'worker', {})
    method.__globals__['PolicySafetyRejected'] = PolicySafetyRejected
    actor.actor_module = worker.actor_module_fsdp
    actor.actor_optimizer = worker.actor_optimizer
    actor._fresh200_attempt_adam_writes = 0
    actor._fresh200_attempt_gradient_micros = 0
    worker.actor = actor
    prepared = prepare_action_prefix_micro(action_micro(), 0, 2)
    at_failure = []

    def fail(*args, **kwargs):
        at_failure.append(len(transport.reductions))
        raise original

    actor._forward_flat_micro_batch = lambda *args: (
        torch.zeros(MAX_ACTIONS, prepared.action_width),
        torch.zeros(MAX_ACTIONS, prepared.action_width))
    if stage == 'prepare':
        namespace['prepare_action_prefix_micro'] = fail
    elif stage == 'scatter':
        namespace['scatter_action_prefix_outputs'] = fail
    else:
        actor._forward_flat_micro_batch = fail

    def update(data):
        for mini in range(2):
            calls.append(mini)
            worker.actor_optimizer.zero_grad(set_to_none=True)
            worker.actor_module_fsdp.weight.grad = torch.ones_like(worker.actor_module_fsdp.weight)
            worker.actor_optimizer.step()
            actor._fresh200_attempt_adam_writes += 1
            actor._fresh200_attempt_gradient_micros += 8
        calls.append(2)
        return forward(actor, action_micro(), 1.)

    actor.update_policy = update
    with pytest.raises(error_type) as caught:
        method(worker, data)
    assert caught.value is original and not isinstance(caught.value, PolicySafetyRejected)
    assert len(transport.reductions) == at_failure[0]
    assert transport.reductions == ([] if stage == 'prepare' else [0])
    assert diagnostics == [] and saved == {}
    assert calls == [0, 1, 2] and stages == ['attempt/entry_actual_counters']
    assert _adam_steps({'optimizer': worker.actor_optimizer.state_dict()}) == 2
    assert actor._fresh200_attempt_adam_writes == 2
    assert actor._fresh200_attempt_gradient_micros == 16
    assert torch.equal(worker.actor_module_fsdp.weight.grad, torch.ones_like(worker.actor_module_fsdp.weight))
    assert data.meta_info['fresh200_processed_opportunities'] == 0
    assert worker.actor_lr_scheduler.last_epoch == 0 and not hasattr(worker, 'fresh200_worker_ledger')


@pytest.mark.parametrize('stage', ['prepare', 'scatter'])
def test_explicit_prefix_structure_refusal_is_common_and_fatal(tmp_path, monkeypatch, stage):
    records = []
    for rank in range(2):
        decisions = [2, 0] if stage == 'prepare' else [0, 2, 0]
        actor, forward, _, transport, diagnostics = actual_actor_fixture(
            tmp_path / str(rank), monkeypatch, rank=rank, decisions=decisions)
        micro = action_micro()
        flat_calls = []
        if rank == 0 and stage == 'prepare':
            micro['loss_mask'][0, 1] = 0
        prepared = prepare_action_prefix_micro(action_micro(), 0, 2)

        def flat(*args):
            flat_calls.append(1)
            width = prepared.action_width + (1 if rank == 0 and stage == 'scatter' else 0)
            return torch.zeros(MAX_ACTIONS, width), torch.zeros(MAX_ACTIONS, width)

        actor._forward_flat_micro_batch = flat
        with pytest.raises(PolicySafetyRejected) as caught:
            forward(actor, micro, 1.)
        record = caught.value.record
        records.append(record)
        assert record['refusal_kind'] == 'structural'
        assert record['local_invalid'] is (rank == 0)
        assert 'matched_guard' not in record and record['global_diagnostic_io_failed'] is False
        assert rejection_disposition(record) == 'fail_closed'
        assert flat_calls == ([] if stage == 'prepare' else [1])
        assert len(transport.reductions) == len(decisions) and len(diagnostics) == 1
    assert {rejection_disposition(record) for record in records} == {'fail_closed'}


@pytest.mark.parametrize('changed', ['negative_identity', 'advantage_disagreement', 'malformed_mask'])
def test_actual_monitor_prevalidation_stops_before_forward_and_statistics(tmp_path, monkeypatch, changed):
    micro = action_micro()
    if changed == 'negative_identity':
        micro['step_ids'][0, 1] = -1
    elif changed == 'advantage_disagreement':
        micro['advantages'][0, 1] = -3.
    else:
        micro['loss_mask'][0, 1] = .5
    actor, guard, monitor, control, _, diagnostics, events, _ = actual_monitor_fixture(
        tmp_path, monkeypatch, micro=micro)
    with pytest.raises(PolicySafetyRejected) as caught:
        guard(actor, monitor, 1., control, dict(outer_step=1, mini_index=2), 'before', 2)
    assert events == ['enter', 'exit']
    assert caught.value.record['refusal_kind'] == 'structural'
    assert caught.value.record['phase'] == 'before_schema0'
    assert rejection_disposition(caught.value.record) == 'fail_closed'
    assert len(diagnostics) == 1


@pytest.mark.parametrize('changed', ['negative_identity', 'advantage_disagreement', 'malformed_mask'])
def test_actual_gradient_structure_stops_before_statistics_and_is_fatal_on_clean_peer(
        tmp_path, monkeypatch, changed):
    for rank in range(2):
        actor, _, namespace, transport, diagnostics = actual_actor_fixture(
            tmp_path / str(rank), monkeypatch, rank=rank, decisions=(2, 0))
        actor.config = SimpleNamespace(use_kl_loss=False)
        micro = action_micro()
        if rank == 0:
            if changed == 'negative_identity':
                micro['step_ids'][0, 0] = -1
            elif changed == 'advantage_disagreement':
                micro['advantages'][0, 1] = -3.
            else:
                micro['loss_mask'][0, 0] = .5
        statistic_calls = []
        def statistics(*args):
            statistic_calls.append(1)
            return local_statistics(*args)
        namespace['local_statistics'] = statistics
        admit = actual_gradient_admission(namespace)
        with pytest.raises(PolicySafetyRejected) as caught:
            admit(actor, micro['old_log_probs'], torch.zeros(1, 3), micro['advantages'],
                micro['loss_mask'], micro, dict(dual_clip=3., kl_log_bound=30.),
                dict(baseline_outer_step=1), 2, 0, 2)
        record = caught.value.record
        assert statistic_calls == ([] if rank == 0 else [1])
        assert record['local_invalid'] is (rank == 0)
        assert record['refusal_kind'] == 'structural' and 'matched_guard' not in record
        assert rejection_disposition(record) == 'fail_closed'
        assert transport.reductions == ([2, 0] if rank == 0 else [0, 0])
        assert len(diagnostics) == 1


@pytest.mark.parametrize('error_type', [ValueError, RuntimeError, torch.cuda.OutOfMemoryError])
def test_unknown_actual_gradient_statistics_exception_has_no_new_max_or_diagnostic(
        tmp_path, monkeypatch, error_type):
    original = error_type('unknown original gradient local_statistics failure')
    actor, _, namespace, transport, diagnostics = actual_actor_fixture(tmp_path, monkeypatch)
    actor.config = SimpleNamespace(use_kl_loss=False)
    def statistics(*args):
        raise original
    namespace['local_statistics'] = statistics
    admit = actual_gradient_admission(namespace)
    micro = action_micro()
    with pytest.raises(error_type) as caught:
        admit(actor, micro['old_log_probs'], torch.zeros(1, 3), micro['advantages'],
            micro['loss_mask'], micro, dict(dual_clip=3., kl_log_bound=30.),
            dict(baseline_outer_step=1), 2, 0, 2)
    assert caught.value is original and not isinstance(caught.value, PolicySafetyRejected)
    assert transport.reductions == [] and diagnostics == []
    assert _adam_steps({'optimizer': actor.actor_optimizer.state_dict()}) == 0


@pytest.mark.parametrize('numeric', ['nan_probability', 'delta_above_bound'])
def test_actual_gradient_numeric_decision_remains_a_recognized_skip(tmp_path, monkeypatch, numeric):
    actor, _, namespace, transport, diagnostics = actual_actor_fixture(tmp_path, monkeypatch)
    actor.config = SimpleNamespace(use_kl_loss=False)
    namespace['local_statistics'] = local_statistics
    admit = actual_gradient_admission(namespace)
    micro = action_micro()
    current = torch.zeros(1, 3, dtype=torch.float64)
    current[0, 0] = float('nan') if numeric == 'nan_probability' else 31.
    with pytest.raises(PolicySafetyRejected) as caught:
        admit(actor, micro['old_log_probs'], current, micro['advantages'], micro['loss_mask'],
            micro, dict(dual_clip=3., kl_log_bound=30.), dict(baseline_outer_step=1), 2, 0, 2)
    record = caught.value.record
    assert record['refusal_kind'] == 'known_numeric'
    assert record['matched_guard'] == 'ORIGINAL_ALL_REDUCE_MAX_NUMERIC_REFUSAL'
    assert record['details'] == 'NONFINITE_OR_EXP_RANGE_EFFECTIVE_GRADIENT_INPUT'
    assert rejection_disposition(record) == 'skip_opportunity'
    assert transport.reductions == [1, 0] and len(diagnostics) == 1


@pytest.mark.parametrize('stage', ['probe_construct', 'probe_enter', 'forward',
    'local_statistics', 'probe_exit', 'global_statistics'])
@pytest.mark.parametrize('error_type', [ValueError, RuntimeError, torch.cuda.OutOfMemoryError])
def test_unknown_monitor_statistic_probe_or_collective_error_is_original(
        tmp_path, monkeypatch, stage, error_type):
    original = error_type('unknown original monitor ' + stage)
    actor, guard, monitor, control, transport, diagnostics, _, at_failure = actual_monitor_fixture(
        tmp_path, monkeypatch, stage=stage, original=original)
    with pytest.raises(error_type) as caught:
        guard(actor, monitor, 1., control, dict(outer_step=1, mini_index=2), 'before', 2)
    assert caught.value is original and not isinstance(caught.value, PolicySafetyRejected)
    assert len(transport.reductions) == at_failure[0]
    assert diagnostics == []
    assert _adam_steps({'optimizer': actor.actor_optimizer.state_dict()}) == 0


@pytest.mark.parametrize('numeric', ['nan_probability', 'delta_above_bound'])
def test_actual_known_numeric_monitor_refusal_retains_partial_writes_and_next_opportunity(
        tmp_path, monkeypatch, numeric):
    current = torch.zeros(1, 3, dtype=torch.float64)
    current[0, 0] = float('nan') if numeric == 'nan_probability' else 31.
    actor, guard, monitor, control, _, diagnostics, _, _ = actual_monitor_fixture(
        tmp_path, monkeypatch, current=current)
    worker, data, method, calls, stages, saved = worker_fixture(tmp_path / 'worker', {})
    method.__globals__['PolicySafetyRejected'] = PolicySafetyRejected
    healthy_update = worker.actor.update_policy
    original_actor = worker.actor
    original_actor.__dict__.update(actor.__dict__)
    original_actor._safety_sync_invalid = MethodType(actor._safety_sync_invalid.__func__, original_actor)
    actor = original_actor
    actor.actor_module = worker.actor_module_fsdp
    actor.actor_optimizer = worker.actor_optimizer
    actor._fresh200_attempt_adam_writes = 0
    actor._fresh200_attempt_gradient_micros = 0

    def update(data):
        if data.meta_info['fresh200_processed_opportunities']:
            return healthy_update(data)
        for mini in range(2):
            calls.append(mini)
            worker.actor_optimizer.zero_grad(set_to_none=True)
            worker.actor_module_fsdp.weight.grad = torch.ones_like(worker.actor_module_fsdp.weight)
            worker.actor_optimizer.step()
            actor._fresh200_attempt_adam_writes += 1
            actor._fresh200_attempt_gradient_micros += 8
        calls.append(2)
        guard(actor, monitor, 1., control, dict(outer_step=1, mini_index=2), 'before', 2)

    actor.update_policy = update
    _, receipt = method(worker, data)
    refusal = receipt['refusals'][0]
    record = refusal['partial_update']['rejection']
    assert record['refusal_kind'] == 'known_numeric_monitor'
    assert record['matched_guard'] == 'ORIGINAL_GLOBAL_STATISTICS_NUMERIC_REFUSAL'
    assert record['global_diagnostic_io_failed'] is False
    assert rejection_disposition(record) == 'skip_opportunity'
    assert record['actual_optimizer_writes'] == refusal['partial_update']['actual_adam_total'] == 2
    assert saved[refusal['state_receipt']['path']]['partial_update']['rejection'] == record
    assert calls == list(range(20)) and receipt['actual_successful_adam_writes'] == 19
    assert receipt['counted_complete'] is True and receipt['gradient_micro_batches'] == 152
    assert _adam_steps({'optimizer': worker.actor_optimizer.state_dict()}) == 19
    assert worker.actor_lr_scheduler.last_epoch == 0 and diagnostics


@pytest.mark.parametrize('stage', ['decision', 'diagnostic_io'])
@pytest.mark.parametrize('error_type', [ValueError, RuntimeError, torch.cuda.OutOfMemoryError])
def test_unknown_original_refusal_collective_does_not_create_marker_or_retry(
        tmp_path, monkeypatch, stage, error_type):
    original = error_type('unknown original refusal ' + stage + ' MAX')
    actor, _, _, transport, diagnostics = actual_actor_fixture(tmp_path, monkeypatch,
        unknown_at=1 if stage == 'decision' else 2, original=original)
    with pytest.raises(error_type) as caught:
        actor._safety_sync_invalid(True, 'cpu', dict(outer_step=1, mini_index=2),
            'gradient_micro0_loss', 2, details='NONFINITE_TRAINING_LOSS', refusal_kind='known_numeric')
    assert caught.value is original and not isinstance(caught.value, PolicySafetyRejected)
    assert len(transport.reductions) == (1 if stage == 'decision' else 2)
    assert len(diagnostics) == (0 if stage == 'decision' else 1)
    assert all('matched_guard' not in state['record'] for state in diagnostics)


def test_phase_alone_and_non_numeric_classifications_never_authorize_safe_skip():
    for phase in ('before', 'after', 'gradient_micro0', 'gradient_micro0_loss',
                  'gradient_action_prefix_schema', 'before_schema0', 'after_stats0'):
        assert rejection_disposition(dict(phase=phase, mini_index=2)) == 'fail_closed'
        assert rejection_disposition(dict(phase=phase, mini_index=2,
            refusal_kind='structural', global_diagnostic_io_failed=False)) == 'fail_closed'
        assert rejection_disposition(dict(phase=phase, mini_index=2,
            refusal_kind='terminal_state', global_diagnostic_io_failed=False)) == 'fail_closed'
