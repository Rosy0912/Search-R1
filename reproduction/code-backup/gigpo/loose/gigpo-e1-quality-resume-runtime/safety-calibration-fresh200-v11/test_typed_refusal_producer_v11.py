"""UNEXECUTED actual source loss-producer -> source disposition -> runtime audit.

No fixture manufactures a matched_guard. Actor/class/disposition bodies are read
from the exactly selected source and compiled only during future authorized CPU
tests. The bounded two-process Gloo case uses real CPU collective transport; it
does not authorize or demonstrate GPU, Ray, Slurm, or formal training behavior.
"""
import ast
import copy
from datetime import timedelta
import json
import multiprocessing
import os
from pathlib import Path
import time
from types import MethodType, SimpleNamespace

import pytest
import torch

from calibration_audit_v1 import validate_attempt_numeric, validate_calibration_update
from test_calibration_audit_v1 import empty_micro


def source_root():
    """Use one canonical helper-bearing PYTHONPATH root, or the local successor."""
    candidates = set()
    for entry in os.environ.get('PYTHONPATH', '').split(os.pathsep):
        if not entry:
            continue
        root = Path(entry).resolve()
        helper = root / 'search_r1/baselines/action_prefix_forward.py'
        if helper.is_file():
            tree = ast.parse(helper.read_text())
            if any(isinstance(node, ast.FunctionDef) and
                   node.name == 'validate_action_statistics_structure' for node in tree.body):
                candidates.add(root)
    if candidates:
        if len(candidates) != 1:
            raise ValueError('Ambiguous canonical typed-safety source roots on PYTHONPATH')
        return candidates.pop()
    fallback = Path(__file__).resolve().parents[2] / 'gigpo-e1-fresh200-typed-safety-20261006'
    helper = fallback / 'search_r1/baselines/action_prefix_forward.py'
    if not helper.is_file() or 'def validate_action_statistics_structure(' not in helper.read_text():
        raise ValueError('No exact typed-safety successor source root')
    return fallback.resolve()


class ImportsProvidedByFixture(ast.NodeTransformer):
    def visit_Import(self, node):
        return ast.copy_location(ast.Pass(), node)

    def visit_ImportFrom(self, node):
        return ast.copy_location(ast.Pass(), node)


def actual_definition(root, relative, name, namespace):
    source = (root / relative).read_text()
    node = next(value for value in ast.walk(ast.parse(source))
                if isinstance(value, (ast.FunctionDef, ast.ClassDef)) and value.name == name)
    node = copy.deepcopy(node)
    if isinstance(node, ast.FunctionDef):
        node.decorator_list = []
        node.returns = None
        for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
            argument.annotation = None
    node = ImportsProvidedByFixture().visit(node)
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, str(root / relative), 'exec'), namespace)
    return namespace[name]


def actual_loss_admission(root, namespace):
    """Preserve the actual isfinite/MAX call and next original backward/counters."""
    path = root / 'verl/workers/actor/dp_actor.py'
    source = path.read_text()
    function = next(node for node in ast.walk(ast.parse(source))
                    if isinstance(node, ast.FunctionDef) and node.name == 'update_policy')
    body = None
    for parent in ast.walk(function):
        statements = getattr(parent, 'body', None)
        if not isinstance(statements, list):
            continue
        for index, node in enumerate(statements):
            if (isinstance(node, ast.If) and isinstance(node.test, ast.Name)
                    and node.test.id == 'safety_enabled'
                    and 'NONFINITE_TRAINING_LOSS' in ast.get_source_segment(source, node)):
                body = copy.deepcopy(statements[index:index + 4])
                break
        if body is not None:
            break
    if body is None or len(body) != 4:
        raise ValueError('Actual completed loss admission/backward boundary not found')
    node = ast.parse('def admit(self,loss,log_prob,outer_identity,batch_idx,micro_index,updates,data,'
        'safety_enabled=True,scaled_joint=False,loss_scale=1.,audit=None,fresh_attempt=True):\n    pass\n').body[0]
    node.body = body
    node = ImportsProvidedByFixture().visit(node)
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace['admit']


class CommonCPUTransport:
    def __init__(self, rank, decisions, unknown_at=None, original=None):
        self.decisions = list(decisions)
        self.calls = []
        self.unknown_at = unknown_at
        self.original = original
        self.distributed = SimpleNamespace(is_initialized=lambda: True,
            get_rank=lambda: rank, all_reduce=self.reduce,
            ReduceOp=SimpleNamespace(MAX='MAX'))

    def __getattr__(self, name):
        return getattr(torch, name)

    def reduce(self, flag, op, group=None):
        self.calls.append(int(flag))
        if len(self.calls) == self.unknown_at:
            raise self.original
        if len(self.calls) <= len(self.decisions):
            flag.fill_(self.decisions[len(self.calls) - 1])


def producer_fixture(root, audit_root, *, rank=0, decisions=(1, 0),
                     local_io_failure=False, unknown_at=None, original=None, transport=None):
    transport = CommonCPUTransport(rank, decisions, unknown_at, original) if transport is None else transport
    diagnostics = []

    def diagnostic_save(state, path):
        diagnostics.append(copy.deepcopy(state))
        if local_io_failure:
            raise OSError('actual local refusal diagnostic write failed')
        return dict(path=str(path))

    os.environ['SEARCH_R1_BASELINE_AUDIT_DIR'] = str(audit_root)
    os.environ['SEARCH_R1_BASELINE_RUN_ID'] = 'CPU-actual-loss-producer-v11'
    os.environ['SEARCH_R1_BASELINE_COMMIT'] = 'f' * 40
    os.environ['SEARCH_R1_BASELINE_CONFIG_SHA256'] = 'a' * 64
    namespace = dict(torch=transport, os=os, json=json, Path=Path,
                     atomic_torch_save=diagnostic_save, cpu_snapshot=lambda value: value)
    rejected = actual_definition(root, 'search_r1/baselines/policy_safety.py',
                                 'PolicySafetyRejected', namespace)
    sync = actual_definition(root, 'verl/workers/actor/dp_actor.py', '_safety_sync_invalid', namespace)
    disposition = actual_definition(root, 'search_r1/baselines/partial_update.py',
                                    'rejection_disposition', namespace)
    actor = SimpleNamespace(actor_module=SimpleNamespace(process_group=None),
        _fresh200_attempt_id=1, _fresh200_attempt_adam_writes=2,
        _fresh200_attempt_gradient_micros=16)
    actor._safety_sync_invalid = MethodType(sync, actor)
    return actor, actual_loss_admission(root, namespace), rejected, disposition, transport, diagnostics


@pytest.fixture
def producer_environment(monkeypatch):
    # Register the producer's exact environment writes for automatic restoration.
    for key in ('SEARCH_R1_BASELINE_AUDIT_DIR', 'SEARCH_R1_BASELINE_RUN_ID',
                'SEARCH_R1_BASELINE_COMMIT', 'SEARCH_R1_BASELINE_CONFIG_SHA256'):
        monkeypatch.setenv(key, os.environ.get(key, ''))


def numeric_attempt_from_actual_record(record):
    """The refusal record is supplied unchanged by the actual source producer."""
    rank, pid = record['rank'], record['pid']
    receipt = dict(rank=rank, pid=pid, attempt_id=record['attempt_id'],
        counted_complete=True, opportunities_processed=20,
        actual_successful_adam_writes=19, skipped_opportunities=1, gradient_micro_batches=152,
        refusals=[dict(partial_update=dict(rejection=record))])
    coordinates = ([(mini, micro) for mini in range(2) for micro in range(8)] +
                   [(2, 0)] + [(mini, micro) for mini in range(3, 20) for micro in range(8)])
    micros = [empty_micro() for _ in coordinates]
    if record['local_invalid']:
        micros[16]['policy_loss'] = dict(count=1, finite=0, minimum=None, maximum=None)
    actor = dict(kind='actor_outer', outer_step=1, rank=rank, pid=pid,
        commit=record['commit'], config_sha256=record['config_sha256'], run_id=record['run_id'],
        attempt_id=record['attempt_id'], counted_complete=True, opportunities_processed=20,
        actual_gradient_micro_batches=152, rows=320, epochs=1, optimizer_steps=19,
        skipped_empty_minibatches=1, micro_stats=micros,
        grad_finite=19, grad_nonzero=0, grad_norms=[0.] * 19,
        fixed_unchanged=True, shard_layout_unchanged=True,
        row_denominators=[dict(mini_index=mini, local_rows=16, actual_denominator=16, world_size=8)
                          for mini in range(20)])
    rows = []
    for index, (mini, micro) in enumerate(coordinates, 1):
        parts = {name: dict(count=1, finite=1, minimum=0., maximum=0.)
                 for name in ('policy', 'entropy', 'kl', 'total')}
        if index == 17 and record['local_invalid']:
            parts['total'] = dict(count=1, finite=0, minimum=None, maximum=None)
        rows.append(dict(attempt_id=record['attempt_id'], rank=rank, pid=pid, micro_index=index,
            mini_index=mini, micro_index_in_mini=micro, commit=record['commit'],
            config_sha256=record['config_sha256'], run_id=record['run_id'],
            statistics=copy.deepcopy(micros[index - 1]),
            reference_minus_current=dict(count=0, finite=0), loss_parts=parts))
    return receipt, actor, rows


def emit_actual_loss(actor, admit, rejected, local_bad=True):
    loss = torch.tensor(float('nan') if local_bad else 0., requires_grad=True)
    with pytest.raises(rejected) as caught:
        admit(actor, loss, torch.zeros(1), dict(baseline_outer_step=1), 2, 0, 2, {})
    assert actor._fresh200_attempt_gradient_micros == 16 and loss.grad is None
    return caught.value.record


@pytest.mark.parametrize('rank,local_bad', [(0, True), (1, False)])
def test_actual_completed_loss_max_record_is_accepted_by_source_and_runtime(
        tmp_path, producer_environment, rank, local_bad):
    actor, admit, rejected, disposition, transport, diagnostics = producer_fixture(
        source_root(), tmp_path, rank=rank)
    record = emit_actual_loss(actor, admit, rejected, local_bad)
    assert transport.calls == [int(local_bad), 0]
    assert record['refusal_kind'] == 'known_numeric' and record['global_diagnostic_io_failed'] is False
    assert record['matched_guard'] == 'ORIGINAL_ALL_REDUCE_MAX_LOSS_REFUSAL'
    assert record['local_invalid'] is local_bad and record['actual_optimizer_writes'] == 2
    assert disposition(record) == 'skip_opportunity'
    assert len(diagnostics) == 1 and 'matched_guard' not in diagnostics[0]['record']
    receipt, actor_row, rows = numeric_attempt_from_actual_record(record)
    checked = validate_calibration_update(actor_row, record['commit'], record['config_sha256'],
        record['run_id'], rank, 1, receipt)
    assert checked['gradient_micro_batches'] == 152 and checked['observed_numeric_micro_batches'] == 153
    assert validate_attempt_numeric(rows, actor_row, receipt) == dict(
        actual_gradient_backward_micro_batches=152, observed_numeric_micro_batches=153,
        matched_refused_loss_observations=1)
    assert receipt['refusals'][0]['partial_update']['rejection'] is record


@pytest.mark.parametrize('rank,local_io_failure', [(0, True), (1, False)])
def test_actual_global_diagnostic_io_failure_cannot_authorize_runtime_loss(
        tmp_path, producer_environment, rank, local_io_failure):
    actor, admit, rejected, disposition, transport, _ = producer_fixture(source_root(), tmp_path,
        rank=rank, decisions=(1, 1), local_io_failure=local_io_failure)
    record = emit_actual_loss(actor, admit, rejected, local_bad=(rank == 0))
    assert transport.calls == [int(rank == 0), int(local_io_failure)]
    assert record['global_diagnostic_io_failed'] is True and 'matched_guard' not in record
    assert disposition(record) == 'fail_closed'
    receipt, actor_row, rows = numeric_attempt_from_actual_record(record)
    with pytest.raises(ValueError):
        validate_attempt_numeric(rows, actor_row, receipt)


@pytest.mark.parametrize('stage', ['loss_max', 'io_max'])
@pytest.mark.parametrize('error_type', [ValueError, RuntimeError, torch.cuda.OutOfMemoryError])
def test_unknown_original_max_error_remains_original_without_loss_authority(
        tmp_path, producer_environment, stage, error_type):
    original = error_type('unknown original CPU ' + stage)
    actor, admit, rejected, _, transport, diagnostics = producer_fixture(source_root(), tmp_path,
        unknown_at=1 if stage == 'loss_max' else 2, original=original)
    loss = torch.tensor(float('nan'), requires_grad=True)
    with pytest.raises(error_type) as caught:
        admit(actor, loss, torch.zeros(1), dict(baseline_outer_step=1), 2, 0, 2, {})
    assert caught.value is original and not isinstance(caught.value, rejected)
    assert len(transport.calls) == (1 if stage == 'loss_max' else 2)
    assert len(diagnostics) == (0 if stage == 'loss_max' else 1)
    assert all('matched_guard' not in state['record'] for state in diagnostics)
    assert actor._fresh200_attempt_gradient_micros == 16 and loss.grad is None


@pytest.mark.parametrize('kind,details', [('structural', 'NONFINITE_TRAINING_LOSS'),
    ('terminal_state', 'NONFINITE_TRAINING_LOSS'), ('known_numeric', 'UNKNOWN_ORIGINAL_ERROR')])
def test_actual_non_loss_provenance_cannot_be_promoted_by_a_loss_phase(
        tmp_path, producer_environment, kind, details):
    actor, _, rejected, disposition, _, _ = producer_fixture(source_root(), tmp_path,
        decisions=(1 if kind == 'known_numeric' else 2, 0))
    with pytest.raises(rejected) as caught:
        actor._safety_sync_invalid(True, 'cpu', dict(outer_step=1, mini_index=2, micro_index_in_mini=0),
            'gradient_micro0_loss', 2, details=details, refusal_kind=kind)
    record = caught.value.record
    assert record.get('matched_guard') != 'ORIGINAL_ALL_REDUCE_MAX_LOSS_REFUSAL'
    assert disposition(record) == 'fail_closed'
    receipt, actor_row, rows = numeric_attempt_from_actual_record(record)
    with pytest.raises(ValueError):
        validate_attempt_numeric(rows, actor_row, receipt)


def _gloo_loss_refusal_process(rank, rendezvous, root, audit_root, output, global_io_failure):
    """Future bounded CPU child: real Gloo MAX plus the actual producer body."""
    torch.set_num_threads(1)
    torch.distributed.init_process_group('gloo', init_method='file://' + rendezvous,
        rank=rank, world_size=2, timeout=timedelta(seconds=12))
    try:
        actor, admit, rejected, disposition, _, _ = producer_fixture(Path(root), Path(audit_root),
            rank=rank, local_io_failure=global_io_failure and rank == 0, transport=torch)
        loss = torch.tensor(float('nan') if rank == 0 else 0., requires_grad=True)
        try:
            admit(actor, loss, torch.zeros(1), dict(baseline_outer_step=1), 2, 0, 2, {})
        except rejected as refusal:
            record = refusal.record
            receipt, actor_row, rows = numeric_attempt_from_actual_record(record)
            audit_accepted = True
            try:
                validate_attempt_numeric(rows, actor_row, receipt)
            except ValueError:
                audit_accepted = False
            result = dict(rank=rank, record=record, disposition=disposition(record),
                runtime_audit_accepted=audit_accepted,
                actual_backward_micros=actor._fresh200_attempt_gradient_micros,
                loss_gradient_present=loss.grad is not None)
            Path(output).write_text(json.dumps(result, allow_nan=False, sort_keys=True))
        else:
            raise AssertionError('A real clean peer escaped the common bad-loss decision')
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize('global_io_failure', [False, True])
def test_real_two_rank_cpu_gloo_loss_and_io_decisions_are_common(tmp_path, global_io_failure):
    root = source_root()
    context = multiprocessing.get_context('spawn')
    rendezvous = str(tmp_path / 'gloo-rendezvous')
    processes = [context.Process(target=_gloo_loss_refusal_process,
        args=(rank, rendezvous, str(root), str(tmp_path / 'diagnostics'),
              str(tmp_path / f'rank-{rank}.json'), global_io_failure)) for rank in range(2)]
    deadline = time.monotonic() + 30.
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(max(0., deadline - time.monotonic()))
        assert not any(process.is_alive() for process in processes), 'Bounded Gloo child timed out'
        assert [process.exitcode for process in processes] == [0, 0]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            if process.pid is not None:
                process.join(3.)
    values = [json.loads((tmp_path / f'rank-{rank}.json').read_text()) for rank in range(2)]
    assert [value['record']['local_invalid'] for value in values] == [True, False]
    assert {value['record']['global_diagnostic_io_failed'] for value in values} == {global_io_failure}
    assert {value['disposition'] for value in values} == {
        'fail_closed' if global_io_failure else 'skip_opportunity'}
    assert {value['runtime_audit_accepted'] for value in values} == {not global_io_failure}
    assert {value['actual_backward_micros'] for value in values} == {16}
    assert not any(value['loss_gradient_present'] for value in values)
    if global_io_failure:
        assert all('matched_guard' not in value['record'] for value in values)
    else:
        assert {value['record']['matched_guard'] for value in values} == {
            'ORIGINAL_ALL_REDUCE_MAX_LOSS_REFUSAL'}
