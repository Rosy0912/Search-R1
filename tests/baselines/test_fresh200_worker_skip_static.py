"""Unexecuted CPU fixtures for the actual worker attempt method.

AST extraction removes only function-local imports so the original method body
can use explicitly substituted CPU transport/snapshot surfaces. It does not
import FSDP workers, initialize Ray/distributed/CUDA, or claim eight-GPU proof.
"""
import ast
import copy
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from search_r1.baselines.partial_update import (_adam_steps,
    materialize_continuation_state, rejection_disposition)
from search_r1.baselines.fresh200 import record_worker_attempt


ROOT = Path(__file__).resolve().parents[2]


class KnownRefusal(Exception):
    def __init__(self, record):
        super().__init__('synthetic matched refusal')
        self.record = record


def source_function(relative, name, namespace):
    tree = ast.parse((ROOT / relative).read_text())
    node = next(value for value in ast.walk(tree)
                if isinstance(value, ast.FunctionDef) and value.name == name)
    node = copy.deepcopy(node)
    node.decorator_list = []

    class ImportsProvidedByFixture(ast.NodeTransformer):
        def visit_Import(self, node): return ast.copy_location(ast.Pass(), node)
        def visit_ImportFrom(self, node): return ast.copy_location(ast.Pass(), node)

    node = ImportsProvidedByFixture().visit(node)
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, str(ROOT / relative), 'exec'), namespace)
    return namespace[name]


def worker_fixture(tmp_path, events):
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    stages, calls, saved = [], [], {}
    scheduler = SimpleNamespace(last_epoch=0, writes=0)
    def advance_scheduler():
        scheduler.last_epoch += 1
        scheduler.writes += 1
    scheduler.step = advance_scheduler

    def identity():
        slots = _adam_steps({'optimizer': optimizer.state_dict()})
        return dict(model=model.weight.detach().clone().tolist(), adam=slots,
                    moments=repr(optimizer.state_dict()), scheduler=scheduler.last_epoch,
                    rng=torch.get_rng_state().tolist(), pid=os.getpid())

    def metadata():
        return dict(fresh200_worker_ledger=dict(schema='FRESH200_WORKER_ACTUAL_ATTEMPTS_V1',
                    attempts=0, completed_outer_updates=0, actual_adam_total=0, history=[]))

    def folder(outer, attempt, refusal=None):
        path = tmp_path / str(attempt) / str(refusal)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def snapshot(state, path):
        torch.save(state, path)
        saved[str(path)] = copy.deepcopy(state)
        return dict(path=str(path), bytes=path.stat().st_size,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    def state_snapshot(*args, **kwargs):
        return dict(schema=1, logical_step=0, pause_identity=identity(),
            model_shards={'weight': model.weight.detach().clone()},
            optimizer=copy.deepcopy(optimizer.state_dict()), scheduler=dict(last_epoch=0),
            rng={'torch_cpu': torch.get_rng_state().clone()}, complete_internal=True)

    def preserve(worker, data, refusal, index):
        state = state_snapshot()
        state['pending_gradients'] = {'weight': None if model.weight.grad is None
                                     else model.weight.grad.detach().clone()}
        part = dict(actual_adam_total=_adam_steps(state), rejection=refusal.record,
                    attempt_id=1, refusal_index=index)
        state['partial_update'] = part
        receipt = snapshot(state, folder(1, 1, index) / 'rank_0.pt')
        return dict(partial_update=part, state_receipt=receipt)

    def matched(phase, error):
        stages.append(phase)
        if error is not None: raise RuntimeError(str(error))

    cpu_transport = SimpleNamespace(Tensor=torch.Tensor, int64=torch.int64,
        isfinite=torch.isfinite, equal=torch.equal,
        tensor=lambda values, dtype, device: torch.tensor(values, dtype=dtype),
        distributed=SimpleNamespace(all_reduce=lambda value, op: None,
            ReduceOp=SimpleNamespace(MIN='min', MAX='max')))
    namespace = dict(torch=cpu_transport, os=os, PolicySafetyRejected=KnownRefusal,
        _checkpoint_matched_stage=matched, _adam_steps=_adam_steps,
        rejection_disposition=rejection_disposition, preserve_worker=preserve,
        worker_training_state=state_snapshot, _folder=folder,
        rng_state=lambda: {'torch_cpu': torch.get_rng_state().clone()},
        atomic_torch_save=snapshot, atomic_json_save=lambda state, path: None)
    continuation = source_function('search_r1/baselines/partial_update.py',
                                  'continue_worker_after_rejection', namespace)
    namespace['continue_worker_after_rejection'] = continuation
    worker_method = source_function('verl/workers/fsdp_workers.py',
                                    '_fresh200_update_attempt', namespace)
    worker = SimpleNamespace(actor_module_fsdp=model, actor_optimizer=optimizer,
        actor_lr_scheduler=scheduler, rank=0, world_size=8,
        rollout_sharding_manager=None, checkpoint_pause_identity=identity,
        checkpoint_runtime_metadata=metadata)
    actor = SimpleNamespace()

    def update(data):
        for mini in range(data.meta_info['fresh200_processed_opportunities'], 20):
            calls.append(mini)
            event = events.get(mini)
            if event == 'unknown': raise OSError('original forward/collective failure')
            optimizer.zero_grad(set_to_none=True)
            model.weight.grad = torch.ones_like(model.weight)
            actor._fresh200_attempt_gradient_micros += 8
            if event != 'pre_numeric':
                optimizer.step()
                actor._fresh200_attempt_adam_writes += 1
            if event == 'post_nonfinite':
                with torch.no_grad(): model.weight.fill_(float('nan'))
            if event is not None:
                record = dict(outer_step=1, mini_index=mini, phase='after' if event in ('kl','tail')
                    else 'global_gradient_before_adam' if event == 'pre_numeric' else 'after',
                    actual_optimizer_writes=actor._fresh200_attempt_adam_writes)
                if event in ('kl','tail'):
                    record['rejection_reason'] = ('OLD_POLICY_TOKEN_KL' if event == 'kl'
                                                  else 'NEGATIVE_JOINT_ACTION_TAIL_COUNT')
                raise KnownRefusal(record)
        return {'actor/opportunities_processed':20}

    actor.update_policy = update
    worker.actor = actor
    data = SimpleNamespace(meta_info=dict(fresh200_attempt_id=1,
        fresh200_completed_before=0, fresh200_adam_total_before=0, baseline_outer_step=1))
    return worker, data, worker_method, calls, stages, saved


@pytest.mark.parametrize('events,writes,processed,counted,abandoned', [
    ({0:'pre_numeric', 2:'post_numeric'},19,20,True,False),
    ({mini:'pre_numeric' for mini in range(20)},0,20,False,False),
    ({2:'kl'},3,3,False,True), ({2:'tail'},3,3,False,True)])
def test_actual_worker_method_retains_writes_and_resumes_only_next_opportunity(
        tmp_path, events, writes, processed, counted, abandoned):
    worker,data,method,calls,stages,saved=worker_fixture(tmp_path,events)
    metrics,receipt=method(worker,data)
    assert calls == list(range(processed))  # No rejected mini is replayed.
    assert receipt['actual_successful_adam_writes']==writes
    assert receipt['actual_adam_total']==writes
    assert receipt['opportunities_processed']==processed
    assert receipt['counted_complete'] is counted and receipt['abandoned'] is abandoned
    assert receipt['skipped_opportunities']==processed-writes
    assert worker.actor_lr_scheduler.last_epoch==0  # update_actor owns the single scheduler advance.
    assert all(parameter.grad is None for parameter in worker.actor_module_fsdp.parameters())
    assert _adam_steps({'optimizer':worker.actor_optimizer.state_dict()})==writes
    for refusal in receipt['refusals']:
        saved_reference=refusal['continuation_state']
        restored=materialize_continuation_state(saved_reference['path'],saved_reference['sha256'])
        assert 'pending_gradients' not in restored
        assert restored['continuation_boundary']['automatic_rollback'] is False
        assert _adam_steps(restored)==refusal['partial_update']['actual_adam_total']
    record_worker_attempt(worker,receipt)
    assert worker.actor_lr_scheduler.writes==int(counted)
    assert worker.actor_lr_scheduler.last_epoch==int(counted)
    assert worker.fresh200_worker_ledger['actual_adam_total']==writes
    assert worker.fresh200_worker_ledger['completed_outer_updates']==int(counted)


def test_actual_postclear_materializer_reloads_exact_tensor_payload_and_rejects_tamper(tmp_path):
    worker,data,method,_,_,saved=worker_fixture(tmp_path,{0:'post_numeric'})
    _,receipt=method(worker,data)
    refusal=receipt['refusals'][0]
    post=refusal['continuation_state']
    pre=refusal['state_receipt']
    restored=materialize_continuation_state(post['path'],post['sha256'])
    original=saved[pre['path']]
    assert torch.equal(restored['model_shards']['weight'],original['model_shards']['weight'])
    assert _adam_steps(restored)==_adam_steps(original)==1
    for key in ('step','exp_avg','exp_avg_sq'):
        assert torch.equal(restored['optimizer']['state'][0][key],original['optimizer']['state'][0][key])
    assert torch.equal(restored['rng']['torch_cpu'],saved[post['path']]['rng']['torch_cpu'])
    assert restored['runtime_metadata']==saved[post['path']]['runtime_metadata']
    assert original['pending_gradients']['weight'] is not None
    assert 'pending_gradients' not in restored
    Path(pre['path']).write_bytes(Path(pre['path']).read_bytes()+b'changed')
    with pytest.raises(ValueError,match='payload bytes changed'):
        materialize_continuation_state(post['path'],post['sha256'])
    Path(post['path']).write_bytes(Path(post['path']).read_bytes()+b'changed')
    with pytest.raises(ValueError,match='snapshot bytes changed'):
        materialize_continuation_state(post['path'],post['sha256'])


def test_unknown_original_failure_escapes_without_recovery_collective(tmp_path):
    worker,data,method,calls,stages,saved=worker_fixture(tmp_path,{2:'unknown'})
    with pytest.raises(OSError,match='original forward/collective'): method(worker,data)
    assert calls==[0,1,2] and not saved
    assert stages==['attempt/entry_actual_counters']
    assert _adam_steps({'optimizer':worker.actor_optimizer.state_dict()})==2


def test_written_nonfinite_state_fails_closed_instead_of_fresh_continuation(tmp_path):
    worker,data,method,calls,stages,saved=worker_fixture(tmp_path,{0:'post_nonfinite'})
    with pytest.raises(RuntimeError,match='nonfinite'): method(worker,data)
    assert calls==[0]
    assert _adam_steps({'optimizer':worker.actor_optimizer.state_dict()})==1
    assert list(saved.values())[0]['partial_update']['actual_adam_total']==1
    assert 'safe_skip/save_postclear_continuation_state' not in stages


def test_unknown_typed_phase_cannot_authorize_safe_skip():
    assert rejection_disposition(dict(phase='after_unknown',mini_index=0))=='fail_closed'
    assert rejection_disposition(dict(phase='before_probe_entry',mini_index=0))=='fail_closed'
    assert rejection_disposition(dict(phase='after_unknown',mini_index=0,
        rejection_reason='OLD_POLICY_TOKEN_KL'))=='fail_closed'
