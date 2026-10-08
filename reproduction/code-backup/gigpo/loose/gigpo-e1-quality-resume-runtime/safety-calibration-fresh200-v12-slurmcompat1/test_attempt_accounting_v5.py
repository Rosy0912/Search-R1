"""UNEXECUTED CPU fixtures: actual sampler/concat/ledger and sparse diagnostics.

Synthetic eight return rows are CPU controls, never real distributed evidence.
No fixture starts Ray, CUDA, a model, a subprocess or an external service.
"""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from verl import DataProto
from search_r1.baselines.recovery import RecoverableSampler
from search_r1.baselines.fresh200 import (begin_attempt, consume_attempt, initialize_attempt_ledger,
    record_epoch_end, record_worker_attempt)
from attempt_accounting_v5 import actual_attempt_totals, validate_actual_worker_ledgers
from calibration_failure_v5 import preserve_attempt_failure
from calibration_failure_v2 import snapshot_resolved_rank_outputs
from worker_rpc_boundary_v4 import track_worker_group_rpcs
from test_worker_rpc_boundary_v4 import make_actual_rpc_fixture


def ledger_fixture(attempts):
    sampler = RecoverableSampler(4097, seed=271)
    loader = torch.utils.data.DataLoader(list(range(4097)), batch_size=512,
        sampler=sampler, num_workers=0, drop_last=True)
    iterator = iter(loader)
    trainer = SimpleNamespace(recovery_sampler=sampler, actual_outer_updates=0,
        _calibration_training_iterator=iterator, validation_trace_serial=0)
    for processed, writes, abandoned, micros in attempts:
        ids = next(iterator)
        batch = DataProto.from_dict({'question_ids': ids.view(512, 1)})
        attempt_id = begin_attempt(trainer, batch)
        counted = processed == 20 and writes > 0 and not abandoned
        rows = [dict(rank=rank, pid=9000 + rank, attempt_id=attempt_id,
            completed_before=batch.meta_info['fresh200_completed_before'],
            opportunities_processed=processed, actual_successful_adam_writes=writes,
            skipped_opportunities=processed - writes, gradient_micro_batches=micros,
            abandoned=abandoned, counted_complete=counted,
            actual_adam_total=batch.meta_info['fresh200_adam_total_before'] + writes,
            scheduler_step=batch.meta_info['fresh200_completed_before'] + int(counted),
            safe_to_continue=True, refusals=[]) for rank in range(8)]
        # Real DataProto.concat is required: meta_info alone would drop seven ranks.
        output = DataProto.concat([DataProto(non_tensor_batch={
            'fresh200_attempt_receipts': np.asarray([row], dtype=object)}) for row in rows])
        consume_attempt(trainer, output)
    return trainer, initialize_attempt_ledger(trainer)


@pytest.mark.parametrize('attempts,total,completed_writes,abandoned_writes', [
    ([(20,0,False,0), (7,3,True,24), (20,1,False,8), (20,19,False,152)], 23,20,3),
    ([(7,6,True,48), (20,20,False,160), (20,20,False,160)], 46,40,6)])
def test_actual_writes_can_be_below_or_above_forty_without_stitching_partial_outers(
        attempts, total, completed_writes, abandoned_writes):
    trainer, ledger = ledger_fixture(attempts)
    observed = actual_attempt_totals(ledger)
    assert ledger['completed_outer_updates'] == 2
    assert observed['diagnostic_counted_optimizer_opportunities'] == 40
    assert observed['actual_optimizer_steps_per_rank'] == total
    assert observed['successful_adam_writes_completed_outers'] == completed_writes
    assert observed['successful_adam_writes_abandoned_attempts'] == abandoned_writes
    assert observed['actual_gradient_micro_per_rank'] == sum(value[3] for value in attempts)
    assert trainer.recovery_sampler.cursor == len(attempts) * 512
    assert [row['scheduler_step'] for row in ledger['history']][-1] == 2
    assert len({row['attempt_id'] for row in ledger['history']}) == len(attempts)


def worker_states(ledger):
    states = []
    for rank in range(8):
        history = [copy.deepcopy(next(row for row in attempt['rank_receipts'] if row['rank'] == rank))
                   for attempt in ledger['history']]
        worker = dict(schema='FRESH200_WORKER_ACTUAL_ATTEMPTS_V1', attempts=ledger['attempts'],
            completed_outer_updates=ledger['completed_outer_updates'], actual_adam_total=ledger['actual_adam_total'],
            history=history)
        states.append(dict(identity=dict(rank=rank, pid=9000 + rank, scheduler_step=2),
            metadata=dict(source_runtime_metadata=dict(fresh200_worker_ledger=worker))))
    return states


@pytest.mark.parametrize('changed', ['pid', 'write_total', 'history', 'scheduler'])
def test_one_rank_worker_reload_disagreement_is_not_accepted(changed):
    _, ledger = ledger_fixture([(7,6,True,48), (20,20,False,160), (20,20,False,160)])
    states = worker_states(ledger)
    assert len(validate_actual_worker_ledgers(states, ledger)) == 8
    if changed == 'pid':
        states[7]['identity']['pid'] += 1
    elif changed == 'write_total':
        states[7]['metadata']['source_runtime_metadata']['fresh200_worker_ledger']['actual_adam_total'] += 1
    elif changed == 'history':
        states[7]['metadata']['source_runtime_metadata']['fresh200_worker_ledger']['history'][0]['safe_to_continue'] = False
    else:
        states[7]['identity']['scheduler_step'] = 3
    with pytest.raises(ValueError):
        validate_actual_worker_ledgers(states, ledger)


def test_actual_update_rank_receipts_survive_local_collect_failure_without_new_rpc(monkeypatch):
    import verl.protocol as protocol
    rank_outputs = [DataProto(non_tensor_batch={'fresh200_attempt_receipts':
        np.asarray([dict(rank=rank, actual_adam_total=6)], dtype=object)}) for rank in range(8)]
    rank_outputs[3] = DataProto(non_tensor_batch={'injected_wrong_key': np.asarray([3], dtype=object)})
    trainer, calls, output, schema, events = make_actual_rpc_fixture(
        failure_method='update_actor', rank_outputs=rank_outputs)
    progress = dict(outer=1, attempt_id=2, actual_adam_total_before=6,
        first_adam_possible=False, workers_safe_to_snapshot=True)
    original = trainer.actor_rollout_wg.update_actor
    original_dict_collect = protocol.list_of_dict_to_dict_of_list
    original_errors = []
    def observe_original_dict_collect(*args, **kwargs):
        try:
            return original_dict_collect(*args, **kwargs)
        except AssertionError as error:
            original_errors.append(error)
            raise  # Retain the genuine current source assertion object/message.
    with monkeypatch.context() as patch:
        patch.setattr(protocol, 'list_of_dict_to_dict_of_list', observe_original_dict_collect)
        with pytest.raises(AssertionError) as caught:
            with track_worker_group_rpcs(((trainer.actor_rollout_wg, 'actor_rollout', ('update_actor',)),),
                                        progress, _schema=schema):
                trainer.actor_rollout_wg.update_actor(output)
    assert len(original_errors) == 1 and caught.value is original_errors[0]
    assert caught.value.args == () and str(caught.value) == ''
    assert protocol.list_of_dict_to_dict_of_list is original_dict_collect
    assert progress['first_adam_possible'] and progress['workers_safe_to_snapshot']
    assert calls == ['update_actor'] and trainer.actor_rollout_wg.update_actor is original
    resolved = progress['resolved_worker_rank_outputs']['actor_rollout/update_actor']['outputs']
    assert len(resolved) == 8 and all(value is expected for value, expected in zip(resolved, rank_outputs))
    saved = snapshot_resolved_rank_outputs(progress, lambda value: value)
    ranks = saved['actor_rollout/update_actor']['ranks']
    assert len(ranks) == 8 and all(row['saved'] for row in ranks)
    assert ranks[0]['output']['tensors'] is None
    assert ranks[0]['output']['non_tensors']['fresh200_attempt_receipts'][0]['actual_adam_total'] == 6
    assert ranks[3]['output']['non_tensors']['injected_wrong_key'].tolist() == [3]
    assert [event[0] for event in events].count('await') == 1
    remote_events = [event for event in events if event[0] == 'remote']
    assert [(event[1], event[2]) for event in remote_events] == [('actor_rollout_update_actor', rank) for rank in range(8)]


@pytest.mark.parametrize('stage', ['execute', 'await'])
def test_unknown_phase_after_prior_partial_attempt_stays_unsafe_and_only_saves_driver(
        tmp_path, monkeypatch, stage):
    from search_r1.baselines import recovery
    failure = RuntimeError('CPU fixture unknown actual phase RPC error')
    trainer, calls, output, schema, _ = make_actual_rpc_fixture(
        failure=failure, failure_stage=stage, failure_method='calibration_phase')
    trainer.actual_outer_updates = 0
    trainer.fresh200_attempt_ledger = dict(actual_adam_total=6, active_attempt=dict(attempt_id=2))
    progress = dict(outer=1, attempt_id=2, actual_adam_total_before=6,
        first_adam_possible=False, workers_safe_to_snapshot=True)
    def save(value, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(value, path)
        return dict(path=str(path))
    monkeypatch.setattr(recovery, 'atomic_torch_save', save)
    monkeypatch.setattr(recovery, 'rng_state', lambda: dict(actual_rng_progress=123))
    def json_save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    with pytest.raises(RuntimeError) as caught:
        with track_worker_group_rpcs(((trainer.actor_rollout_wg, 'actor_rollout', ('calibration_phase',)),),
                                    progress, _schema=schema):
            trainer.actor_rollout_wg.calibration_phase('train2')
    assert caught.value is failure and not progress['workers_safe_to_snapshot']
    report = preserve_attempt_failure(trainer, tmp_path, progress, failure,
        driver_snapshot=lambda *args: dict(cursor=512), json_save=json_save)
    assert calls == ['calibration_phase'] and not report['worker_recovery_rpc_dispatched']
    saved = torch.load(report['driver']['path'], weights_only=False)
    assert saved['attempt_ledger']['actual_adam_total'] == 6
    assert saved['rng']['actual_rng_progress'] == 123
    assert not report['automatic_resume_allowed'] and not report['automatic_rollback']


def test_actual_worker_completion_seam_steps_real_scheduler_only_for_countable_outer():
    parameter = torch.nn.Parameter(torch.tensor([1.]))
    optimizer = torch.optim.AdamW([parameter], lr=1e-6, weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: float(step) / 57.)
    worker = SimpleNamespace(actor_lr_scheduler=scheduler)
    total, completed = 0, 0
    for attempt_id, (processed, writes, abandoned) in enumerate([(7,6,True), (20,1,False), (20,19,False)], 1):
        lr = optimizer.param_groups[0]['lr']
        assert lr == 1e-6 * (float(completed) / 57.)
        for _ in range(writes):
            optimizer.zero_grad(set_to_none=True)
            parameter.grad = torch.ones_like(parameter)
            optimizer.step()  # Genuine CPU AdamW writes, not an inferred20×outer count.
        total += writes
        assert int(optimizer.state[parameter]['step'].item()) == total
        counted = processed == 20 and writes > 0 and not abandoned
        receipt = dict(rank=0, pid=9000, attempt_id=attempt_id, completed_before=completed,
            opportunities_processed=processed, actual_successful_adam_writes=writes,
            skipped_opportunities=processed - writes, gradient_micro_batches=writes * 8,
            abandoned=abandoned, counted_complete=counted, actual_adam_total=total,
            scheduler_step=completed, safe_to_continue=True, refusals=[])
        returned = record_worker_attempt(worker, receipt)
        completed += int(counted)
        assert returned is receipt and scheduler.last_epoch == completed
        assert worker.fresh200_worker_ledger['actual_adam_total'] == total
    assert total == 26 and completed == 2
    assert [row['scheduler_step'] for row in worker.fresh200_worker_ledger['history']] == [0,1,2]
    assert [row['actual_successful_adam_writes'] for row in worker.fresh200_worker_ledger['history']] == [6,1,19]


def test_actual_drop_last_tail_and_new_epoch_are_distinct_from_completed_outer_count():
    sampler = RecoverableSampler(1025, seed=271)
    loader = torch.utils.data.DataLoader(list(range(1025)), batch_size=512,
        sampler=sampler, num_workers=0, drop_last=True)
    iterator = iter(loader)
    trainer = SimpleNamespace(recovery_sampler=sampler, actual_outer_updates=0,
        _calibration_training_iterator=iterator)
    batches = []
    for attempt_id in (1, 2):
        ids = next(iterator)
        batches.append(ids.tolist())
        batch = DataProto.from_dict({'question_ids': ids.view(512, 1)})
        assert begin_attempt(trainer, batch) == attempt_id
        rows = [dict(rank=rank, pid=9000+rank, attempt_id=attempt_id, completed_before=0,
            opportunities_processed=20, actual_successful_adam_writes=0, skipped_opportunities=20,
            gradient_micro_batches=0, abandoned=False, counted_complete=False,
            actual_adam_total=0, scheduler_step=0, safe_to_continue=True, refusals=[]) for rank in range(8)]
        consume_attempt(trainer, DataProto.concat([DataProto(non_tensor_batch={
            'fresh200_attempt_receipts': np.asarray([row], dtype=object)}) for row in rows]))
    with pytest.raises(StopIteration):
        next(iterator)
    record_epoch_end(trainer)
    trainer._calibration_training_iterator = iter(loader)
    ids = next(trainer._calibration_training_iterator)
    batch = DataProto.from_dict({'question_ids': ids.view(512, 1)})
    assert begin_attempt(trainer, batch) == 3
    ledger = trainer.fresh200_attempt_ledger
    assert ledger['completed_outer_updates'] == 0 and ledger['consumed_training_questions'] == 1536
    assert ledger['epoch_ends'][0]['dropped_tail_rows'] == 1
    assert ledger['active_attempt']['sampler_position']['epoch'] == 1
    assert ledger['active_attempt']['sampler_position']['cursor'] == 512
    assert ledger['active_attempt']['sampler_position']['permutation_sha256'] != ledger['history'][0]['sampler_position']['permutation_sha256']
    assert len(set(batches[0]) & set(batches[1])) == 0
