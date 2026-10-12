"""Unexecuted CPU controls for human-approved opportunities versus Adam writes.

These use the actual sampler, DataLoader, DataProto concatenation and driver
ledger seams. Eight returned rows are synthetic CPU fixtures, never evidence
of eight GPU ranks. No fixture initializes CUDA/distributed/Ray or downloads.
Run only after a new context-free review approves these exact frozen bytes.
"""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl import DataProto
from search_r1.baselines.recovery import RecoverableSampler
from search_r1.baselines.fresh200 import (begin_attempt, consume_attempt,
    driver_flow_state, initialize_attempt_ledger, record_epoch_end,
    restore_driver_flow, restore_attempt_ledger, validate_attempt_ledger)
from search_r1.baselines.partial_update import rejection_disposition
from tests.baselines.test_fresh200_worker_skip_static import KnownRefusal, actual_worker_rejection


def training_fixture(size=4097):
    sampler = RecoverableSampler(size, seed=271)
    loader = torch.utils.data.DataLoader(list(range(size)), batch_size=512,
        sampler=sampler, num_workers=0, drop_last=True)
    iterator = iter(loader)
    trainer = SimpleNamespace(recovery_sampler=sampler,
        fresh200_active_train_iterator=iterator, actual_outer_updates=0,
        validation_trace_serial=0)
    return trainer, loader, iterator


def fetched_batch(trainer, iterator):
    ids = next(iterator)
    batch = DataProto.from_dict({'question_ids': ids.view(512, 1)})
    begin_attempt(trainer, batch)
    return batch


def returned_receipts(batch, processed, writes, *, abandoned=False,
                      gradient_micro_batches=None):
    counted = processed == 20 and writes > 0 and not abandoned
    if gradient_micro_batches is None:
        gradient_micro_batches = processed * 8
    rows = []
    for rank in range(8):
        value = dict(rank=rank, pid=9000+rank,
            attempt_id=batch.meta_info['fresh200_attempt_id'],
            completed_before=batch.meta_info['fresh200_completed_before'],
            opportunities_processed=processed, actual_successful_adam_writes=writes,
            skipped_opportunities=processed-writes,
            gradient_micro_batches=gradient_micro_batches, abandoned=abandoned,
            counted_complete=counted,
            actual_adam_total=batch.meta_info['fresh200_adam_total_before']+writes,
            scheduler_step=batch.meta_info['fresh200_completed_before']+int(counted),
            safe_to_continue=True)
        rows.append(DataProto(non_tensor_batch={
            'fresh200_attempt_receipts': np.array([value], dtype=object)}))
    # This is the real return concatenator. Meta_info alone would retain rank0
    # and therefore cannot prove all-rank decision/counter agreement.
    return DataProto.concat(rows)


@pytest.mark.parametrize('processed,writes,abandoned,counted', [
    (20,20,False,True), (20,1,False,True), (20,19,False,True),
    (20,0,False,False), (7,3,True,False), (20,19,True,False),
    (0,0,True,False)])
def test_complete_count_requires_all_opportunities_and_actual_write(
        processed,writes,abandoned,counted):
    trainer,_,iterator = training_fixture()
    batch=fetched_batch(trainer,iterator)
    output=returned_receipts(batch,processed,writes,abandoned=abandoned)
    assert len(output.non_tensor_batch['fresh200_attempt_receipts']) == 8
    result=consume_attempt(trainer,output)
    ledger=result['ledger']
    assert result['counted_complete'] is counted
    assert trainer.actual_outer_updates == int(counted)
    assert ledger['actual_adam_total'] == writes
    assert ledger['consumed_training_questions'] == 512
    assert ledger['history'][0]['scheduler_step'] == int(counted)


def test_abandoned_partial_writes_remain_real_without_inflating_complete_count():
    trainer,_,iterator=training_fixture()
    first_ids=[]
    for processed,writes,abandoned in [(20,0,False),(7,3,True),
                                       (20,1,False),(20,20,False)]:
        batch=fetched_batch(trainer,iterator)
        first_ids.extend(batch.batch['question_ids'].flatten().tolist())
        consume_attempt(trainer,returned_receipts(batch,processed,writes,
            abandoned=abandoned))
    ledger=initialize_attempt_ledger(trainer)
    assert trainer.actual_outer_updates == 2
    assert ledger['attempts'] == 4 and ledger['skipped_attempts'] == 2
    assert ledger['actual_adam_total'] == 24
    assert ledger['successful_adam_writes_completed_outers'] == 21
    assert ledger['successful_adam_writes_abandoned_attempts'] == 3
    assert ledger['consumed_training_questions'] == 2048
    assert trainer.recovery_sampler.cursor == 2048
    assert len(set(first_ids)) == 2048
    assert [row['scheduler_step'] for row in ledger['history']] == [0,0,1,2]
    assert [row['attempt_id'] for row in ledger['history']] == [1,2,3,4]
    assert ledger['actual_adam_total'] != 20*trainer.actual_outer_updates


@pytest.mark.parametrize('key,wrong', [
    ('rank',0), ('pid',9000), ('actual_successful_adam_writes',19),
    ('opportunities_processed',19), ('scheduler_step',0),
    ('attempt_id',99), ('safe_to_continue',False), ('abandoned',True),
    ('counted_complete',False), ('gradient_micro_batches',159)])
def test_one_rank_inconsistent_receipt_never_commits_driver_counts(key,wrong):
    trainer,_,iterator=training_fixture()
    batch=fetched_batch(trainer,iterator)
    output=returned_receipts(batch,20,20)
    saved=copy.deepcopy(initialize_attempt_ledger(trainer))
    output.non_tensor_batch['fresh200_attempt_receipts'][7][key]=wrong
    with pytest.raises(ValueError): consume_attempt(trainer,output)
    assert trainer.fresh200_attempt_ledger == saved
    assert trainer.actual_outer_updates == 0


@pytest.mark.parametrize('key,wrong', [
    ('scheduler_step',1), ('counted_complete',True), ('actual_adam_total',1),
    ('opportunities_processed',True), ('gradient_micro_batches',False),
    ('safe_to_continue',False)])
def test_all_rank_false_complete_or_scheduler_advance_is_rejected(key,wrong):
    trainer,_,iterator=training_fixture()
    batch=fetched_batch(trainer,iterator)
    output=returned_receipts(batch,20,0)
    for row in output.non_tensor_batch['fresh200_attempt_receipts']:
        row[key]=wrong
    with pytest.raises(ValueError): consume_attempt(trainer,output)
    assert trainer.actual_outer_updates == 0


def test_missing_rank_and_duplicate_return_cannot_be_reused():
    trainer,_,iterator=training_fixture()
    batch=fetched_batch(trainer,iterator)
    output=returned_receipts(batch,20,1)
    missing=DataProto(non_tensor_batch={'fresh200_attempt_receipts':
        output.non_tensor_batch['fresh200_attempt_receipts'][:7].copy()})
    with pytest.raises(ValueError): consume_attempt(trainer,missing)
    consume_attempt(trainer,output)
    saved=copy.deepcopy(trainer.fresh200_attempt_ledger)
    with pytest.raises(ValueError): consume_attempt(trainer,output)
    assert trainer.fresh200_attempt_ledger == saved


@pytest.mark.parametrize('key', ['actual_adam_total','completed_outer_updates',
    'successful_adam_writes_completed_outers',
    'successful_adam_writes_abandoned_attempts','gradient_micro_batches',
    'skipped_opportunities','consumed_training_questions','attempts'])
def test_saved_aggregate_corruption_cannot_be_accepted_as_resume_state(key):
    trainer,_,iterator=training_fixture()
    batch=fetched_batch(trainer,iterator)
    consume_attempt(trainer,returned_receipts(batch,20,1))
    saved=copy.deepcopy(trainer.fresh200_attempt_ledger)
    saved[key]+=1
    with pytest.raises(ValueError): validate_attempt_ledger(saved)
    with pytest.raises(ValueError): restore_attempt_ledger(trainer,saved)


def test_same_live_iterator_and_rng_survive_partial_write_ledger_roundtrip():
    trainer,_,iterator=training_fixture()
    batch=fetched_batch(trainer,iterator)
    consume_attempt(trainer,returned_receipts(batch,7,3,abandoned=True))
    saved=driver_flow_state(trainer)
    rng=torch.get_rng_state().clone()
    identity=id(iterator)
    receipt=restore_driver_flow(trainer,copy.deepcopy(saved))
    assert receipt['same_live_iterator'] and id(iterator)==identity
    assert torch.equal(torch.get_rng_state(),rng)
    assert trainer.actual_outer_updates==0
    assert trainer.fresh200_attempt_ledger['actual_adam_total']==3
    assert trainer.recovery_sampler.cursor==512
    next_batch=fetched_batch(trainer,iterator)
    assert next_batch.meta_info['fresh200_attempt_id']==2
    assert next_batch.meta_info['baseline_outer_step']==1
    assert next_batch.meta_info['fresh200_adam_total_before']==3


def test_drop_last_epoch_tail_is_recorded_separately_from_complete_iterations():
    trainer,loader,iterator=training_fixture(size=1025)
    for _ in range(2):
        batch=fetched_batch(trainer,iterator)
        consume_attempt(trainer,returned_receipts(batch,20,0))
    with pytest.raises(StopIteration): next(iterator)
    record_epoch_end(trainer)
    trainer.fresh200_active_train_iterator=iter(loader)
    batch=fetched_batch(trainer,trainer.fresh200_active_train_iterator)
    consume_attempt(trainer,returned_receipts(batch,20,1))
    ledger=trainer.fresh200_attempt_ledger
    assert trainer.actual_outer_updates==1
    assert ledger['consumed_training_questions']==1536
    assert ledger['epoch_ends'][0]['dropped_tail_rows']==1
    assert trainer.recovery_sampler.epoch==1
    assert trainer.recovery_sampler.cursor==512


@pytest.mark.parametrize('reason', ['OLD_POLICY_TOKEN_KL',
    'CALIBRATION_CANDIDATE_OLD_POLICY_TOKEN_KL','NEGATIVE_JOINT_ACTION_TAIL_COUNT'])
def test_old_policy_probability_support_rejection_abandons_stale_rollout(reason):
    actor = SimpleNamespace(actor_module=torch.nn.Linear(1, 1, bias=False),
        _fresh200_attempt_id=1, _last_safety_grad_norm_after=torch.tensor(0.))
    event = {'OLD_POLICY_TOKEN_KL': 'kl',
             'CALIBRATION_CANDIDATE_OLD_POLICY_TOKEN_KL': 'calibration_kl',
             'NEGATIVE_JOINT_ACTION_TAIL_COUNT': 'tail'}[reason]
    with pytest.raises(KnownRefusal) as caught:
        actual_worker_rejection(actor, 4, 0, event)
    assert caught.value.record['rejection_reason'] == reason
    assert rejection_disposition(caught.value.record) == 'abandon'
    assert rejection_disposition({'rejection_reason':reason,'mini_index':4,
                                  'phase':'before'})=='fail_closed'


def test_unknown_exception_text_is_not_a_safe_numerical_skip():
    assert rejection_disposition({'phase':'NCCL timeout','mini_index':4})=='fail_closed'
    assert rejection_disposition({'phase':'gradient_micro_0','mini_index':None})=='fail_closed'
