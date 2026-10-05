import pytest
import torch
from search_r1.baselines.training_state import CheckpointBatchSampler, migrate_sampler_batch
from search_r1.baselines.training_state import worker_snapshot, restore_worker, digest


def test_recovery_preserves_consumed_questions_and_next_batch():
    old=CheckpointBatchSampler(169028,512,42,'fixed-split')
    it=iter(old)
    for _ in range(100):next(it)
    saved=old.state_dict()
    new=CheckpointBatchSampler(169028,640,99,'fixed-split')
    proof=migrate_sampler_batch(new,saved,640)
    assert proof['consumed_cursor']==51200
    assert next(iter(new))==saved['order'][51200:51840]
    assert new.cursor==51840
    assert new.order==saved['order']
    assert new.generator.get_state().equal(saved['generator'])


def test_recovery_rejects_unaligned_cursor_and_different_split():
    old=CheckpointBatchSampler(10000,512,42,'original')
    next(iter(old))
    with pytest.raises(ValueError,match='align'):
        migrate_sampler_batch(CheckpointBatchSampler(10000,640,42,'original'),old.state_dict(),640)
    with pytest.raises(ValueError,match='identity'):
        migrate_sampler_batch(CheckpointBatchSampler(10000,640,42,'different'),old.state_dict(),640)


def test_cold_adam_restore_matches_next_actual_update_without_dummy_steps():
    torch.manual_seed(9)
    old=torch.nn.Linear(4,2)
    opt=torch.optim.AdamW(old.parameters(),lr=.001)
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1)
    for _ in range(3):
        opt.zero_grad(); old(torch.ones(3,4)).square().sum().backward();opt.step();sched.step()
    saved=worker_snapshot(old,opt,sched,{'parent':'verified'})
    new=torch.nn.Linear(4,2); newopt=torch.optim.AdamW(new.parameters(),lr=.001)
    newsched=torch.optim.lr_scheduler.LambdaLR(newopt,lambda _:1)
    before=digest(worker_snapshot(new,newopt,newsched,{'parent':'verified'}))
    bad=worker_snapshot(old,opt,sched,{'parent':'verified'});bad['optimizer']['state']={}
    with pytest.raises(ValueError,match='coverage'):
        restore_worker(new,newopt,newsched,bad,{'parent':'verified'},cold=True)
    assert digest(worker_snapshot(new,newopt,newsched,{'parent':'verified'}))==before
    restore_worker(new,newopt,newsched,saved,{'parent':'verified'},cold=True)
    assert digest(worker_snapshot(new,newopt,newsched,{'parent':'verified'}))==digest(saved)
    for model,optimizer,scheduler in [(old,opt,sched),(new,newopt,newsched)]:
        optimizer.zero_grad();model(torch.ones(3,4)).square().sum().backward();optimizer.step();scheduler.step()
    assert digest(worker_snapshot(new,newopt,newsched,{'parent':'verified'}))==digest(worker_snapshot(old,opt,sched,{'parent':'verified'}))
