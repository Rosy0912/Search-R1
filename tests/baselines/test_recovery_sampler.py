import pytest
import torch
from search_r1.baselines.training_state import CheckpointBatchSampler, migrate_sampler_batch
from search_r1.baselines.training_state import worker_snapshot, restore_worker, digest
from search_r1.baselines.training_state import driver_snapshot,restore_driver
from search_r1.baselines.training_state import verified_state_file,atomic_save,file_sha
from search_r1.baselines.bapo import BoundaryStage
from types import SimpleNamespace
import random


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
    # Production offload touches state[p] before any update, creating empty
    # entries. These are untouched Adam, not initialized moments.
    for p in new.parameters():newopt.state[p]
    before=digest(worker_snapshot(new,newopt,newsched,{'parent':'verified'}))
    bad=worker_snapshot(old,opt,sched,{'parent':'verified'});bad['optimizer']['state']={}
    with pytest.raises(ValueError,match='coverage'):
        restore_worker(new,newopt,newsched,bad,{'parent':'verified'},cold=True)
    assert digest(worker_snapshot(new,newopt,newsched,{'parent':'verified'}))==before
    restore_worker(new,newopt,newsched,saved,{'parent':'verified'},cold=True)
    assert digest(worker_snapshot(new,newopt,newsched,{'parent':'verified'}))==digest(saved)
    with pytest.raises(ValueError,match='untouched'):
        restore_worker(new,newopt,newsched,saved,{'parent':'verified'},cold=True)
    for model,optimizer,scheduler in [(old,opt,sched),(new,newopt,newsched)]:
        optimizer.zero_grad();model(torch.ones(3,4)).square().sum().backward();optimizer.step();scheduler.step()
    assert digest(worker_snapshot(new,newopt,newsched,{'parent':'verified'}))==digest(worker_snapshot(old,opt,sched,{'parent':'verified'}))


def test_mixed_batch_checkpoint_can_restore_after_epoch_crossing():
    old=CheckpointBatchSampler(169028,512,42,'fixed')
    it=iter(old)
    for _ in range(100):next(it)
    new=CheckpointBatchSampler(169028,640,42,'fixed')
    migrate_sampler_batch(new,old.state_dict(),640)
    left=200
    while left:
        for _ in new:
            left-=1
            if not left:break
    assert new.epoch==1 and new.cursor==10240
    trainer=SimpleNamespace(checkpoint_sampler=new,global_steps=300,actual_outer_updates=300,
        baseline_controller=SimpleNamespace(method='tree_grpo',tree_rng=random.Random(42),stage=BoundaryStage()),
        recovery_provenance=dict(parent_completed_step=100,parent_cursor=51200,parent_epoch=0,new_batch_size=640))
    saved=driver_snapshot(trainer,{'identity':'recovery'})
    fresh=SimpleNamespace(checkpoint_sampler=CheckpointBatchSampler(169028,640,17,'fixed'),global_steps=0,
        actual_outer_updates=0,baseline_controller=SimpleNamespace(method='tree_grpo',tree_rng=random.Random(1),stage=BoundaryStage()))
    restore_driver(fresh,saved,{'identity':'recovery'})
    assert digest(driver_snapshot(fresh,{'identity':'recovery'}))==digest(saved)


def test_parent_loader_verifies_exact_file_before_deserialization(tmp_path):
    p=tmp_path/'parent.pt';atomic_save({'value':torch.tensor([7.])},p)
    r={'path':str(p),'sha256':file_sha(p)}
    assert verified_state_file(r)['value'].item()==7
    with pytest.raises(ValueError,match='SHA'):
        verified_state_file({**r,'sha256':'0'*64})
