import copy, json, random
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from search_r1.baselines.training_state import (
    CheckpointBatchSampler, digest, worker_snapshot, restore_worker, snapshot_worker_to_file,
    driver_snapshot, restore_driver, wait_for_acceptance, atomic_save, file_sha)
from search_r1.baselines.bapo import BoundaryStage
from verl.utils.torch_functional import get_constant_schedule_with_warmup


def test_real_adam_scheduler_all_values_and_rng_restore(tmp_path):
    torch.manual_seed(17); random.seed(17); np.random.seed(17)
    model=torch.nn.Linear(4,2); optimizer=torch.optim.AdamW(model.parameters(),lr=1e-6)
    scheduler=get_constant_schedule_with_warmup(optimizer,num_warmup_steps=57)
    used=[]
    for outer in range(5):
        used.append(optimizer.param_groups[0]['lr'])
        for _ in range(24):
            optimizer.zero_grad();model(torch.randn(3,4)).square().sum().backward();optimizer.step()
        scheduler.step()
    assert used[0]==0 and used[1]==pytest.approx(1e-6/57)
    identity=dict(rank=0,world_size=1,source='test',step=5)
    saved=worker_snapshot(model,optimizer,scheduler,identity);expected=digest(saved)
    expected_rng=(random.random(),np.random.random(),torch.rand(5))
    # Actually disturb model, Adam moments/groups, scheduler and all RNGs.
    with torch.no_grad():
        for p in model.parameters():p.add_(100)
    for state in optimizer.state.values():state['exp_avg'].add_(200);state['exp_avg_sq'].add_(300)
    optimizer.param_groups[0]['lr']=.5;scheduler.step();random.random();np.random.random();torch.rand(99)
    restore_worker(model,optimizer,scheduler,saved,identity)
    assert digest(worker_snapshot(model,optimizer,scheduler,identity))==expected
    actual_rng=(random.random(),np.random.random(),torch.rand(5))
    assert actual_rng[0:2]==expected_rng[0:2] and torch.equal(actual_rng[2],expected_rng[2])
    record=snapshot_worker_to_file(model,optimizer,scheduler,tmp_path/'worker.pt',identity,verify=True)
    assert record['exact_same_world_reload_verified'] and record['optimizer_state_entries']==2
    assert all(s['step'].item()==120 for s in optimizer.state.values())
    with pytest.raises(FileExistsError):snapshot_worker_to_file(model,optimizer,scheduler,tmp_path/'worker.pt',identity)
    with pytest.raises(ValueError):restore_worker(model,optimizer,scheduler,saved,{**identity,'world_size':2})


def test_sampler_tree_rng_and_driver_restore_does_not_replay_firstfive(tmp_path):
    sampler=CheckpointBatchSampler(40,4,42,'dataset');iterator=iter(sampler)
    prefix=[next(iterator) for _ in range(5)]
    trainer=SimpleNamespace(checkpoint_sampler=sampler,global_steps=5,actual_outer_updates=5,
                            baseline_controller=SimpleNamespace(method='tree_grpo',tree_rng=random.Random(42),stage=BoundaryStage()))
    for _ in range(31):trainer.baseline_controller.tree_rng.random()
    saved=driver_snapshot(trainer,{'code':'test'});expected=digest(saved)
    next_batch=next(iterator);next_tree=trainer.baseline_controller.tree_rng.random()
    trainer.global_steps=77;trainer.actual_outer_updates=77;torch.rand(17)
    restore_driver(trainer,saved,{'code':'test'})
    assert digest(driver_snapshot(trainer,{'code':'test'}))==expected
    assert next(iterator)==next_batch and trainer.baseline_controller.tree_rng.random()==next_tree
    assert set(next_batch).isdisjoint(set(sum(prefix,[])))
    bad=copy.deepcopy(saved['sampler']);bad['identity']='wrong'
    with pytest.raises(ValueError):sampler.load_state_dict(bad)
    bad=copy.deepcopy(saved['sampler']);bad['order'][0]=bad['order'][1]
    with pytest.raises(ValueError):sampler.load_state_dict(bad)


def test_gate_timeout_or_wrong_identity_does_not_fetch_sixth_batch(tmp_path):
    sampler=CheckpointBatchSampler(40,4,42,'dataset');iterator=iter(sampler)
    for _ in range(5):next(iterator)
    cursor=sampler.cursor
    ready=dict(job_id='1',code_commit='code',config_sha256='cfg',data_identity='data',model_identity='model')
    with pytest.raises(TimeoutError):wait_for_acceptance(tmp_path/'timeout',ready,-1)
    assert sampler.cursor==cursor==20
    directory=tmp_path/'invalid';directory.mkdir();(directory/'approved.json').write_text(json.dumps({'status':'PASS','ready_sha256':'wrong'}))
    with pytest.raises(ValueError):wait_for_acceptance(directory,ready,1)
    assert sampler.cursor==20
    directory=tmp_path/'valid';directory.mkdir()
    # Atomic ready bytes are deterministic, allowing a prearranged toy reviewer.
    from search_r1.baselines.training_state import atomic_save
    atomic_save(ready,directory/'precomputed.json',json_format=True)
    approval=dict(ready,status='PASS',ready_sha256=file_sha(directory/'precomputed.json'),
                  full_state_reload_accepted=True,full_hf_reload_accepted=True,rank_tree_numerics_accepted=True)
    (directory/'approved.json').write_text(json.dumps(approval))
    wait_for_acceptance(directory,ready,1)
    assert sampler.cursor==20 and (directory/'released.json').is_file()
    next(iterator);assert sampler.cursor==24


def test_57_outer_scheduler_and_300_budget():
    parameter=torch.nn.Parameter(torch.ones(()));optimizer=torch.optim.AdamW([parameter],lr=1e-6)
    scheduler=get_constant_schedule_with_warmup(optimizer,num_warmup_steps=57)
    used=[]
    for _ in range(300):
        used.append(optimizer.param_groups[0]['lr']);optimizer.zero_grad();parameter.square().backward();optimizer.step();scheduler.step()
    assert len(used)==300 and used[0]==0 and used[56]==pytest.approx(56e-6/57)
    assert used[57:]==pytest.approx([1e-6]*243) and scheduler.last_epoch==300


def test_corrupt_adam_is_refused_before_parameter_mutation():
    model=torch.nn.Linear(4,2);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-6)
    scheduler=get_constant_schedule_with_warmup(optimizer,57)
    model(torch.randn(3,4)).sum().backward();optimizer.step()
    saved=worker_snapshot(model,optimizer,scheduler,{'source':'fixed'})
    before=digest(worker_snapshot(model,optimizer,scheduler,{'source':'fixed'}))
    next(iter(saved['optimizer']['state'].values()))['exp_avg']=torch.ones(13)
    with pytest.raises(ValueError,match='Adam moment'):
        restore_worker(model,optimizer,scheduler,saved,{'source':'fixed'})
    assert digest(worker_snapshot(model,optimizer,scheduler,{'source':'fixed'}))==before
    missing=worker_snapshot(model,optimizer,scheduler,{'source':'fixed'});missing['optimizer']['state']={}
    with pytest.raises(ValueError,match='coverage'):restore_worker(model,optimizer,scheduler,missing,{'source':'fixed'})
    assert digest(worker_snapshot(model,optimizer,scheduler,{'source':'fixed'}))==before


def test_incoherent_sampler_and_string_receipt_are_refused(tmp_path):
    sampler=CheckpointBatchSampler(40,4,42,'dataset')
    bad=sampler.state_dict();bad.update(cursor=20,epoch=91)
    with pytest.raises(ValueError,match='Incoherent'):sampler.load_state_dict(bad)
    ready=dict(job_id='1',code_commit='code',config_sha256='cfg',data_identity='data',model_identity='model')
    atomic_save(ready,tmp_path/'expected.json',json_format=True)
    approval=dict(ready,status='PASS',ready_sha256=file_sha(tmp_path/'expected.json'),
        full_state_reload_accepted='false',full_hf_reload_accepted='false',rank_tree_numerics_accepted='false')
    (tmp_path/'approved.json').write_text(json.dumps(approval))
    with pytest.raises(ValueError,match='acceptance missing'):wait_for_acceptance(tmp_path,ready,1)
    assert not (tmp_path/'released.json').exists()


def test_fsdp_independent_local_storage_restored_with_real_adam(tmp_path):
    model=torch.nn.Linear(4,2);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
    scheduler=get_constant_schedule_with_warmup(optimizer,57)
    model(torch.randn(2,4)).square().sum().backward();optimizer.step();scheduler.step()
    for parameter in model.parameters():parameter._local_shard=parameter.detach().clone()
    identity={'rank':0,'world_size':1}
    saved=worker_snapshot(model,optimizer,scheduler,identity)
    pointers={name:(parameter.data_ptr(),parameter._local_shard.data_ptr()) for name,parameter in model.named_parameters()}
    with torch.no_grad():
        for parameter in model.parameters():parameter.add_(13);parameter._local_shard.add_(29)
    restore_worker(model,optimizer,scheduler,saved,identity)
    for name,parameter in model.named_parameters():
        assert torch.equal(parameter,saved['parameters'][name])
        assert torch.equal(parameter._local_shard,saved['local_shards'][name])
        assert (parameter.data_ptr(),parameter._local_shard.data_ptr())==pointers[name]
    assert digest(worker_snapshot(model,optimizer,scheduler,identity))==digest(saved)
    snapshot_worker_to_file(model,optimizer,scheduler,tmp_path/'fsdp-shard.pt',identity,verify=True)


@pytest.mark.parametrize('damage',['coverage','shape'])
def test_bad_fsdp_local_shard_rejected_before_mutating(damage):
    model=torch.nn.Linear(4,2);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
    scheduler=get_constant_schedule_with_warmup(optimizer,57)
    model(torch.randn(2,4)).square().sum().backward();optimizer.step();scheduler.step()
    for parameter in model.parameters():parameter._local_shard=parameter.detach().clone()
    saved=worker_snapshot(model,optimizer,scheduler,{})
    before=digest(worker_snapshot(model,optimizer,scheduler,{}))
    if damage=='coverage':saved['local_shards']={}
    else:saved['local_shards']['weight']=torch.zeros(1)
    with pytest.raises(ValueError,match='FSDP local shard'):restore_worker(model,optimizer,scheduler,saved,{})
    assert digest(worker_snapshot(model,optimizer,scheduler,{}))==before
