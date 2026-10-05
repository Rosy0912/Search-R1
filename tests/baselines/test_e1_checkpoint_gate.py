import os
import copy
import hashlib
import json
import random
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from search_r1.baselines.full_state import StatefulSampler, atomic_save, state_digest, rng_state, restore_rng
from search_r1.baselines.checkpoint_gate import digest, write_manifest, validate_release, wait_for_release


def test_cpu_adam_scheduler_rng_exact_reload_continuation(tmp_path):
    """Actual CPU objects, five counted in same300 horizon, no optimizer rebuild."""
    torch.manual_seed(42);random.seed(42);np.random.seed(42)
    model=torch.nn.Linear(3,2);optim=torch.optim.AdamW(model.parameters(),lr=1e-6)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optim,lambda step:min(step/114,1))
    for _ in range(5):
        optim.zero_grad();model(torch.randn(4,3)).square().mean().backward();optim.step();scheduler.step()
    payload=dict(model=copy.deepcopy(model.state_dict()),optim=copy.deepcopy(optim.state_dict()),
                 scheduler=copy.deepcopy(scheduler.state_dict()),rng=rng_state())
    path=tmp_path/'rank.pt';atomic_save(payload,path)
    loaded=torch.load(path,weights_only=False);assert state_digest(loaded)==state_digest(payload)
    # The control sixth update and same-object reloaded sixth must be identical.
    def sixth():
        optim.zero_grad();model(torch.randn(4,3)).square().mean().backward();optim.step();scheduler.step()
        return state_digest(dict(model=model.state_dict(),optim=optim.state_dict(),scheduler=scheduler.state_dict(),rng=rng_state()))
    expected=sixth();model.load_state_dict(loaded['model']);optim.load_state_dict(loaded['optim'])
    scheduler.load_state_dict(loaded['scheduler']);restore_rng(loaded['rng'])
    assert sixth()==expected and scheduler.last_epoch==6
    assert optim.state and scheduler.get_last_lr()[0]==pytest.approx(1e-6*6/114)


def test_sampler_active_iterator_preserves_next_batch_and_epoch():
    sampler=StatefulSampler(range(33),42);it=iter(sampler)
    first=[next(it)for _ in range(10)];saved=copy.deepcopy(sampler.state_dict())
    expected=[next(it)for _ in range(6)];sampler.load_state_dict(saved)
    assert [next(it)for _ in range(6)]==expected
    assert len(set(first+expected))==16
    bad=copy.deepcopy(saved);bad['order']=[0]*33
    with pytest.raises(ValueError):sampler.load_state_dict(bad)


def test_rng_roundtrip_all_cpu_domains():
    saved=rng_state();a=(random.random(),np.random.random(),torch.rand(3))
    restore_rng(saved);b=(random.random(),np.random.random(),torch.rand(3))
    assert a[:2]==b[:2] and torch.equal(a[2],b[2])


def fixture(root):
    repo=root/'repo';repo.mkdir();subprocess.run(['git','init','-q',str(repo)],check=True)
    subprocess.run(['git','-C',str(repo),'-c','user.email=t@t','-c','user.name=t','commit','--allow-empty','-qm','fixture'],check=True)
    source=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    run=root/'runs'/'run5';directory=run/'full_state'/'global_step_5';directory.mkdir(parents=True)
    driver=directory/'driver-reload.json';driver_state=directory/'driver.pt'
    driver_payload=dict(driver_pid=os.getpid(),global_step=5,next_global_step=6,actual_outer_updates=5)
    atomic_save(driver_payload,driver_state)
    driver.write_text(json.dumps(dict(status='EXACT_DRIVER_RELOAD_OK',step=5,pid=os.getpid(),
        path=str(driver_state),sha256=digest(driver_state),state_digest=state_digest(driver_payload))))
    rankfiles=[];reports=[]
    for rank in range(8):
        f=directory/f'rank_{rank}.pt';atomic_save(dict(rank=rank,adam=rank),f);rankfiles.append(f)
        report=dict(status='EXACT_FULL_STATE_RELOAD_OK',rank=rank,world_size=8,step=5,scheduler_last_epoch=5,optimizer_state_entries=1,model_state_dict_type='SHARDED_STATE_DICT',model_dtensor_entries=1,path=str(f),sha256=digest(f),pid=100+rank,state_digest='a'*64)
        reports.append(report);rf=f.with_suffix('.json');rf.write_text(json.dumps(report));rankfiles.append(rf)
    (run/'train.log').write_text(''.join('BAPO_TRAIN_METRICS_JSON='+json.dumps(dict(step=i,metrics={'bapo/actual_outer_updates':i}))+'\n'for i in range(1,6)))
    snapshot=root/'step5-prefix.log';snapshot.write_bytes((run/'train.log').read_bytes())
    (run/'source-commit.txt').write_text(source)
    manifest=write_manifest(rankfiles+[driver,driver_state,run/'source-commit.txt'],directory,dict(trainer=dict(total_training_steps=301,e1_gate_enabled=True,acceptance_step=5),actor_rollout_ref=dict(actor=dict(optim=dict(num_warmup_steps=114,total_training_steps=301)))),source,run,5,reports)
    m=json.loads(manifest.read_text());review=root/'review.md';review.write_text('Independent fixture review')
    receipt=root/'release.json';value={k:m[k]for k in ('run_root','source_commit','global_step','actual_outer_updates','horizon','warmup_steps','config_sha256','pause_nonce')}
    value.update(status='PASS_INDEPENDENT_E1_STEP5_REVIEW',checkpoint_manifest_sha256=digest(manifest),
       evidence=[dict(path=x['path'],sha256=x['sha256'])for x in m['files']]+[dict(path=str(manifest),sha256=digest(manifest)),dict(path=str(snapshot),sha256=digest(snapshot))],
       independent_review=dict(path=str(review),sha256=digest(review)),
       train_log_prefix=dict(live_path=str(run/'train.log'),snapshot_path=str(snapshot),bytes=snapshot.stat().st_size,sha256=digest(snapshot)))
    return repo,manifest,receipt,value


def test_absent_receipt_actually_blocks_same_process_before_six(tmp_path):
    repo,manifest,receipt,value=fixture(tmp_path);steps=[1,2,3,4,5];errors=[]
    def worker():
        try:wait_for_release(receipt,manifest,repo,interval=.01);steps.append(6)
        except Exception as e:errors.append(e)
    thread=threading.Thread(target=worker);thread.start();time.sleep(.05)
    assert thread.is_alive() and steps==[1,2,3,4,5]
    receipt.write_text(json.dumps(value));thread.join(5)
    assert not errors and not thread.is_alive() and steps==[1,2,3,4,5,6]


@pytest.mark.parametrize('kind',['wrong-source','wrong-config','wrong-run','wrong-nonce','wrong-step','wrong-horizon','wrong-warmup','wrong-status','changed-checkpoint','missing-evidence','changed-review','dirty-source','missing-rank','bad-scheduler','driver-not-reloaded'])
def test_release_fails_closed(tmp_path,kind):
    repo,manifest,receipt,value=fixture(tmp_path)
    keys={'wrong-source':'source_commit','wrong-config':'config_sha256','wrong-run':'run_root','wrong-nonce':'pause_nonce','wrong-step':'global_step','wrong-horizon':'horizon','wrong-warmup':'warmup_steps','wrong-status':'status'}
    if kind in keys:value[keys[kind]]='invalid'
    elif kind=='changed-checkpoint':Path(value['evidence'][0]['path']).write_bytes(b'changed')
    elif kind=='missing-evidence':value['evidence']=[]
    elif kind=='changed-review':Path(value['independent_review']['path']).write_text('changed')
    elif kind=='dirty-source':(repo/'dirty.txt').write_text('changed')
    else:
        m=json.loads(manifest.read_text())
        if kind=='missing-rank':m['rank_roundtrips'].pop()
        elif kind=='bad-scheduler':m['rank_roundtrips'][0]['scheduler_last_epoch']=0
        else:Path(next(x['path']for x in m['files']if x['path'].endswith('/driver-reload.json'))).write_text('{}')
        manifest.write_text(json.dumps(m));value['checkpoint_manifest_sha256']=digest(manifest)
        for x in value['evidence']:
            if x['path']==str(manifest):x['sha256']=digest(manifest)
    receipt.write_text(json.dumps(value))
    with pytest.raises(ValueError):validate_release(receipt,manifest,repo)


def test_stateful_default_sampler_matches_existing_randomsampler_order():
    data=list(range(53))
    torch.manual_seed(42)
    expected=list(torch.utils.data.DataLoader(data,batch_size=7,shuffle=True,drop_last=True,num_workers=0))
    expected_rng=torch.get_rng_state()
    torch.manual_seed(42)
    got=list(torch.utils.data.DataLoader(data,batch_size=7,sampler=StatefulSampler(data),drop_last=True,num_workers=0))
    assert all(torch.equal(x,y)for x,y in zip(expected,got))
    assert torch.equal(torch.get_rng_state(),expected_rng)


def test_actual_worker_save_method_cpu_roundtrip_preserves_objects(tmp_path,monkeypatch):
    """Execute worker method on CPU nn/Adam; FSDP collectives are not GPU proof."""
    import ast,contextlib,types
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    source=Path(__file__).resolve().parents[2]/'verl/workers/fsdp_workers.py'
    tree=ast.parse(source.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='ActorRolloutRefWorker')
    method=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='save_training_state'))
    method.decorator_list=[];module=ast.Module(body=[method],type_ignores=[]);ast.fix_missing_locations(module)
    namespace=dict(torch=torch,os=__import__('os'))
    exec(compile(module,str(source),'exec'),namespace)
    requested_state_types=[]
    def fake_context(*a,**kw):
        requested_state_types.append(a[1]);return contextlib.nullcontext()
    monkeypatch.setattr(FSDP,'state_dict_type',fake_context)
    monkeypatch.setattr(torch.distributed,'barrier',lambda:None)
    monkeypatch.setattr(torch.cuda,'current_device',lambda:0)
    monkeypatch.setattr(torch.cuda,'get_rng_state',lambda device:torch.get_rng_state())
    monkeypatch.setattr(torch.cuda,'set_rng_state',lambda state,device:torch.set_rng_state(state))
    model=torch.nn.Linear(2,1);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-6)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:min(step/114,1))
    for _ in range(5):
        optimizer.zero_grad();model(torch.ones(2,2)).sum().backward();optimizer.step();scheduler.step()
    worker=types.SimpleNamespace(_is_actor=True,_is_offload_param=False,_is_offload_optimizer=False,
        actor_module_fsdp=model,actor_optimizer=optimizer,actor_lr_scheduler=scheduler,rank=0,world_size=1)
    # New v3 includes the actual sharding-manager generation-RNG methods.
    manager_source=source.parent/'sharding_manager/fsdp_vllm.py'
    manager_tree=ast.parse(manager_source.read_text())
    manager_cls=next(n for n in manager_tree.body if isinstance(n,ast.ClassDef)and n.name=='FSDPVLLMShardingManager')
    methods=[copy.deepcopy(n)for n in manager_cls.body if isinstance(n,ast.FunctionDef)and n.name in ('generation_rng_state','load_generation_rng_state')]
    manager_module=ast.Module(body=methods,type_ignores=[]);ast.fix_missing_locations(manager_module)
    manager_namespace=dict(torch=torch);exec(compile(manager_module,str(manager_source),'exec'),manager_namespace)
    manager=types.SimpleNamespace(torch_random_states=torch.get_rng_state().clone(),gen_random_states=None,device_mesh=None)
    for name,fn in manager_namespace.items():
        if name in ('generation_rng_state','load_generation_rng_state'):setattr(manager,name,types.MethodType(fn,manager))
    worker.rollout_sharding_manager=manager
    gen_before=state_digest(manager.generation_rng_state())
    original=(id(model),id(optimizer),id(scheduler));expected=state_digest(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),rng=rng_state()))
    report=namespace['save_training_state'](worker,str(tmp_path),5,True)
    native=torch.load(tmp_path/'rank_0.pt',map_location='cpu',weights_only=False)
    assert native['version']==3 and state_digest(native['generation_rng'])==gen_before
    assert state_digest(manager.generation_rng_state())==gen_before
    assert report['status']=='EXACT_FULL_STATE_RELOAD_OK' and report['optimizer_state_entries']==2 and report['scheduler_last_epoch']==5
    from torch.distributed.fsdp import StateDictType
    assert requested_state_types==[StateDictType.SHARDED_STATE_DICT]
    assert report['model_state_dict_type']=='SHARDED_STATE_DICT'
    assert report['model_dtensor_entries']==0  # This CPU model shim is not native FSDP.
    assert (id(model),id(optimizer),id(scheduler))==original
    assert expected==state_digest(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),rng=rng_state()))


def test_explicit_warmup57_overrides_ratio85_and_rejects_bad_types():
    from search_r1.baselines.full_state import resolve_warmup_steps
    config=dict(total_training_steps=301,lr_warmup_steps_ratio=.285,num_warmup_steps=57)
    assert resolve_warmup_steps(config)==57
    assert resolve_warmup_steps(dict(total_training_steps=301,lr_warmup_steps_ratio=.285))==85
    for v in (True,57.0,-1,'57'):
        with pytest.raises(ValueError):resolve_warmup_steps(dict(config,num_warmup_steps=v))


def test_driver_rng_never_initializes_cuda_contexts(monkeypatch):
    def refuse(*a,**kw):raise AssertionError('Unexpected CUDA initialization')
    monkeypatch.setattr(torch.cuda,'get_rng_state_all',refuse)
    monkeypatch.setattr(torch.cuda,'get_rng_state',refuse)
    assert rng_state()['cuda']==[]


def test_actual_trainer_tail_has_one_300_horizon_and_five_pause():
    import runpy,types
    flow=runpy.run_path(str(Path(__file__).resolve().parents[1]/'reproduction/test_final_checkpoint.py'))
    trainer=flow['FakeTrainer'](1,301,save_freq=25,test_freq=0)
    trainer.config.trainer.e1_gate_enabled=True;trainer.config.trainer.acceptance_step=5
    controller=types.SimpleNamespace(method='bapo',actual_outer_updates=0)
    def after(batch,metrics):controller.actual_outer_updates+=1;return {}
    controller.after_actor=after;controller.stage_validation=lambda step:{}
    trainer.baseline_controller=controller
    trainer._e1_pause_after_five=lambda:trainer.events.append(('paused',trainer.global_steps,trainer.weight))
    run=flow['real_iteration_tail']()
    for _ in range(300):run(trainer)
    updates=[x for x in trainer.events if x[0]=='update']
    saves=[x[1]for x in trainer.events if x[0]=='save']
    assert len(updates)==300 and [x[1]for x in updates]==list(range(1,301))
    assert saves==[5]+list(range(25,301,25))
    assert controller.actual_outer_updates==300 and trainer.global_steps==300
    assert ('paused',5,5) in trainer.events
    assert trainer.events.index(('save',5,5))<trainer.events.index(('paused',5,5))<trainer.events.index(('update',6,6))
    assert trainer.events[-1]==('validate',300,300)


def test_live_log_may_grow_but_signed_fifth_prefix_cannot_change(tmp_path):
    repo,manifest,receipt,value=fixture(tmp_path);receipt.write_text(json.dumps(value))
    log=Path(value['train_log_prefix']['live_path'])
    with log.open('a')as f:f.write('late Ray warning while waiting\n')
    assert validate_release(receipt,manifest,repo)['global_step']==5
    contents=log.read_bytes();log.write_bytes(b'changed'+contents)
    with pytest.raises(ValueError):validate_release(receipt,manifest,repo)

@pytest.mark.parametrize('kind',['config-header','rank-alias','rank-json','driver-step','driver-pid','driver-digest','driver-path'])
def test_release_rejects_coherently_rehashed_false_state_proof(tmp_path,kind):
    repo,manifest,receipt,value=fixture(tmp_path)
    m=json.loads(manifest.read_text())
    if kind=='config-header':
        m['config_sha256']='b'*64;value['config_sha256']='b'*64
    elif kind=='rank-alias':
        first=m['rank_roundtrips'][0]
        for report in m['rank_roundtrips'][1:]:
            report['path']=first['path'];report['sha256']=first['sha256']
    elif kind=='rank-json':
        path=Path(m['rank_roundtrips'][0]['path']).with_suffix('.json')
        item=json.loads(path.read_text());item['pid']=999;path.write_text(json.dumps(item))
    else:
        path=Path(next(x['path']for x in m['files']if x['path'].endswith('/driver-reload.json')))
        item=json.loads(path.read_text())
        item[{'driver-step':'step','driver-pid':'pid','driver-digest':'state_digest','driver-path':'path'}[kind]]=999 if kind in ('driver-step','driver-pid') else 'wrong'
        path.write_text(json.dumps(item))
    for item in m['files']:item['sha256']=digest(item['path'])
    manifest.write_text(json.dumps(m));value['checkpoint_manifest_sha256']=digest(manifest)
    for item in value['evidence']:item['sha256']=digest(item['path'])
    receipt.write_text(json.dumps(value))
    with pytest.raises(ValueError):validate_release(receipt,manifest,repo)


def test_release_rejects_coherent_foreign_driver_pid(tmp_path):
    repo,manifest,receipt,value=fixture(tmp_path)
    m=json.loads(manifest.read_text())
    state_path=Path(next(x['path']for x in m['files']if x['path'].endswith('/driver.pt')))
    report_path=state_path.with_name('driver-reload.json')
    state=torch.load(state_path,weights_only=False);state['driver_pid']=os.getpid()+100000
    atomic_save(state,state_path)
    report=json.loads(report_path.read_text());report.update(pid=state['driver_pid'],
        sha256=digest(state_path),state_digest=state_digest(state))
    report_path.write_text(json.dumps(report))
    for item in m['files']:item['sha256']=digest(item['path'])
    manifest.write_text(json.dumps(m));value['checkpoint_manifest_sha256']=digest(manifest)
    for item in value['evidence']:item['sha256']=digest(item['path'])
    receipt.write_text(json.dumps(value))
    # Offline consistency alone is insufficient; live release must bind its PID.
    validate_release(receipt,manifest,repo,expected_driver_pid=state['driver_pid'])
    with pytest.raises(ValueError):
        validate_release(receipt,manifest,repo,expected_driver_pid=os.getpid())
    with pytest.raises(ValueError):wait_for_release(receipt,manifest,repo,interval=.01)


@pytest.mark.parametrize('kind',['old-local-format','no-dtensor'])
def test_release_requires_new_sharded_dtensor_proof(tmp_path,kind):
    repo,manifest,receipt,value=fixture(tmp_path);m=json.loads(manifest.read_text())
    report=m['rank_roundtrips'][0]
    if kind=='old-local-format':report['model_state_dict_type']='LOCAL_STATE_DICT'
    else:report['model_dtensor_entries']=0
    Path(report['path']).with_suffix('.json').write_text(json.dumps(report))
    for item in m['files']:item['sha256']=digest(item['path'])
    manifest.write_text(json.dumps(m));value['checkpoint_manifest_sha256']=digest(manifest)
    for item in value['evidence']:item['sha256']=digest(item['path'])
    receipt.write_text(json.dumps(value))
    with pytest.raises(ValueError):validate_release(receipt,manifest,repo)
