"""Tiny CPU COMPLETE5 fixtures: no distributed/GPU restore or release acceptance."""
import copy,json
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
import torch
from search_r1.baselines import aligned_complete5_resume as resume
from search_r1.baselines import aligned_complete5_driver as driver
from search_r1.baselines.full_state import state_digest,rng_state
from search_r1.baselines.fresh200_driver import Fresh200Progress


class Mesh:
    device_type='cpu';mesh=torch.arange(8);mesh_dim_names=('dp',)
    def get_coordinate(self):return [0]
class Shard:
    device_mesh=Mesh();placements=('Shard(dim=0)',);shape=(32,);dtype=torch.float32
    def __init__(self):self.local=torch.ones(4)
    def to_local(self):return self.local
    def stride(self):return (1,)


def events():
    return [dict(rank=r,baseline_aligned=True,actual_outer=5,attempt_id=5,status='COMPLETE',
        optimizer_steps=20,expected_optimizer_steps=20,optimizer_opportunities_total=20,
        processed_opportunities=20,skipped_opportunities=0,unprocessed_opportunities=0,
        postabort=False,update_complete=True,attempt_complete=True,
        outer_commit_phase='STAGE_COMMITTED',accepted_complete_outer=True,scheduler_before=4,scheduler_after=5)for r in range(8)]


@pytest.fixture
def manifest(tmp_path):
    return dict(old_run_root=str(resume.OLD_RUN),old_source_commit=resume.OLD_SOURCE,
        target_run_root=str(tmp_path/'runs/continue-9999'),target_job_id='9999',target_source_commit='b'*40,
        ranks=[dict(rank=r)for r in range(8)],rank_write_ledger=events())


@pytest.fixture
def native(manifest):
    parameter=torch.nn.Parameter(torch.ones(4));optimizer=torch.optim.Adam([parameter],lr=1e-6*(6/57))
    for _ in range(100):
        optimizer.zero_grad();parameter.square().sum().backward();optimizer.step()
    payload=dict(version=3,rank=0,world_size=8,step=5,model_state_dict_type='SHARDED_STATE_DICT',
        update_status='COMPLETE',attempted_outer=5,previous_complete_outer=5,attempt_id=5,
        latest_rank_event=manifest['rank_write_ledger'][0],aligned_pending_outer=None,
        model={'weight':Shard()},optimizer=optimizer.state_dict(),scheduler={'last_epoch':5,'_last_lr':[1e-6*(6/57)]},
        rng={'cuda':[{'device':0,'state':torch.zeros(10,dtype=torch.uint8)}]},
        generation_rng=dict(version=1,torch_random_states=torch.zeros(10,dtype=torch.uint8),gen_random_states=torch.ones(10,dtype=torch.uint8)))
    report=dict(status='EXACT_FULL_STATE_RELOAD_OK',native_state_version=3,rank=0,world_size=8,step=5,
        roundtrip=True,pid=123,scheduler_last_epoch=5,model_dtensor_entries=1,optimizer_state_entries=1,
        latest_rank_event=payload['latest_rank_event'],aligned_pending_outer=None,
        generation_rng_before_digest=state_digest(payload['generation_rng']),generation_rng_after_digest=state_digest(payload['generation_rng']))
    report['state_digest']=report['reloaded_state_digest']=state_digest(payload)
    return payload,report


def test_real_cpu_adam100_complete5_state_validates_without_mutation(native,manifest):
    payload,report=native;before=state_digest(payload)
    assert resume.validate_manifest(manifest)==Path(manifest['target_run_root'])
    assert resume.validate_native(payload,report,0,manifest)is payload
    assert state_digest(payload)==before and payload['optimizer']['state'][0]['step'].item()==100


@pytest.mark.parametrize('fault',['partial','adam20','scheduler0','pending','missing_pending','generation_rng','nonfinite_model','wrong_rank'])
def test_complete5_rejects_partial_or_inexact_state(native,manifest,fault):
    payload,report=native
    if fault=='partial':payload['update_status']='PARTIAL'
    if fault=='adam20':payload['optimizer']['state'][0]['step']=torch.tensor(20.)
    if fault=='scheduler0':payload['scheduler']['last_epoch']=0
    if fault=='pending':payload['aligned_pending_outer']={'actual_outer':6}
    if fault=='missing_pending':payload.pop('aligned_pending_outer')
    if fault=='generation_rng':report['generation_rng_after_digest']='a'*64
    if fault=='nonfinite_model':payload['model']['weight'].local[0]=float('nan')
    if fault=='wrong_rank':payload['rank']=1
    report['state_digest']=report['reloaded_state_digest']=state_digest(payload)
    with pytest.raises(ValueError):resume.validate_native(payload,report,0,manifest)


@pytest.fixture
def original_driver(tmp_path,monkeypatch,manifest):
    old=tmp_path/'old/runs/original';old.mkdir(parents=True)
    monkeypatch.setattr(driver,'OLD_RUN',old);monkeypatch.setattr(resume,'OLD_RUN',old)
    d=dict(status='COMPLETE',protocol='BAPO_FRESH200_V1',source_commit=resume.OLD_SOURCE,driver_pid=123,
        global_step=5,next_global_step=6,attempt_id=5,actual_outer_updates=5,attempted_outer=5,
        previous_completed_outer=4,epoch=0,rank_write_ledger=manifest['rank_write_ledger'],
        sampler={'cursor':2560},rng=rng_state(),bapo=dict(best=.4,stalled=1,early=True,patience=5),
        progress=Fresh200Progress(baseline_aligned=True,attempts=5,completed_outers=5,
            actual_adam_writes_per_rank=[100]*8,recorded_attempts=5).state_dict())
    path=old/'checkpoints/fresh200_state/outer_5_COMPLETE/driver.pt';path.parent.mkdir(parents=True);torch.save(d,path)
    def entry(p):return dict(path=str(p),sha256=__import__('hashlib').sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
    side=path.with_name('driver-reload.json');side.write_text(json.dumps(dict(status='EXACT_DRIVER_RELOAD_OK',step=5,pid=123,path=str(path),sha256=entry(path)['sha256'],state_digest=state_digest(d))))
    manifest.update(old_run_root=str(old),driver=entry(path),driver_report=entry(side))
    return d,manifest


def test_complete5_driver_pauses_before6_without_repeating_updates(original_driver,tmp_path,monkeypatch):
    d,manifest=original_driver;run=Path(manifest['target_run_root']);run.mkdir(parents=True)
    class State:
        def load_state_dict(self,value):self.value=copy.deepcopy(value)
        def state_dict(self):return copy.deepcopy(self.value)
    class Loader:
        def __iter__(self):
            torch.rand(3)  # actual DataLoader iterator likewise consumes a seed
            return iter(['next-original-order-batch'])
    calls=[];sampler=State();stage=State()
    trainer=NS(aligned_complete5_manifest=manifest,aligned_complete5_rank_witnesses=[],e1_sampler=sampler,
        baseline_controller=NS(stage=stage),train_dataloader=Loader())
    def save():
        assert trainer.global_steps==5 and trainer.fresh200_progress.actual_adam_writes_per_rank==[100]*8
        assert state_digest(rng_state())==state_digest(d['rng']);calls.append('save-exact5')
    def pause():
        assert trainer.global_steps==5;calls.append('pause-current-process')
    trainer._save_checkpoint=save;trainer._fresh200_pause_after_five=pause
    monkeypatch.setattr(driver.os,'getpid',lambda:456)
    iterator,epoch=driver.restore_driver_and_pause(trainer)
    assert calls==['save-exact5','pause-current-process'] and trainer.global_steps==6 and epoch==0
    assert next(iterator)=='next-original-order-batch'
    assert sampler.state_dict()==d['sampler'] and stage.state_dict()==d['bapo']
    witness=json.loads((run/'complete5-resume/driver-witness.json').read_text())
    assert witness['no_generation_or_optimizer_or_stage_observe_or_scheduler_step']is True
    assert witness['old_pid']==123 and witness['new_pid']==456


@pytest.mark.parametrize('key,value',[('global_step',4),('next_global_step',5),('actual_outer_updates',4),('previous_completed_outer',5)])
def test_driver_counters_cannot_restart_or_skip_updates(original_driver,key,value):
    d,manifest=original_driver;d[key]=value
    path=Path(manifest['driver']['path']);torch.save(d,path)
    manifest['driver'].update(sha256=__import__('hashlib').sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)
    side=Path(manifest['driver_report']['path']);report=json.loads(side.read_text());report.update(sha256=manifest['driver']['sha256'],state_digest=state_digest(d));side.write_text(json.dumps(report))
    manifest['driver_report'].update(sha256=__import__('hashlib').sha256(side.read_bytes()).hexdigest(),bytes=side.stat().st_size)
    with pytest.raises(ValueError):driver.validate_driver(manifest)
