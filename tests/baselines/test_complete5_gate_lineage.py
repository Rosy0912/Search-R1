"""Tiny CPU lineage fixtures, never an actual COMPLETE5/GPU acceptance."""
import copy,json
from pathlib import Path
import pytest
import torch
from search_r1.baselines import fresh200_gate as gate
from search_r1.baselines import aligned_complete5_resume as resume
from search_r1.baselines.full_state import state_digest

@pytest.fixture
def fixture(tmp_path,monkeypatch):
    old=tmp_path/'old/runs/original3357';new=tmp_path/'new/runs/current9999'
    old.mkdir(parents=True);(new/'complete5-resume').mkdir(parents=True)
    monkeypatch.setattr(resume,'OLD_RUN',old)
    monkeypatch.setattr(resume,'validate_manifest',lambda m:new)
    monkeypatch.setattr(gate,'partial_lineage',lambda *a,**k:{'originals':{}})
    def save(name,value):
        p=old/name;p.parent.mkdir(parents=True,exist_ok=True)
        if p.suffix=='.pt':torch.save(value,p)
        else:p.write_text(json.dumps(value))
        return dict(path=str(p),bytes=p.stat().st_size,sha256=gate.digest(p))
    ledger=[dict(rank=i,actual_outer=5)for i in range(8)]
    m=dict(old_run_root=str(old),old_source_commit=resume.OLD_SOURCE,target_run_root=str(new),target_source_commit='b'*40,target_job_id='9999',rank_write_ledger=ledger,ranks=[],original_closure=[])
    common={k:m[k]for k in ('old_run_root','old_source_commit','target_run_root','target_source_commit','target_job_id')}
    witnesses=[];reports=[]
    for rank in range(8):
        state=save(f'rank_{rank}.pt',{'tiny':True})
        report=dict(path=state['path'],sha256=state['sha256'],pid=100+rank,status='EXACT_FULL_STATE_RELOAD_OK',native_state_version=3,roundtrip=True,state_digest='a'*64,reloaded_state_digest='a'*64,aligned_pending_outer=None,rank=rank,world_size=8,step=5,scheduler_last_epoch=5,latest_rank_event=ledger[rank],generation_rng_before_digest='c'*64,generation_rng_after_digest='c'*64)
        side=save(f'rank_{rank}.json',report);m['ranks'].append(dict(rank=rank,state=state,report=side));m['original_closure']+= [state,side]
        witnesses.append(dict(common,status='EXACT_CROSS_JOB_ALIGNED_COMPLETE5_V3_RELOAD_OK',rank=rank,old_pid=100+rank,new_pid=200+rank,original_state_sha256=state['sha256'],state_digest='a'*64,reloaded_state_digest='a'*64,aligned_pending_outer=None,generation_rng_digest='c'*64,no_generation_or_optimizer_or_stage_observe_or_scheduler_step=True,world_size=8,step=5,attempt_id=5,scheduler_last_epoch=5,adam_steps=100))
        reports.append(dict(report,pid=200+rank))
    from search_r1.baselines.fresh200_driver import Fresh200Progress
    driver=dict(driver_pid=123,rank_write_ledger=ledger,rng={'tiny':1},sampler={'cursor':2560},bapo={'best':.2},status='COMPLETE',source_commit=resume.OLD_SOURCE,protocol='BAPO_FRESH200_V1',global_step=5,next_global_step=6,actual_outer_updates=5,attempt_id=5,previous_completed_outer=4,progress=Fresh200Progress(baseline_aligned=True,attempts=5,completed_outers=5,actual_adam_writes_per_rank=[100]*8,recorded_attempts=5).state_dict())
    m['driver']=save('driver.pt',driver)
    m['driver_report']=save('driver-reload.json',dict(pid=123,state_digest=state_digest(driver),status='EXACT_DRIVER_RELOAD_OK',step=5,path=m['driver']['path'],sha256=m['driver']['sha256']))
    m['original_config']=save('original-config.json',dict(baseline_aligned=True,baselines={'algorithm_source_commit':resume.OLD_SOURCE}))
    m['original_closure'] += [m[k]for k in ('driver','driver_report','original_config')]
    witness=dict(common,status='EXACT_ALIGNED_COMPLETE5_DRIVER_RELOAD_OK',old_pid=123,new_pid=456,loaded_state_digest=state_digest(driver),no_generation_or_optimizer_or_stage_observe_or_scheduler_step=True,step=5,next_global_step=6,completed_outer_before=5,attempt_id=5,sampler_cursor=2560)
    for key in ('rng','sampler','bapo'):witness['saved_'+key+'_digest']=witness['restored_'+key+'_digest']=state_digest(driver[key])
    document=dict(manifest=m,rank_witnesses=witnesses,driver_witness=witness)
    path=new/'complete5-resume/lineage.json'
    def write():
        path.write_text(json.dumps(document));return {**{x['path']:x['sha256']for x in m['original_closure']},str(path):gate.digest(path)}
    return new,document,reports,write

def test_preserved_complete5_maps_new_pids_without_retagging_originals(fixture):
    new,d,reports,write=fixture
    value=gate.complete5_lineage(new,'b'*40,write(),reports,456)
    assert value['old_driver_pid']==123
    assert value['reports'][0]['pid']==100 and reports[0]['pid']==200

@pytest.mark.parametrize('mutation',['rank_old_pid','rank_new_pid','rank_digest','rank_adam','rank_pending','rank_updated','driver_pid','driver_rng','driver_cursor','driver_updated','closure_missing','nested_missing'])
def test_inexact_complete5_continuity_refused(fixture,mutation,monkeypatch):
    new,d,reports,write=fixture;w=d['rank_witnesses'][0];driver=d['driver_witness']
    if mutation=='rank_old_pid':w['old_pid']=99
    elif mutation=='rank_new_pid':w['new_pid']=99
    elif mutation=='rank_digest':w['state_digest']='f'*64
    elif mutation=='rank_adam':w['adam_steps']=120
    elif mutation=='rank_pending':w['aligned_pending_outer']={'attempt_id':6}
    elif mutation=='rank_updated':w['no_generation_or_optimizer_or_stage_observe_or_scheduler_step']=False
    elif mutation=='driver_pid':driver['new_pid']=999
    elif mutation=='driver_rng':driver['restored_rng_digest']='f'*64
    elif mutation=='driver_cursor':driver['sampler_cursor']=3072
    elif mutation=='driver_updated':driver['no_generation_or_optimizer_or_stage_observe_or_scheduler_step']=False
    elif mutation=='closure_missing':d['manifest']['original_closure'].pop()
    else:monkeypatch.setattr(gate,'partial_lineage',lambda *a,**k:None)
    with pytest.raises(ValueError):gate.complete5_lineage(new,'b'*40,write(),reports,456)

@pytest.mark.parametrize('mutation',[None,'old_source_bytes','old_source_inventory','current_claim','current_driver','current_input'])
def test_original_five_validator_and_current_binding_are_both_required(tmp_path,monkeypatch,mutation):
    import subprocess
    from search_r1.baselines import baseline_alignment,fresh200_contract
    old=tmp_path/'old/runs/old';new=tmp_path/'new/runs/new';repo=old.parent.parent/'repo'
    old.mkdir(parents=True);new.mkdir(parents=True);(repo/'search_r1/baselines').mkdir(parents=True)
    source_file=repo/'search_r1/baselines/fresh200_contract.py'
    source_file.write_text("def critical_source_hashes():\n    names=('search_r1/baselines/fresh200_contract.py',)\n    return {}\n")
    for cmd in [['git','init','-q',str(repo)],['git','-C',str(repo),'add','.'],['git','-C',str(repo),'-c','user.name=fixture','-c','user.email=f@f','commit','-qm','CPU fixture']]:subprocess.run(cmd,check=True)
    old_source=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    old_hashes={'search_r1/baselines/fresh200_contract.py':gate.digest(source_file)}
    if mutation=='old_source_inventory':old_hashes={}
    old_binding=old/'aligned-step5-bindings.json';old_binding.write_text(json.dumps({'source_sha256':old_hashes}))
    if mutation=='old_source_bytes':source_file.write_text('changed')
    inp=tmp_path/'input.json';inp.write_text('{}');sha=gate.digest(inp)
    config=dict(baselines=dict(holdout=dict(manifest_path=str(inp),manifest_sha256=sha),alignment=dict(order_manifest=str(inp),order_sha256=sha)),retriever=dict(identity_manifest=str(inp),identity_manifest_sha256=sha))
    shared=dict(content_identity='same',common_input_binding=dict(content_identity='same',train_rows=169028))
    claim=new/'.aligned-startup-claim.json';claim.write_text(json.dumps(dict(scope='same-experiment-complete5-verify-only-resume',source_commit='b'*40,launcher_pid=4)))
    value=dict(version=1,status='ALIGNED_STEP5_FACTUAL_BINDINGS_NOT_RELEASE',source_commit='b'*40,source_sha256={'current':'hash'},run_root=str(new),driver_pid=456,actual_outer_updates=5,next_global_step=6,new_run_claim=dict(path=str(claim),sha256=gate.digest(claim)),shared_inputs=shared,inputs={k:dict(path=str(inp),sha256=sha)for k in ('shared_input_manifest','retriever_identity_manifest','order_manifest')})
    if mutation=='current_driver':value['driver_pid']=123
    if mutation=='current_input':value['inputs']['order_manifest']['sha256']='a'*64
    if mutation=='current_claim':claim.write_text(json.dumps(dict(scope='new-run-only-no-resume',source_commit='b'*40,launcher_pid=4)))
    binding=new/'aligned-step5-bindings.json';binding.write_text(json.dumps(value))
    members={str(p):gate.digest(p)for p in (old_binding,binding,claim)}
    lineage=dict(old_run=old,old_source=old_source,config=config,old_driver_pid=123,reports=[{'pid':100}])
    monkeypatch.setattr(fresh200_contract,'critical_source_hashes',lambda:{'current':'hash'})
    monkeypatch.setattr(baseline_alignment,'prepare_aligned_inputs',lambda c:shared)
    calls=[]
    monkeypatch.setattr(gate,'validate_aligned_first_five_evidence',lambda *a,**k:calls.append((a,k)))
    if mutation is None:
        gate.validate_complete5_first_five(members,config,'b'*40,new,456,lineage,[],[])
        assert len(calls)==1 and calls[0][0][2:5]==(old_source,old,123)
        assert calls[0][1]['expected_source_hashes']==old_hashes
    else:
        with pytest.raises(ValueError):gate.validate_complete5_first_five(members,config,'b'*40,new,456,lineage,[],[])
