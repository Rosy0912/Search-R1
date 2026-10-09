"""Tiny original-file/restore-witness fixtures; never a real CUDA acceptance."""
import copy
import hashlib
import json
from pathlib import Path
import pytest
import torch
from search_r1.baselines import fresh200_gate as gate
from search_r1.baselines import aligned_partial_resume as resume
from search_r1.baselines.aligned_outer_commit import _digest
from search_r1.baselines.full_state import state_digest


def fixture(tmp_path, monkeypatch):
    old=tmp_path/'old/runs/actual3356';new=tmp_path/'new/runs/continuation-9999'
    old.mkdir(parents=True);(new/'partial-resume').mkdir(parents=True)
    monkeypatch.setattr(resume,'OLD_RUN',old)
    closure=[]
    def bind(path, value, binary=False, original=True):
        path.parent.mkdir(parents=True,exist_ok=True)
        if binary:torch.save(value,path)
        else:path.write_text(json.dumps(value,sort_keys=True,allow_nan=False))
        entry=dict(path=str(path),sha256=gate.digest(path),bytes=path.stat().st_size)
        if original:closure.append(entry)
        return entry
    events=[dict(rank=r,baseline_aligned=True,actual_outer=1,attempt_id=1,status='COMPLETE',optimizer_steps=20,
        expected_optimizer_steps=20,optimizer_opportunities_total=20,processed_opportunities=20,
        skipped_opportunities=0,unprocessed_opportunities=0,postabort=False,update_complete=True,
        attempt_complete=True,outer_commit_phase='WRITES_PREPARED',accepted_complete_outer=False,
        scheduler_before=0,scheduler_after=0)for r in range(8)]
    pending=dict(actual_outer=1,attempt_id=1,scheduler_before=0,write_ledger_sha256=_digest(events))
    original_config=bind(old/'resolved-config.json',dict(baseline_aligned=True,baselines=dict(algorithm_source_commit=resume.OLD_SOURCE)))
    ranks=[];witnesses=[];reports=[]
    identity=dict(old_run_root=str(old),old_source_commit=resume.OLD_SOURCE,target_run_root=str(new),
        target_job_id='9999',target_source_commit='b'*40)
    for rank in range(8):
        state=bind(old/f'ranks/rank_{rank}.pt',{'tiny_fixture':rank},binary=True)
        report=dict(path=state['path'],sha256=state['sha256'],pid=100+rank,
            status='PARTIAL_CURRENT_STATE_SAVED',native_state_version=3,scheduler_last_epoch=0,
            latest_rank_event=events[rank],aligned_pending_outer=pending,state_digest='c'*64,
            generation_rng_before_digest='d'*64,generation_rng_after_digest='d'*64)
        side=bind(old/f'ranks/rank_{rank}.json',report)
        ranks.append(dict(rank=rank,state=state,report=side));reports.append(dict(rank=rank,pid=200+rank))
        witnesses.append(dict(identity,status='EXACT_CROSS_JOB_ALIGNED_PARTIAL_V3_RELOAD_OK',rank=rank,
            world_size=8,step=1,attempt_id=1,old_pid=100+rank,new_pid=200+rank,
            original_state_sha256=state['sha256'],state_digest='c'*64,reloaded_state_digest='c'*64,
            scheduler_last_epoch=0,adam_steps=20,aligned_pending_outer=pending,generation_rng_digest='d'*64,
            no_adam_or_stage_observe_or_scheduler_step=True))
    driver=dict(status='PARTIAL',source_commit=resume.OLD_SOURCE,driver_pid=123,global_step=1,
        next_global_step=1,attempt_id=1,actual_outer_updates=0,previous_completed_outer=0,
        rank_write_ledger=events,sampler={'cursor':512},rng={'tiny_cpu_fixture':True},
        bapo=dict(best=14/64,stalled=0,early=True,patience=5))
    driver_entry=bind(old/'driver.pt',driver,binary=True)
    driver_report=bind(old/'driver-reload.json',dict(path=driver_entry['path'],sha256=driver_entry['sha256'],
        status='PARTIAL_CURRENT_STATE_SAVED',state_digest=state_digest(driver)))
    dw=dict(identity,status='EXACT_ALIGNED_PARTIAL_DRIVER_RELOAD_OK',old_pid=123,new_pid=456,
        loaded_state_digest=state_digest(driver),scheduler_before=0,completed_outer_before=0,
        attempt_id=1,sampler_cursor=512,no_generation_or_optimizer_or_stage_observe=True)
    for name in ('rng','sampler','bapo'):
        dw['saved_'+name+'_digest']=dw['restored_'+name+'_digest']=state_digest(driver[name])
    metrics={'val/evaluated_count/nq':64,'val/test_score/nq':14/64,'val/qa_count/all':64}
    raw=json.dumps(metrics,sort_keys=True)
    log=old/'train.log';log.write_text('BASELINE_EVALUATION_METRICS_JSON='+raw+'\n')
    log_entry=dict(path=str(log),sha256=gate.digest(log),bytes=log.stat().st_size);closure.append(log_entry)
    stage=bind(new/'partial-resume/original-stage.json',dict(metrics=metrics,score=14/64,
        marker_line_sha256=hashlib.sha256(raw.encode()).hexdigest(),original_log=log_entry,
        observed_in_saved_bapo=True),original=False)
    manifest=dict(identity,ranks=ranks,rank_write_ledger=events,driver=driver_entry,
        driver_report=driver_report,stage_metrics=stage,original_config=original_config,original_closure=closure)
    document=dict(manifest=manifest,rank_witnesses=witnesses,driver_witness=dw)
    lineage=bind(new/'partial-resume/lineage.json',document,original=False)
    members={entry['path']:entry['sha256']for entry in closure+[stage,lineage]}
    return new,document,members,reports


def test_frozen_originals_and_exact_restored_witnesses(tmp_path,monkeypatch):
    new,document,members,reports=fixture(tmp_path,monkeypatch)
    verified=gate.partial_lineage(new,'b'*40,members,reports,456)
    assert verified['old_driver_pid']==123
    assert verified['manifest']['rank_write_ledger'][0]['optimizer_steps']==20


@pytest.mark.parametrize('fault',['missing_state','different_state','new_pid','extra_adam','driver_component','stage_repeated','original_member_missing','target_source','stage_score','stage_short'])
def test_lineage_rejects_missing_or_changed_provenance(tmp_path,monkeypatch,fault):
    new,document,members,reports=fixture(tmp_path,monkeypatch)
    if fault=='missing_state':Path(document['manifest']['ranks'][0]['state']['path']).unlink()
    if fault=='different_state':Path(document['manifest']['ranks'][0]['state']['path']).write_bytes(b'changed')
    if fault=='new_pid':document['rank_witnesses'][0]['new_pid']=999
    if fault=='extra_adam':document['rank_witnesses'][0]['adam_steps']=21
    if fault=='driver_component':document['driver_witness']['saved_sampler_digest']=document['driver_witness']['restored_sampler_digest']='e'*64
    if fault=='stage_repeated':document['driver_witness']['no_generation_or_optimizer_or_stage_observe']=False
    if fault=='original_member_missing':members.pop(document['manifest']['ranks'][0]['state']['path'])
    if fault=='target_source':document['manifest']['target_source_commit']='a'*40
    if fault in ('stage_score','stage_short'):
        entry=document['manifest']['stage_metrics'];stage=Path(entry['path']);value=json.loads(stage.read_text())
        if fault=='stage_score':value['score']=.5
        else:value['metrics']['val/evaluated_count/nq']=63
        stage.write_text(json.dumps(value));entry['sha256']=gate.digest(stage);entry['bytes']=stage.stat().st_size
        members[str(stage)]=entry['sha256']
    path=new/'partial-resume/lineage.json';path.write_text(json.dumps(document));members[str(path)]=gate.digest(path)
    with pytest.raises(ValueError):gate.partial_lineage(new,'b'*40,members,reports,456)


def test_actual_original_yaml_config_is_consumed_without_rewriting(tmp_path,monkeypatch):
    new,document,members,reports=fixture(tmp_path,monkeypatch)
    old_entry=document['manifest']['original_config'];old=Path(old_entry['path'])
    path=old.with_suffix('.yaml')
    path.write_text('baseline_aligned: true\nbaselines:\n  algorithm_source_commit: '+resume.OLD_SOURCE+'\n')
    entry=dict(path=str(path),sha256=gate.digest(path),bytes=path.stat().st_size)
    document['manifest']['original_config']=entry
    document['manifest']['original_closure'].remove(old_entry)
    document['manifest']['original_closure'].append(entry)
    members.pop(str(old));members[str(path)]=entry['sha256']
    lineage=new/'partial-resume/lineage.json';lineage.write_text(json.dumps(document));members[str(lineage)]=gate.digest(lineage)
    assert gate.partial_lineage(new,'b'*40,members,reports,456)['config']['baseline_aligned']is True
