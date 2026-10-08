import copy
import importlib.util
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('audit',Path(__file__).with_name('resume_final_audit_v1.py'))
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def fixture(monkeypatch,tmp_path):
    monkeypatch.setattr(audit,'ROOT',tmp_path)
    (tmp_path/'common-data').mkdir()
    for name in ('stage','validation','final'):(tmp_path/'common-data'/f'{name}.parquet').write_bytes(name.encode())
    records=[]
    for phase,step in [('stage64',85)]+[('periodic512',s) for s in range(100,301,50)]+[('final51713',300)]:
        name={'stage64':'stage','periodic512':'validation','final51713':'final'}[phase]
        file=tmp_path/f'common-data/{name}.parquet';count={'stage64':64,'periodic512':512,'final51713':51713}[phase]
        metrics={'val/qa_count/all':count}
        if count in (64,512):metrics.update({'val/qa_count/nq':count//2,'val/qa_count/hotpotqa':count//2})
        else:
            for index,n in enumerate([1,2,3,4,5,6,51692]):metrics[f'val/qa_count/set{index}']=n
        records.append(dict(phase=phase,logical_step=step,actual_new_updates=step-80,job_id='new',commit='source',config_sha256='config',pid=12,data_identity=audit.CONTENT,dataset_files=[dict(path=str(file),sha256=audit.sha(file))],metrics=metrics))
    return records


def test_complete_cross_job_additional_sequence_stops300():
    audit.validate_update_sequence([dict(outer_step=s) for s in range(81,301)],81,300)
    for numbers in [list(range(81,300)),list(range(81,302)),[81]+list(range(81,301)),list(range(1,221))]:
        with pytest.raises(RuntimeError):audit.validate_update_sequence([dict(outer_step=s) for s in numbers],81,300)


def test_actual_phase_identity_accepted(monkeypatch,tmp_path):
    r=fixture(monkeypatch,tmp_path);audit.audit_new_evaluations(r,'new','source','config',12)


@pytest.mark.parametrize('change',['duplicate','wrong-step','wrong-job','wrong-source','wrong-data','wrong-actual','wrong-pid','missing-set','wrong-final-count'])
def test_same_size_or_count_cannot_replace_phase_identity(monkeypatch,tmp_path,change):
    records=fixture(monkeypatch,tmp_path)
    if change=='duplicate':records.append(copy.deepcopy(records[-1]))
    elif change=='wrong-step':records[1]['logical_step']=150
    elif change=='wrong-job':records[-1]['job_id']='3185'
    elif change=='wrong-source':records[-1]['commit']='base'
    elif change=='wrong-data':records[0]['dataset_files'][0]['sha256']='bad'
    elif change=='wrong-actual':records[-1]['actual_new_updates']=300
    elif change=='wrong-pid':records[-1]['pid']=13
    elif change=='missing-set':del records[-1]['metrics']['val/qa_count/set0']
    elif change=='wrong-final-count':records[-1]['metrics']['val/qa_count/all']=64
    with pytest.raises(RuntimeError):audit.audit_new_evaluations(records,'new','source','config',12)


def test_raw_selection_ignores_latest_symlink_and_preserves_forwarded(monkeypatch,tmp_path):
    (tmp_path/'audit').mkdir();(tmp_path/'audit/driver.jsonl').write_text(json.dumps({'pid':12})+'\n')
    logs=tmp_path/'ray/session_fixed/logs';logs.mkdir(parents=True)
    raw=logs/'worker-a-b-12.out';raw.write_text('complete raw data\n')
    (tmp_path/'ray/session_latest').symlink_to(logs.parent,target_is_directory=True)
    forwarded=tmp_path/'training.log';forwarded.write_text('incomplete forwarded\n')
    content,receipt=audit.raw_driver_snapshot(tmp_path)
    assert content=='complete raw data\n' and receipt['path']==str(raw)
    assert forwarded.read_text()=='incomplete forwarded\n'
    (logs/'worker-other-b-12.out').write_text('duplicate')
    with pytest.raises(RuntimeError,match='ambiguous'):audit.raw_driver_snapshot(tmp_path)


def full_metadata_fixture(monkeypatch,tmp_path):
    import os
    from search_r1.baselines import instruct300
    monkeypatch.setattr(instruct300,'check_manifest',lambda value:None)  # Physical PT/HF proof remains real runtime/native scope.
    monkeypatch.setattr(audit,'ROOT',tmp_path);old=tmp_path/'old';prior=tmp_path/'prior';run=tmp_path/'new'
    monkeypatch.setattr(audit,'OLD',old);monkeypatch.setattr(audit,'PRIOR',prior);monkeypatch.setenv('SLURM_JOB_ID','new')
    (run/'audit').mkdir(parents=True);(run/'audit/entry-config.yaml').write_text('config')
    config_hash=audit.sha(run/'audit/entry-config.yaml')
    entry=dict(kind='config_boundary',actual_config_sha256=config_hash,actual_data_identity=audit.CONTENT,entry_total_training_steps=300)
    (run/'audit/entry.jsonl').write_text(json.dumps(entry)+'\n')
    drivers=[dict(outer_step=s,pid=12,commit='source',config_sha256=config_hash,data_identity=audit.CONTENT,questions=512,rows=2560,method='gigpo',tensors={},observation_tokens=1) for s in range(81,301)]
    (run/'audit/driver.jsonl').write_text('\n'.join(json.dumps(r) for r in drivers)+'\n')
    workers=[dict(rank=r,pid=100+r,start_ticks=str(r)) for r in range(8)]
    restored=dict(status='RESTORED_FULL_STATE80',source_manifest={'sha256':audit.PRIOR_MANIFEST80},driver_cursor=40960,retained_logical_step=80,resumed_actual_updates=0,workers=workers)
    (run/'resume80-restored.json').write_text(json.dumps(restored))
    stats=dict(finite=1,count=1,nonzero=1)
    for rank in range(8):
        records=[dict(outer_step=s,pid=100+rank,rank=rank,epochs=1,optimizer_steps=20,commit='source',config_sha256=config_hash,data_identity=audit.CONTENT,micro_stats=[dict(joint_action_ratios=stats)]*160,grad_finite=20,grad_nonzero=20,skipped_empty_minibatches=0,fixed_unchanged=True,shard_layout_unchanged=True,mask_binary=True,grad_norms=[1.]*20,tensors={},shard_delta=stats) for s in range(81,301)]
        (run/'audit'/f'{rank}.jsonl').write_text('\n'.join(json.dumps(r) for r in records)+'\n')
    original={}
    for point in [5,25,50,75,80,85]+list(range(100,301,25)):
        old_point=point<=75;prior_point=point==80;base=old if old_point else prior if prior_point else run
        actor=base/f'checkpoints/actor/global_step_{point}/training_state';actor.mkdir(parents=True)
        algorithm=base/f'checkpoints/algorithm/global_step_{point}/state.json';algorithm.parent.mkdir(parents=True)
        algorithm.write_text(json.dumps(dict(method='gigpo',global_step=point,resume_supported=True,loss_reduction='sequence_mean')))
        manifest=dict(logical_step=point,actual_new_updates=point if old_point else 5 if prior_point else point-80,job_id='3185' if old_point else '3225' if prior_point else 'new',code_commit=audit.OLD_COMMIT if old_point else audit.PRIOR_COMMIT if prior_point else 'source',recovery_mode='fresh_official_instruct' if old_point else 'full_state_instruct75' if prior_point else 'full_state_instruct80',data_identity=audit.CONTENT,source=str(tmp_path/'assets/qwen2.5-3b-instruct-aa8e725'),config_sha256='b5deaadd43ca16cebaf9c4f0a4320b47a4d7664830cc07d0367ad1e72463d2a8' if prior_point else config_hash,files=[dict(path=str(algorithm))]+[dict(pause_identity=w) for w in workers])
        path=actor/'manifest.json';path.write_text(json.dumps(manifest))
        if old_point:original[point]=audit.sha(path)
        if prior_point:monkeypatch.setattr(audit,'PRIOR_MANIFEST80',audit.sha(path))
    monkeypatch.setattr(audit,'OLD_MANIFESTS',original)
    restored['source_manifest']['sha256']=audit.PRIOR_MANIFEST80;(run/'resume80-restored.json').write_text(json.dumps(restored))
    receipt=dict(logical_step=300,actual_new_updates=220,retained_original_updates=80,driver_cursor=153600,manifest=dict(path=str(path),sha256=audit.sha(path),identity=manifest),actual_state_roundtrip=[dict(rank=r,logical_step=300) for r in range(8)],checkpoint_pause_identity=workers)
    (run/'final300-state-roundtrip.json').write_text(json.dumps(receipt))
    return run,config_hash


def test_all_fifteen_control_identities_and_final_roundtrip_accepted(monkeypatch,tmp_path):
    run,config_hash=full_metadata_fixture(monkeypatch,tmp_path)
    assert audit.audit_gate(run,300,'source',config_hash)==run/'checkpoints/actor/global_step_300'


@pytest.mark.parametrize('bad',['zero-gradient','missing-norm','zero-parameter-delta','pid-reuse','forged-final-embedded-manifest'])
def test_full_acceptance_refuses_false_update_or_forged_roundtrip(monkeypatch,tmp_path,bad):
    run,config_hash=full_metadata_fixture(monkeypatch,tmp_path)
    if bad in ('zero-gradient','missing-norm','zero-parameter-delta'):
        path=run/'audit/0.jsonl';records=audit.rows(path)
        if bad=='zero-gradient':records[10]['grad_nonzero']=19;records[10]['grad_norms'][0]=0.
        elif bad=='missing-norm':records[10]['grad_norms'].pop()
        else:records[10]['shard_delta']['nonzero']=0
        path.write_text('\n'.join(json.dumps(r) for r in records)+'\n')
    else:
        path=run/'final300-state-roundtrip.json';record=json.loads(path.read_text())
        if bad=='pid-reuse':record['checkpoint_pause_identity'][0]['start_ticks']='wrong-start'
        else:
            record['manifest']['identity']['files'][1]['pause_identity']['start_ticks']='forged-start'
            record['checkpoint_pause_identity'][0]['start_ticks']='forged-start'
        path.write_text(json.dumps(record))
    with pytest.raises(RuntimeError):audit.audit_gate(run,300,'source',config_hash)



def test_supervisor_uses_new85_gate_but_restores80():
    source = Path(__file__).with_name('formal_resume_v1.py').read_text()
    assert "ready=run/'resume85-ready.json'" in source
    assert "run/'resume85-approval.json'" in source
    assert "run/'checkpoints/actor/global_step_85'" in source
    assert 'checkpoint85-cpu-verification' in source
    assert 'resume80-ready.json' not in source and 'resume80-approval.json' not in source
    assert 'start_logical=80,stop_logical=300,new_actual_budget=220' in source
