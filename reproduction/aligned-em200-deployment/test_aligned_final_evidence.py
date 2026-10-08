"""Code/CPU fixtures only; no formal job, terminal observation or GPU acceptance."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import pytest

HERE=Path(__file__).parent;sys.path.insert(0,str(HERE))
spec=importlib.util.spec_from_file_location('final_candidate',HERE/'aligned_final_evidence.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)


def submission(root=h.ROOT):
    token='b'*32;job='9999'
    return dict(source=h.SOURCE,root=str(root),job=job,token=token,run=str(root/'runs'/f'formal-aligned-em200-{token}-{job}'),
        comment='BAPO_ALIGNED_EM200:formal:'+h.SOURCE+':'+token,scope='ACTUAL_SINGLE_FORMAL_JOB_SUBMITTED_NOT_TRAINING_COMPLETION')


def scheduler(submit,**changes):
    row=dict(JobId=submit['job'],JobName='bapo-aligned-em200-formal',Comment=submit['comment'],WorkDir=submit['root'],ReqNodeList='h100-5',NodeList='h100-5',
        JobState='COMPLETED',ExitCode='0:0',SubmitTime='2026-10-09T01:00:00',StartTime='2026-10-09T01:01:00',EndTime='2026-10-09T02:00:00',
        Command=submit['root']+'/tools/deployment/formal.sbatch '+submit['token'])
    row.update(changes)
    return dict(exit_code=0,stdout=' '.join(k+'='+v for k,v in row.items()),stderr='')


def test_actual_terminal_and_pending_no_false_terminal():
    s=submission();value=h.terminal_observation(s,lambda argv:scheduler(s))
    assert value['terminal_verified']and value['completed_successfully']and not value['acceptance']
    value=h.terminal_observation(s,lambda argv:scheduler(s,JobState='PENDING',EndTime='Unknown',StartTime='Unknown'))
    assert not value['terminal_verified']
    value=h.terminal_observation(s,lambda argv:dict(exit_code=1,stdout='',stderr='job unavailable'))
    assert not value['terminal_verified']


@pytest.mark.parametrize('changes',[dict(JobId='3315'),dict(Comment='old'),dict(WorkDir='/old'),dict(Command='/foreign'),dict(NodeList='foreign'),
    dict(EndTime='Unknown'),dict(EndTime='2026-10-09T00:00:00'),dict(StartTime='2026-10-09T03:00:00'),dict(ExitCode='invalid')])
def test_terminal_strong_identity_timestamp_refusal(changes):
    s=submission()
    with pytest.raises(ValueError):h.terminal_observation(s,lambda argv:scheduler(s,**changes))


def metrics(counts):
    result={'val/qa_count/all':51713}
    for source,count in counts.items():
        result['val/evaluated_count/'+source]=count;result['val/qa_count/'+source]=count
        for key in ('em','f1','answer_format','format_rate','generated_tokens','search_calls'):result['val/'+key+'/'+source]=.5
    for key in ('em','f1','answer_format','format_rate','generated_tokens','search_calls'):result['val/'+key+'/all']=.5
    return result


def test_complete_final_aggregate_no_invented_answer_replay():
    from search_r1.baselines.holdout import E1_FINAL_COUNTS
    m=metrics(E1_FINAL_COUNTS);result=h.final_aggregate(m,E1_FINAL_COUNTS)
    assert result['evaluated_count']==51713 and 'NO_PER_QUESTION' in result['scope']
    for key,value in [('val/evaluated_count/nq',3609),('val/em/all',.6),('val/f1/nq',float('nan'))]:
        bad=dict(m);bad[key]=value
        with pytest.raises(ValueError):h.final_aggregate(bad,E1_FINAL_COUNTS)


def fixture200(root):
    from search_r1.baselines.fresh200_driver import Fresh200Progress
    from search_r1.baselines.fresh200_contract import fresh200_lr_factor
    from search_r1.baselines.aligned_outer_commit import _digest
    run=root/'run';fault=run/'diagnostics';pid=123;source_hashes={'actual_source_fixture':'d'*64}
    config={'baselines':{'holdout':{'manifest_path':'/data','manifest_sha256':'a'*64},'alignment':{'order_manifest':'/order','order_sha256':'b'*64}},
        'retriever':{'identity_manifest':'/retrieval','identity_manifest_sha256':'c'*64},'actor_rollout_ref':{'actor':{'fresh200_fault_dir':str(fault)}}}
    shared={'common_input_binding':{'cohort':169028},'stage':{'count':64}}
    progress=Fresh200Progress(baseline_aligned=True);attempts=[];events=[];files={};calls=[]
    for outer in range(1,201):
        progress.begin_attempt(outer)
        stage=dict(actual_outer=outer,attempt_id=outer,evaluated_count=64,score=.5,driver_pid=pid,run_root=str(run),source_commit=h.SOURCE,
            source_sha256=source_hashes,common_input_binding=shared['common_input_binding'],stage_input_binding=shared['stage'],order_manifest='/order',order_manifest_sha256='b'*64)
        stage['sha256']=_digest(stage);files[str(run/'aligned-stage-proof'/f'outer_{outer:04d}-attempt_{outer:04d}.json')]=stage
        ranks=[];prior=[]
        for rank in range(8):
            event=dict(rank=rank,actual_outer=outer,attempt_id=outer,baseline_aligned=True,status='COMPLETE',optimizer_steps=20,expected_optimizer_steps=20,
                optimizer_opportunities_total=20,processed_opportunities=20,skipped_opportunities=0,unprocessed_opportunities=0,postabort=False,
                attempt_complete=True,update_complete=True,outer_commit_phase='STAGE_COMMITTED',accepted_complete_outer=True,stage_validation=stage,
                scheduler_before=outer-1,scheduler_after=outer,scheduler_lr_before=[1e-6*fresh200_lr_factor(outer-1)],scheduler_lr_after=[1e-6*fresh200_lr_factor(outer)],
                policy_monitor=[dict(stage=phase,denominator_role='frozen_hf_old_policy',diagnostic_only=True,monitoring_scope='full_frozen_rollout')for phase in ('before_adam','after_adam')])
            p=copy.deepcopy(event);p.update(outer_commit_phase='WRITES_PREPARED',accepted_complete_outer=False,scheduler_after=outer-1);p.pop('stage_validation');p.pop('scheduler_lr_after')
            ranks.append(event);prior.append(p)
            files[str(fault/f'rank_{rank}-outer_{outer}-attempt_{outer}.json')]=event
            files[str(fault/f'rank_{rank}-outer_{outer}-attempt_{outer}.prepared.json')]=p
        progress.record(ranks,outer)
        ledger=run/'checkpoints/diagnostics'/f'attempt_{outer}_candidate_outer_{outer}/all-rank-write-ledger.json'
        attempt=dict(attempt_id=outer,candidate_outer=outer,driver_pid=pid,source_commit=h.SOURCE,status='COMPLETE',accepted_complete_outer=True,
            ledger_path=str(ledger),rank_write_ledger=ranks,progress=progress.state_dict())
        files[str(ledger)]=attempt;attempts.append(attempt)
        files[str(run/'aligned-outer-prepared'/f'outer_{outer:04d}-attempt_{outer:04d}.json')]=dict(status='WRITES_PREPARED_NOT_ACCEPTED_OUTER',actual_outer=outer,attempt_id=outer,
            driver_pid=pid,source_commit=h.SOURCE,previous_completed_outer=outer-1,rank_write_ledger=prior)
        events.append(dict(step=outer,attempt_id=outer,accepted_complete_outer=True,metrics={'actor/optimizer_steps':20,'actor/scheduler_outer_steps':outer,
            'bapo/actual_outer_updates':outer,'bapo/stage_evaluated_count':64,'bapo/stage_validation':.5,'fresh200/stage_proof_sha256':stage['sha256'],'train_em/fixture':outer}))
    def em(path,expected):
        calls.append(expected)
        return {'metrics':{'train_em/fixture':expected['actual_outer']}}
    return dict(attempts=attempts,events=events,run=run,config=config,driver_pid=pid,read_record=lambda p:files[str(p)],verify_em=em,
        shared_inputs=shared,source_hashes=source_hashes),files,calls


def test_full200_real_progress_stage_ledger_replay_not_mocked_counter(tmp_path):
    arguments,files,calls=fixture200(tmp_path)
    report=h.replay_accounting(**arguments)
    assert report['progress']['actual_adam_writes_per_rank']==[4000]*8
    assert report['progress']['actual_adam_writes_all_ranks']==32000 and len(calls)==200
    assert calls[-1]['policy_snapshot_id']==str(arguments['run'])+':attempt_200:before_outer_200'
    assert calls[-1]['expected_questions']==512 and calls[-1]['siblings']==5
    # EM execution is a spy in this200-loop fixture, explicitly not200 actual audits.


@pytest.mark.parametrize('broken',['legacy_switch','skip','wrong_stage','scheduler_early','wrong_em','PID_changed','missing_rank'])
def test_full200_rejects_actual_post50_faults(tmp_path,broken):
    args,files,calls=fixture200(tmp_path);attempt=args['attempts'][149];rank=attempt['rank_write_ledger'][0]
    if broken=='legacy_switch':rank.update(optimizer_steps=10,expected_optimizer_steps=10)
    if broken=='skip':rank.update(skipped_opportunities=1,optimizer_steps=19)
    if broken=='wrong_stage':rank['stage_validation']={'score':.5}
    if broken=='scheduler_early':rank['scheduler_before']=150
    if broken=='wrong_em':args['events'][149]['metrics']['train_em/fixture']=0
    if broken=='PID_changed':attempt['driver_pid']=456
    if broken=='missing_rank':attempt['rank_write_ledger'].pop()
    with pytest.raises(ValueError):h.replay_accounting(**args)


def test_pending_does_not_open_or_hash_run_files(tmp_path,monkeypatch):
    s=submission();output=tmp_path/'output';output.mkdir()
    touched=[];monkeypatch.setattr(h,'regular',lambda *args:touched.append(args))
    with pytest.raises(ValueError,match='terminal'):h.collect_phase1(s,output,None,lambda argv:scheduler(s,JobState='PENDING',EndTime='Unknown'))
    assert touched==[] and (output/'scheduler-terminal-observation.json').exists()
    assert json.loads((output/'scheduler-terminal-observation.json').read_text())['acceptance']is False


def test_actual512x5_authored_EM_audit_replayed_in_full200_plumbing(tmp_path):
    import os
    import numpy as np
    import torch
    from types import SimpleNamespace
    from search_r1.baselines.training_em import TrainingEMJournal,verify_training_em_audit
    args,files,calls=fixture200(tmp_path)
    pid=os.getpid();args['driver_pid']=pid
    # Preserve identity across all producer records for this actual-journal fixture.
    for value in files.values():
        if 'driver_pid'in value:value['driver_pid']=pid
    from search_r1.baselines.aligned_outer_commit import _digest
    for outer in range(1,201):
        stage=files[str(args['run']/'aligned-stage-proof'/f'outer_{outer:04d}-attempt_{outer:04d}.json')]
        stage['sha256']=_digest({k:v for k,v in stage.items()if k!='sha256'})
        args['events'][outer-1]['metrics']['fresh200/stage_proof_sha256']=stage['sha256']
    class Batch(SimpleNamespace):
        def __len__(self):return len(self.batch['responses'])
    class Tokenizer:
        def decode(self,ids):return '<answer>Paris</answer>'
    ids=[f'nq:train:{i}'for i in range(512)for _ in range(5)]
    qa=Batch(batch={'responses':torch.ones(2560,1,dtype=torch.long)},non_tensor_batch={'index':np.array(ids,dtype=object),
        'reward_model':np.array([{'ground_truth':{'target':['Paris']}}for _ in ids],dtype=object)})
    binding={'data_manifest_path':'/data','data_manifest_sha256':'a'*64,'retrieval_manifest_path':'/retrieval','retrieval_manifest_sha256':'c'*64,
        'order_manifest_path':'/order','order_manifest_sha256':'b'*64,'common_input_binding':args['shared_inputs']['common_input_binding']}
    journal=TrainingEMJournal(qa,Tokenizer(),actual_outer=1,attempt_id=1,run_root=args['run'],policy_snapshot_id=f"{args['run']}:attempt_1:before_outer_1",
        source_commit=h.SOURCE,bindings=binding,expected_questions=512,siblings=5)
    output=Batch(batch={'responses':torch.ones(2560,1,dtype=torch.long),'info_mask':torch.ones(2560,1,dtype=torch.long),
        'attention_mask':torch.ones(2560,1,dtype=torch.long)},non_tensor_batch={'trajectory_uid':np.array([str(i)for i in range(2560)],dtype=object)})
    journal.capture(output,range(2560),'initial',0);metrics,proof=journal.finish(output,range(2560))
    args['events'][0]['metrics'].update(metrics);args['events'][0]['metrics'].pop('train_em/fixture')
    spy=args['verify_em']
    def verifier(path,expected):
        if expected['actual_outer']==1:
            assert str(path)==proof['path']
            return verify_training_em_audit(path,proof['sha256'],expected)
        return spy(path,expected)
    args['verify_em']=verifier
    result=h.replay_accounting(**args)
    actual=result['training_em'][0]['metrics']
    assert actual['train_em/initial/correct_response_count']==2560
    assert actual['train_em/initial/correct_question_count']==512
    assert actual['train_em/retained/correct_response_count']==2560
    assert len(calls)==199 # Only one audit is real; remaining199 are provenance spies.


def test_dev512_requires_all_four_scheduled_outers(tmp_path):
    args,_,_=fixture200(tmp_path)
    for step in (50,100,150,200):args['events'][step-1]['metrics'].update({'val/evaluated_count/nq':256,'val/evaluated_count/hotpotqa':256,'val/qa_count/all':512,'val/qa_count/nq':256,'val/qa_count/hotpotqa':256})
    assert [v['actual_outer']for v in h.dev_schedule(args['events'])]==[50,100,150,200]
    args['events'][149]['metrics']['val/evaluated_count/nq']=255
    with pytest.raises(ValueError,match='dev512'):h.dev_schedule(args['events'])


def test_original_same_PID_release_byte_binding_and_sequence(tmp_path):
    run=tmp_path/'run';folder=run/'checkpoints/fresh200_state/outer_5_COMPLETE';folder.mkdir(parents=True)
    manifest_path=folder/'manifest.json';receipt_path=tmp_path/'release.json';pid=123
    manifest=dict(version=3,baseline_aligned=True,run_root=str(run),source_commit=h.SOURCE,global_step=5,actual_outer_updates=5,horizon=200,warmup_steps=57,
        protocol='BAPO_BASELINE_ALIGNED_FRESH200_V1',update_semantics='COMPLETE20_NO_SKIP_ACCEPTED_OUTERS_V1',mini_batch_switch_outer=None,
        optimizer_opportunities_before_switch=20,optimizer_opportunities_after_switch=20,config_sha256='a'*64,pause_nonce='b'*64,attempt_accounting={'actual':5})
    manifest_path.write_text(json.dumps(manifest))
    receipt={k:v for k,v in manifest.items()if k!='version'}
    receipt.update(status='PASS_INDEPENDENT_BASELINE_ALIGNED_FRESH200_STEP5_REVIEW',checkpoint_manifest_sha256=h.binding(manifest_path)['sha256'])
    receipt_path.write_text(json.dumps(receipt))
    paused={'pid':pid,'manifest_path':str(manifest_path),'manifest_sha256':h.binding(manifest_path)['sha256']}
    released={'pid':pid,'receipt':str(receipt_path),'receipt_sha256':h.binding(receipt_path)['sha256']}
    prefix='FRESH200_STEP5_PAUSED_JSON='+json.dumps(paused)+'\n'
    release_line='FRESH200_STEP5_RELEASED_JSON='+json.dumps(released)+'\n'
    sixth='FRESH200_ATTEMPT_JSON='+json.dumps({'attempt_id':6})+'\n'
    config={'trainer':{'fresh200_gate_receipt':str(receipt_path)}}
    reader=lambda p:json.loads(p.read_text())
    proof=h.fifth_release_binding(prefix+release_line+sixth,config,run,pid,reader)
    assert 'NOT_REISSUED' in proof['scope']
    with pytest.raises(ValueError,match='precedes'):h.fifth_release_binding(prefix+sixth+release_line,config,run,pid,reader)
    receipt_path.write_text(json.dumps({**receipt,'source_commit':'old'}))
    with pytest.raises(ValueError):h.fifth_release_binding(prefix+release_line+sixth,config,run,pid,reader)


def test_resource_pause_recovery_is_retained_not_erased(tmp_path):
    run=tmp_path/'run'
    rows=[dict(run=str(run),slurm_job_id='9999',time=i,pause=p,reason='resource')for i,p in enumerate((False,True,True,False))]
    report=h.resource_metadata(rows,run,'9999')
    assert report['pause_samples']==2 and report['pause_intervals'][0]['recovered']is True
    assert report['pause_intervals'][0]['start_unix_time']==1 and report['pause_intervals'][0]['end_unix_time']==3
    with pytest.raises(ValueError,match='healthy'):h.resource_metadata(rows[:-1],run,'9999')
    bad=copy.deepcopy(rows);bad[1]['pause']=1
    with pytest.raises(ValueError,match='schema'):h.resource_metadata(bad,run,'9999')


@pytest.mark.parametrize('broken',['missing','duplicate','mismatch','before_train','after_final'])
def test_actual_completed_marker_missing_duplicate_mismatch_or_order_refusal(broken):
    progress={'completed_outers':200,'actual_adam_writes_per_rank':[4000]*8}
    train='BAPO_TRAIN_METRICS_JSON={}\n';marker='FRESH200_COMPLETED_RUN_JSON='+json.dumps(progress)+'\n';final='SEARCH_R1_FINAL_METRICS_JSON={}\n'
    assert h.completed_marker(train+marker+final,progress)==progress
    text={'missing':train+final,'duplicate':train+marker+marker+final,'mismatch':train+'FRESH200_COMPLETED_RUN_JSON={}\n'+final,
        'before_train':marker+train+final,'after_final':train+final+marker}[broken]
    with pytest.raises(ValueError):h.completed_marker(text,progress)


def test_completed_null_actual_node_is_not_requested_node_proof():
    s=submission()
    with pytest.raises(ValueError,match='actual node'):h.terminal_observation(s,lambda argv:scheduler(s,NodeList='(null)'))


@pytest.mark.parametrize('bad',[{'status':'ABORTED_CURRENT_STATE_SAVED'},{'roundtrip':True},{'attempt_id':199},{'path':'/old/native.pt'},{'scheduler_last_epoch':200.0}])
def test_native200_metadata_exact_serialized_status_without_invented_reload(tmp_path,bad):
    path=tmp_path/'rank_0.pt';event={'actual_outer':200}
    side=dict(rank=0,step=200,world_size=8,scheduler_last_epoch=200,native_state_version=3,attempt_id=200,status='FULL_STATE_SERIALIZED_OK',
        roundtrip=False,path=str(path),aligned_pending_outer=None,latest_rank_event=event)
    h.native_metadata(side,0,200,event,path)
    with pytest.raises(ValueError):h.native_metadata({**side,**bad},0,200,event,path)
