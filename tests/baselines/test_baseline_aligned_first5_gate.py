"""CPU receipt fixtures only; these simulated files grant no production release."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys
import pytest
import torch
from search_r1.baselines import fresh200_gate as gate
from search_r1.baselines.full_state import state_digest
from search_r1.baselines.fresh200_driver import Fresh200Progress
from search_r1.baselines.fresh200_contract import critical_source_hashes


def build(tmp_path,real_em=False):
    run=tmp_path/'runs'/'aligned-fixture';directory=run/'fresh200_state'/'outer_5_COMPLETE';directory.mkdir(parents=True)
    source='a'*40;pid=123;progress=Fresh200Progress(baseline_aligned=True);attempts=[];events=[];paths=[]
    for outer in range(1,6):
        attempt_id=progress.begin_attempt(outer)
        ranks=[dict(rank=rank,baseline_aligned=True,actual_outer=outer,attempt_id=attempt_id,status='COMPLETE',
            optimizer_steps=20,expected_optimizer_steps=20,optimizer_opportunities_total=20,
            processed_opportunities=20,skipped_opportunities=0,unprocessed_opportunities=0,
            update_complete=True,attempt_complete=True,postabort=False,policy_monitor=[dict(stage=stage,
                denominator_role='frozen_hf_old_policy',diagnostic_only=True,monitoring_scope='full_frozen_rollout',
                sampling_policy_kl_status='unavailable_on_retokenized_training_actions',
                raw_sampler_diagnostic=dict(diagnostic_only=True,available=False,unavailable_reason='CPU_FIXTURE'))
                for stage in ('before_adam','after_adam')])for rank in range(8)]
        progress.record(ranks,outer)
        for event in ranks:
            rank_path=run/'native-events'/f"rank_{event['rank']}-outer_{outer}-attempt_{outer}.json"
            rank_path.parent.mkdir(exist_ok=True);rank_path.write_text(json.dumps(event));paths.append(rank_path)
        path=run/'diagnostics'/f'attempt_{attempt_id}'/'all-rank-write-ledger.json';path.parent.mkdir(parents=True)
        attempt=dict(attempt_id=attempt_id,candidate_outer=outer,driver_pid=pid,source_commit=source,status='COMPLETE',
            accepted_complete_outer=True,rank_write_ledger=ranks,progress=progress.state_dict(),ledger_path=str(path))
        path.write_text(json.dumps(attempt));paths.append(path);attempts.append(attempt)
        metrics={'actor/optimizer_steps':20,'bapo/actual_outer_updates':outer,'train_em/actual_outer':outer}
        events.append(dict(step=outer,attempt_id=outer,accepted_complete_outer=True,metrics=metrics))
        audit=run/'training-em-audit'/f'outer_{outer:04d}-attempt_{outer:04d}.jsonl';audit.parent.mkdir(exist_ok=True)
        audit.write_text(json.dumps(dict(cpu_fixture=True,metrics={'train_em/actual_outer':outer})));paths.append(audit)
    source_file=run/'source-commit.txt';source_file.write_text(source);paths.append(source_file)
    claim=run/'.aligned-startup-claim.json';claim.write_text(json.dumps(dict(source_commit=source,launcher_pid=42,scope='new-run-only-no-resume')));paths.append(claim)
    holdout=tmp_path/'data-manifest.json';holdout.write_text('{}')
    retriever=tmp_path/'retrieval-manifest.json';retriever.write_text('{}')
    order=tmp_path/'order-manifest.json';order.write_text('{}')
    config=dict(baseline_aligned=True,baselines=dict(alignment=dict(order_manifest=str(order),order_sha256=gate.digest(order)),holdout=dict(manifest_path=str(holdout),manifest_sha256=gate.digest(holdout))),
        retriever=dict(identity_manifest=str(retriever),identity_manifest_sha256=gate.digest(retriever)),
        actor_rollout_ref=dict(model=dict(path=str(tmp_path/'official-model'))))
    em_verifier=verify_fixture_em
    if real_em:
        import runpy
        import numpy as np
        qa_em=runpy.run_path(str(Path(__file__).resolve().parents[2]/'verl/utils/reward_score/qa_em.py'))
        with patch.dict(sys.modules,{'verl.utils.reward_score.qa_em':SimpleNamespace(em_check=qa_em['em_check'])}):
            em=runpy.run_path(str(Path(__file__).resolve().parents[2]/'search_r1/baselines/training_em.py'))
        em_verifier=em['verify_training_em_audit']
        class Batch(SimpleNamespace):
            def __len__(self):return self.batch['responses'].shape[0]
        tokenizer=SimpleNamespace(decode=lambda ids:'<answer>Paris</answer>' if list(ids) else '')
        ids=torch.ones(2560,1,dtype=torch.long)
        output=Batch(batch=dict(responses=ids,attention_mask=ids,info_mask=ids),
            non_tensor_batch={'trajectory_uid':np.array([str(i)for i in range(2560)],dtype=object)})
        questions=Batch(batch=dict(responses=ids),non_tensor_batch=dict(
            index=np.array([f'q-{i}'for i in range(512)for _ in range(5)],dtype=object),
            reward_model=np.array([{'ground_truth':{'target':['Paris']}}for _ in range(2560)],dtype=object)))
        for attempt,event in zip(attempts,events):
            outer=attempt['candidate_outer']
            binding=dict(data_manifest_path=str(holdout),data_manifest_sha256=gate.digest(holdout),
                retrieval_manifest_path=str(retriever),retrieval_manifest_sha256=gate.digest(retriever),
                order_manifest_path=str(order),order_manifest_sha256=gate.digest(order),common_input_binding=None)
            with patch('os.getpid',return_value=pid):
                journal=em['TrainingEMJournal'](questions,tokenizer,actual_outer=outer,attempt_id=outer,
                    run_root=run,policy_snapshot_id=f'{run}:attempt_{outer}:before_outer_{outer}',
                    source_commit=source,bindings=binding)
            journal.capture(output,list(range(2560)),'initial',0)
            audit=run/'training-em-audit'/f'outer_{outer:04d}-attempt_{outer:04d}.jsonl';audit.unlink()
            metrics,proof=journal.finish(output,list(range(2560)))
            event['metrics'].update(metrics)
    from search_r1.baselines.aligned_outer_commit import stage_evidence
    for attempt,event in zip(attempts,events):
        outer=attempt['candidate_outer']
        stage=stage_evidence(outer,outer,{'bapo/stage_evaluated_count':64,'bapo/stage_validation':1.},bindings=dict(
            run_root=str(run),driver_pid=pid,source_commit=source,source_sha256=critical_source_hashes(),
            common_input_binding=None,stage_input_binding=None,order_manifest=str(order),order_manifest_sha256=gate.digest(order)))
        prepared=[]
        for rank_event in attempt['rank_write_ledger']:
            rank=rank_event['rank'];prior=copy.deepcopy(rank_event)
            prior.update(outer_commit_phase='WRITES_PREPARED',accepted_complete_outer=False,scheduler_before=outer-1,scheduler_after=outer-1)
            prepared.append(prior)
            prior_path=run/'native-events'/f'rank_{rank}-outer_{outer}-attempt_{outer}.prepared.json'
            prior_path.write_text(json.dumps(prior));paths.append(prior_path)
            rank_event.update(outer_commit_phase='STAGE_COMMITTED',accepted_complete_outer=True,
                scheduler_before=outer-1,scheduler_after=outer,stage_validation=stage,
                scheduler_lr_before=[1e-6*outer/57],scheduler_lr_after=[1e-6*(outer+1)/57])
            final_path=run/'native-events'/f'rank_{rank}-outer_{outer}-attempt_{outer}.json'
            final_path.write_text(json.dumps(rank_event))
        Path(attempt['ledger_path']).write_text(json.dumps(attempt))
        event['metrics'].update({'actor/scheduler_outer_steps':outer,'bapo/stage_evaluated_count':64,
            'bapo/stage_validation':1.,'fresh200/stage_proof_sha256':stage['sha256']})
        for scope,document in (('aligned-stage-proof',stage),('aligned-outer-prepared',dict(
                status='WRITES_PREPARED_NOT_ACCEPTED_OUTER',actual_outer=outer,attempt_id=outer,driver_pid=pid,
                source_commit=source,previous_completed_outer=outer-1,rank_write_ledger=prepared))):
            target=run/scope/f'outer_{outer:04d}-attempt_{outer:04d}.json';target.parent.mkdir(exist_ok=True)
            target.write_text(json.dumps(document));paths.append(target)
    bindings=gate.write_aligned_step5_bindings(directory,config,source,run,pid,{'cpu_fixture':True});paths.append(bindings)
    driver=dict(driver_pid=pid,global_step=5,next_global_step=6,actual_outer_updates=5,
        progress=progress.state_dict(),rank_write_ledger=attempts[-1]['rank_write_ledger'])
    driver_path=directory/'driver.pt';torch.save(driver,driver_path);paths.append(driver_path)
    driver_report=directory/'driver-reload.json'
    driver_report.write_text(json.dumps(dict(status='EXACT_DRIVER_RELOAD_OK',step=5,path=str(driver_path),sha256=gate.digest(driver_path),pid=pid,state_digest=state_digest(driver))))
    paths.append(driver_report);reports=[]
    official=run/'initial'/'native-initial-official-hf.pt';official.parent.mkdir();official.write_text('cpu fixture')
    alignment=dict(path=str(official),sha256=gate.digest(official),official_model_root=config['actor_rollout_ref']['model']['path'])
    alignment_path=official.with_name('native-initial-official-alignment.json');alignment_path.write_text(json.dumps(alignment));paths.extend([official,alignment_path])
    for rank in range(8):
        path=directory/'ranks'/f'rank_{rank}.pt';path.parent.mkdir(exist_ok=True);torch.save({'cpu_fixture':True},path)
        report=dict(rank=rank,status='EXACT_FULL_STATE_RELOAD_OK',step=5,world_size=8,scheduler_last_epoch=5,
            model_state_dict_type='SHARDED_STATE_DICT',model_dtensor_entries=1,optimizer_state_entries=1,
            path=str(path),sha256=gate.digest(path),pid=1000+rank,state_digest='b'*64,
            native_state_version=3,roundtrip=True,aligned_pending_outer=None,reloaded_state_digest='b'*64,
            generation_rng_before_digest='c'*64,generation_rng_after_digest='c'*64,
            latest_rank_event=attempts[-1]['rank_write_ledger'][rank])
        report_path=path.with_suffix('.json');report_path.write_text(json.dumps(report));paths.extend([path,report_path]);reports.append(report)
        initial=official.parent/f'native-initial-rank_{rank}.pt';initial.write_text('cpu fixture')
        initial_report=initial.with_suffix('.json');initial_report.write_text(json.dumps(dict(path=str(initial),sha256=gate.digest(initial),
            official_alignment=alignment,source_sha256=critical_source_hashes(),
            official_model_manifest_sha256='7f53ce6c010ac076b1ec482031bea7c9e6b8d0a096a61b880fe8ad4971bc9e44',
            optimizer_state_entries=0,scheduler_last_epoch=0,lr=[1e-6/57],status='ACTUAL_INITIAL_STATE_CAPTURED_NOT_RELEASE',
            native_process=dict(rank=rank,world_size=8,pid=1000+rank,run_root=str(run)))))
        paths.extend([initial,initial_report])
    with patch.object(gate,'validate_gate_config',return_value=(200,57)):
        manifest=gate.write_manifest(paths,directory,config,source,run,5,reports,progress.state_dict())
    prefix=''.join('FRESH200_ATTEMPT_JSON='+json.dumps(a)+'\nBAPO_TRAIN_METRICS_JSON='+json.dumps(e)+'\n'for a,e in zip(attempts,events))
    (run/'train.log').write_text(prefix);snapshot=tmp_path/'prefix.log';snapshot.write_text(prefix)
    review=tmp_path/'independent-review.json';review.write_text('{"scope":"CPU_FIXTURE_NOT_PRODUCTION"}')
    m=json.loads(manifest.read_text())
    evidence=m['files']+[dict(path=str(manifest),sha256=gate.digest(manifest)),dict(path=str(snapshot),sha256=gate.digest(snapshot))]
    receipt=dict(m,status='PASS_INDEPENDENT_BASELINE_ALIGNED_FRESH200_STEP5_REVIEW',checkpoint_manifest_sha256=gate.digest(manifest),
        evidence=evidence,independent_review=dict(path=str(review),sha256=gate.digest(review)),
        train_log_prefix=dict(live_path=str(run/'train.log'),snapshot_path=str(snapshot),bytes=snapshot.stat().st_size,sha256=gate.digest(snapshot)))
    receipt_path=tmp_path/'release.json';receipt_path.write_text(json.dumps(receipt))
    return SimpleNamespace(run=run,manifest=manifest,receipt=receipt,receipt_path=receipt_path,source=source,pid=pid,progress=progress,attempts=attempts,em_verifier=em_verifier)


def verify_fixture_em(path,expected_sha256,expected):
    assert gate.digest(path)==expected_sha256
    assert expected['expected_questions']==512 and expected['siblings']==5
    assert expected['driver_pid']==123 and expected['source_commit']=='a'*40
    return {'metrics':json.loads(Path(path).read_text())['metrics']}


def validate(fixture,tmp_path):
    with patch.object(gate,'validate_gate_config',return_value=(200,57)),\
         patch('search_r1.baselines.baseline_alignment.prepare_aligned_inputs',return_value={'cpu_fixture':True}),\
         patch('search_r1.baselines.fresh200_native_evidence.verify_native_process'),\
         patch.dict(sys.modules,{'search_r1.baselines.training_em':SimpleNamespace(verify_training_em_audit=fixture.em_verifier),
             'omegaconf':SimpleNamespace(OmegaConf=SimpleNamespace(create=lambda config:config))}),\
         patch.object(gate.subprocess,'check_output',side_effect=lambda args,**kw:fixture.source+'\n'if 'rev-parse'in args else ''),\
         patch.object(gate,'validate_first_sampling_evidence',side_effect=AssertionError('Legacy LP gate called for aligned profile')):
        return gate.validate_release(fixture.receipt_path,fixture.manifest,tmp_path,expected_driver_pid=fixture.pid)


def test_aligned_own_protocol_release_keeps_original_process_and_fixed20(tmp_path):
    fixture=build(tmp_path);receipt=validate(fixture,tmp_path)
    assert receipt['protocol']=='BAPO_BASELINE_ALIGNED_FRESH200_V1'
    assert receipt['mini_batch_switch_outer'] is None
    assert receipt['optimizer_opportunities_after_switch']==20
    assert receipt['attempt_accounting']['actual_adam_writes_per_rank']==[100]*8


@pytest.mark.parametrize('field,value',[('pause_nonce','forged'),('protocol','BAPO_FRESH200_V1'),
    ('optimizer_opportunities_after_switch',10),('status','PASS_INDEPENDENT_FRESH200_STEP5_REVIEW')])
def test_aligned_cannot_reuse_legacy_release_or_other_pause(tmp_path,field,value):
    fixture=build(tmp_path);fixture.receipt[field]=value;fixture.receipt_path.write_text(json.dumps(fixture.receipt))
    with pytest.raises(ValueError):validate(fixture,tmp_path)


def test_aligned_exactfive_replay_rejects_legacy_profile_and_skips(tmp_path):
    fixture=build(tmp_path)
    assert len(gate.validate_first_five_attempts(fixture.attempts,fixture.progress.state_dict(),123,True))==5
    with pytest.raises(ValueError):gate.validate_first_five_attempts(fixture.attempts,fixture.progress.state_dict(),123)
    changed=copy.deepcopy(fixture.attempts);changed[0]['rank_write_ledger'][0]['optimizer_steps']=19
    with pytest.raises(ValueError):gate.validate_first_five_attempts(changed,fixture.progress.state_dict(),123,True)


def test_aligned_release_rejects_changed_em_inputs_and_driver_pid(tmp_path):
    fixture=build(tmp_path)
    fixture.pid=456
    with pytest.raises(ValueError,match='reload not proven'):validate(fixture,tmp_path)
    fixture.pid=123
    audit=fixture.run/'training-em-audit'/'outer_0003-attempt_0003.jsonl';audit.write_text('changed')
    with pytest.raises(ValueError,match='aligned first-five evidence'):validate(fixture,tmp_path)


def test_actual_pause_method_collects_aligned_audits_and_never_legacy_sampling(tmp_path):
    import ast
    import os
    from search_r1.baselines import fresh200_gate
    fixture=build(tmp_path)
    root=Path(__file__).resolve().parents[2]
    tree=ast.parse((root/'verl/trainer/ppo/ray_trainer.py').read_text())
    method=next(n for cls in tree.body if isinstance(cls,ast.ClassDef) for n in cls.body
                if isinstance(n,ast.FunctionDef) and n.name=='_fresh200_pause_after_five')
    class Attributes(dict):
        __getattr__=dict.__getitem__
    config=Attributes(baseline_aligned=True,baselines=Attributes(algorithm_source_commit=fixture.source),
        trainer=Attributes(default_local_dir=str(fixture.run),fresh200_gate_receipt=str(tmp_path/'receipt')),
        actor_rollout_ref=Attributes(actor=Attributes(fresh200_fault_dir=str(fixture.run/'initial'))))
    namespace={'os':os,'OmegaConf':SimpleNamespace(to_container=lambda config,resolve:dict(config))}
    exec(compile(ast.Module(body=[method],type_ignores=[]),'ray_trainer.py','exec'),namespace)
    trainer=SimpleNamespace(config=config,fresh200_state_directory=fixture.manifest.parent,
        fresh200_state_reports=[],fresh200_progress=fixture.progress,fresh200_shared_input_bindings={'fixture':True})
    with patch.dict(os.environ,{'BAPO_RUN_ID':str(fixture.run)}),\
         patch('subprocess.check_output',return_value=fixture.source+'\n'),\
         patch.object(fresh200_gate,'write_aligned_step5_bindings',return_value=fixture.manifest.parent/'aligned-step5-bindings.json'),\
         patch.object(fresh200_gate,'write_manifest',return_value=fixture.manifest)as writer,\
         patch.object(fresh200_gate,'wait_for_release')as wait:
        namespace['_fresh200_pause_after_five'](trainer)
    files=writer.call_args.args[0]
    assert len([p for p in files if p.parent.name=='training-em-audit'])==5
    assert fixture.run/'.aligned-startup-claim.json' in files
    assert not any('formal-first-sampling-batch' in p.name for p in files)
    wait.assert_called_once()


def test_aligned_release_detects_input_order_byte_changes(tmp_path):
    fixture=build(tmp_path)
    bindings=json.loads((fixture.manifest.parent/'aligned-step5-bindings.json').read_text())
    Path(bindings['inputs']['order_manifest']['path']).write_text('changed order')
    with pytest.raises(ValueError,match='order_manifest'):validate(fixture,tmp_path)


def test_actual_training_em_producer_and_replay_helper_accept_exact_gate_bindings(tmp_path):
    fixture=build(tmp_path,real_em=True)
    receipt=validate(fixture,tmp_path)
    assert receipt['attempt_accounting']['completed_outers']==5
    path=fixture.run/'training-em-audit'/'outer_0001-attempt_0001.jsonl'
    header=json.loads(path.read_text().splitlines()[0])
    assert 'order_manifest_path' in header['bindings']
    assert header['expected_questions']==512 and header['siblings']==5


@pytest.mark.parametrize('change', ['prepared_only','wrong_lr','bool_stage_score','fractional_stage_count','outside_em_score'])
def test_aligned_evidence_rejects_prepared_only_wrong_lr_and_malformed_stage(tmp_path,change):
    fixture=build(tmp_path)
    manifest=json.loads(fixture.manifest.read_text())
    members={item['path']:item['sha256']for item in manifest['files']}
    config=json.loads((fixture.manifest.parent/'resolved-config.json').read_text())
    events=[json.loads(line.split('=',1)[1])for line in (fixture.run/'train.log').read_text().splitlines()
            if line.startswith('BAPO_TRAIN_METRICS_JSON=')]
    if change=='prepared_only':
        fixture.attempts[0]['rank_write_ledger'][0]['outer_commit_phase']='WRITES_PREPARED'
    elif change=='wrong_lr':
        fixture.attempts[0]['rank_write_ledger'][0]['scheduler_lr_after']=[1e-6]
    else:
        from search_r1.baselines.aligned_outer_commit import _digest
        path=fixture.run/'aligned-stage-proof'/'outer_0001-attempt_0001.json'
        stage=json.loads(path.read_text())
        if change=='bool_stage_score':stage['score']=True;events[0]['metrics']['bapo/stage_validation']=True
        elif change=='fractional_stage_count':stage['evaluated_count']=64.
        else:stage['score']=1.1;events[0]['metrics']['bapo/stage_validation']=1.1
        stage['sha256']=_digest({k:v for k,v in stage.items()if k!='sha256'})
        path.write_text(json.dumps(stage));members[str(path)]=gate.digest(path)
        events[0]['metrics']['fresh200/stage_proof_sha256']=stage['sha256']
    with patch('search_r1.baselines.baseline_alignment.prepare_aligned_inputs',return_value={'cpu_fixture':True}),\
         patch('search_r1.baselines.fresh200_native_evidence.verify_native_process'),\
         patch.dict(sys.modules,{'search_r1.baselines.training_em':SimpleNamespace(verify_training_em_audit=verify_fixture_em),
             'omegaconf':SimpleNamespace(OmegaConf=SimpleNamespace(create=lambda config:config))}),\
         pytest.raises(ValueError,match='stage|warmup57'):
        gate.validate_aligned_first_five_evidence(members,config,fixture.source,fixture.run,fixture.pid,
            manifest['rank_roundtrips'],fixture.attempts,events)


@pytest.mark.parametrize('pending',["missing", {"actual_outer":5}])
def test_aligned_release_rejects_pending_native_state(tmp_path,pending):
    fixture=build(tmp_path)
    manifest=json.loads(fixture.manifest.read_text())
    report=manifest['rank_roundtrips'][0]
    if pending=="missing":report.pop('aligned_pending_outer')
    else:report['aligned_pending_outer']=pending
    report_path=Path(report['path']).with_suffix('.json')
    report_path.write_text(json.dumps(report))
    for item in manifest['files']:
        if item['path']==str(report_path):item['sha256']=gate.digest(report_path)
    fixture.manifest.write_text(json.dumps(manifest))
    fixture.receipt['checkpoint_manifest_sha256']=gate.digest(fixture.manifest)
    fixture.receipt_path.write_text(json.dumps(fixture.receipt))
    with pytest.raises(ValueError,match='pending outer'):
        validate(fixture,tmp_path)
