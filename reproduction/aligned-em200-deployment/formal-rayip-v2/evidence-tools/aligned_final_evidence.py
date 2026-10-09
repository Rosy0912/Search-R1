"""SOURCE080 terminal/raw-accounting candidate, never full acceptance or release.

Phase 1 only: no native deserialization, asset scan, HF load or GPU operation.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import sys
import time
import traceback

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from aligned_first5_cpu_evidence import ROOT,SOURCE,COMMON_SHA,identity,regular,binding,write_new,marked,bind_source,verify_original_bindings,cpu_only
from aligned_first5_math import require
from collect_hardware_evidence import execute,copy_regular,TERMINAL

PENDING=['full_official_12_file_hash','HF_and_native_all_files_at_5_50_100_150_200',
    'actual_driver_and_native_state200_CPU_inspection','real_official_and_each_export_CPU_HF_tokenizer_tied_all_parameters_finite',
    'full_precision_aligned_weight_change_scan_at_each_export','complete_own_cleanup_GPU_exit_UUID_lease_and_resource_validation','fault_or_resume_cross_job_identity_and_retained_partial_write_accounting','independent_final_review']


def terminal_observation(submission,query=execute,root=ROOT):
    who=identity(submission,root)
    raw=query(['scontrol','show','job','-o',who['job']])
    result=dict(raw=raw,terminal_verified=False,acceptance=False)
    if raw.get('exit_code')!=0:return result
    values=dict(re.findall(r'(\w+)=([^\s]*)',raw['stdout']))
    command=re.search(r'(?:^|\s)Command=(.*?)(?=\s+\w+=|$)',raw['stdout'])
    expected_command=str(root/'tools/deployment/formal-rayip-v2/formal.sbatch')+' '+who['token']
    for key,value in dict(JobId=who['job'],JobName='bapo-aligned-em200-formal',Comment=submission['comment'],
            WorkDir=str(root),ReqNodeList='h100-5').items():
        require(values.get(key)==value,'original formal scheduler identity differs: '+key)
    require(command and command.group(1).strip()==expected_command,'original formal scheduler command differs')
    require(values.get('NodeList')in('h100-5','(null)','None'),'foreign formal scheduler node')
    state=values.get('JobState');result['primary']=values
    if state not in TERMINAL:return result
    def timestamp(key):
        value=values.get(key,'')
        require(re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',value),'actual scheduler timestamp required: '+key)
        return datetime.datetime.strptime(value,'%Y-%m-%dT%H:%M:%S')
    submit,end=timestamp('SubmitTime'),timestamp('EndTime')
    require(end>=submit and re.fullmatch(r'\d+:\d+',values.get('ExitCode','')),'terminal timeline/exit differs')
    if state=='COMPLETED':
        require(values.get('NodeList')=='h100-5','successful formal actual node differs')
        require(submit<=timestamp('StartTime')<=end,'completed scheduler timeline differs')
    result['terminal_verified']=True
    result['completed_successfully']=state=='COMPLETED'and values['ExitCode']=='0:0'
    return result


def final_aggregate(metrics,counts):
    require(sum(counts.values())==51713 and len(counts)==7,'actual seven-benchmark counts required')
    actual={key.removeprefix('val/evaluated_count/'):value for key,value in metrics.items()if key.startswith('val/evaluated_count/')}
    require(actual==counts and all(type(v)is int for v in actual.values()),'complete final51713 source coverage differs')
    require(metrics.get('val/qa_count/all')==51713 and type(metrics.get('val/qa_count/all'))is int,'complete QA count differs')
    for source,count in counts.items():require(type(metrics.get('val/qa_count/'+source))is int and metrics.get('val/qa_count/'+source)==count,'QA benchmark count differs')
    for key in ('em','f1','answer_format','format_rate','generated_tokens','search_calls'):
        values=[metrics.get('val/'+key+'/'+source)for source in counts]
        require(all(type(v)in(int,float)and math.isfinite(v)and v>=0 for v in values),'nonfinite/negative final aggregate')
        if key in ('em','f1','answer_format','format_rate'):require(all(v<=1 for v in values),'final rate outside0..1')
        average=sum(count*metrics['val/'+key+'/'+source]for source,count in counts.items())/51713
        require(type(metrics.get('val/'+key+'/all'))in(int,float)and math.isclose(metrics['val/'+key+'/all'],average,rel_tol=1e-12,abs_tol=1e-12),'final count-weighted aggregate differs: '+key)
    return dict(evaluated_count=51713,source_counts=counts,metrics=metrics,
        scope='ORIGINAL_AGGREGATE_CONSISTENCY_ONLY_NO_PER_QUESTION_ANSWER_RECOMPUTATION')


def dev_schedule(events):
    reports=[]
    for event in events:
        counts={k.removeprefix('val/evaluated_count/'):v for k,v in event['metrics'].items()if k.startswith('val/evaluated_count/')}
        if event['step']in(50,100,150,200):
            require(counts=={'nq':256,'hotpotqa':256}and all(type(v)is int for v in counts.values())
                and type(event['metrics'].get('val/qa_count/all'))is int and event['metrics'].get('val/qa_count/all')==512
                and all(type(event['metrics'].get('val/qa_count/'+source))is int and event['metrics']['val/qa_count/'+source]==256 for source in ('nq','hotpotqa')),'scheduled dev512 actual coverage differs')
            reports.append(dict(actual_outer=event['step'],metrics={k:v for k,v in event['metrics'].items()if k.startswith('val/')}))
        else:require(not counts,'unexpected ordinary dev evaluation outer')
    require([r['actual_outer']for r in reports]==[50,100,150,200],'four ordinary dev512 summaries required')
    return reports


def fifth_release_binding(text,config,run,pid,read_record):
    paused=marked(text,'FRESH200_STEP5_PAUSED_JSON=');released=marked(text,'FRESH200_STEP5_RELEASED_JSON=')
    require(len(paused)==len(released)==1 and paused[0].get('pid')==released[0].get('pid')==pid,'original same-PID fifth pause/release required')
    path=run/'checkpoints/fresh200_state/outer_5_COMPLETE/manifest.json';manifest=read_record(path)
    receipt_path=Path(config['trainer']['fresh200_gate_receipt']);receipt=read_record(receipt_path)
    require(paused[0].get('manifest_path')==str(path)and paused[0].get('manifest_sha256')==binding(path)['sha256']
        and released[0].get('receipt')==str(receipt_path)and released[0].get('receipt_sha256')==binding(receipt_path)['sha256']
        and receipt.get('checkpoint_manifest_sha256')==paused[0]['manifest_sha256']
        and receipt.get('status')=='PASS_INDEPENDENT_BASELINE_ALIGNED_FRESH200_STEP5_REVIEW'
        and receipt.get('baseline_aligned')is True and manifest.get('version')==3,'original aligned release receipt bytes differ')
    for key in ('run_root','source_commit','global_step','actual_outer_updates','horizon','warmup_steps','config_sha256','pause_nonce','protocol',
                'mini_batch_switch_outer','update_semantics','attempt_accounting','optimizer_opportunities_before_switch','optimizer_opportunities_after_switch'):
        require(receipt.get(key)==manifest.get(key),'original aligned release metadata differs: '+key)
    require(manifest.get('run_root')==str(run)and manifest.get('source_commit')==SOURCE and manifest.get('global_step')==5
        and manifest.get('actual_outer_updates')==5 and manifest.get('baseline_aligned')is True
        and manifest.get('protocol')=='BAPO_BASELINE_ALIGNED_FRESH200_V1'and manifest.get('update_semantics')=='COMPLETE20_NO_SKIP_ACCEPTED_OUTERS_V1'
        and manifest.get('horizon')==200 and manifest.get('warmup_steps')==57 and manifest.get('mini_batch_switch_outer')is None
        and manifest.get('optimizer_opportunities_before_switch')==manifest.get('optimizer_opportunities_after_switch')==20,'original source080 fixed20 release profile differs')
    lines=text.splitlines();pause_index=next(i for i,l in enumerate(lines)if 'FRESH200_STEP5_PAUSED_JSON='in l)
    release_index=next(i for i,l in enumerate(lines)if 'FRESH200_STEP5_RELEASED_JSON='in l)
    sixth_index=next(i for i,l in enumerate(lines)if 'FRESH200_ATTEMPT_JSON='in l and json.loads(l.split('FRESH200_ATTEMPT_JSON=',1)[1]).get('attempt_id')==6)
    require(pause_index<release_index<sixth_index,'sixth attempt precedes original fifth release')
    return dict(paused=paused[0],released=released[0],manifest=binding(path),receipt=binding(receipt_path),
        scope='ORIGINAL_RELEASE_RECORD_BYTES_AND_SEQUENCE_ONLY_NOT_REISSUED_OR_FULL_GATE_REVALIDATED')


def resource_metadata(rows,run,job):
    require(rows and all(row.get('run')==str(run)and row.get('slurm_job_id')==job and type(row.get('pause'))is bool
        and type(row.get('time'))in(int,float)and math.isfinite(row['time'])for row in rows),'original own resource observation schema differs')
    require(all(a['time']<=b['time']for a,b in zip(rows,rows[1:])),'resource observation timeline differs')
    intervals=[];active=None
    for row in rows:
        if row['pause']and active is None:active={'start_unix_time':row['time'],'reasons':[]}
        if row['pause']:active['reasons'].append(row.get('reason'))
        elif active is not None:
            active.update(end_unix_time=row['time'],recovered=True);intervals.append(active);active=None
    if active is not None:active.update(end_unix_time=None,recovered=False);intervals.append(active)
    require(rows[-1]['pause']is False,'last original resource observation is not healthy')
    return dict(samples=len(rows),first_unix_time=rows[0]['time'],last_unix_time=rows[-1]['time'],
        pause_samples=sum(row['pause']for row in rows),pause_intervals=intervals,last_observation_healthy=True,
        scope='RAW_OWN_WATCHER_METADATA_ONLY_COMPLETE_UUID_GPU_EXIT_LEASE_CLEANUP_REVIEW_PENDING')


def completed_marker(text,progress):
    records=marked(text,'FRESH200_COMPLETED_RUN_JSON=')
    require(len(records)==1 and records[0]==progress,'unique actual completed200 producer marker differs')
    lines=text.splitlines()
    completed=next(i for i,line in enumerate(lines)if 'FRESH200_COMPLETED_RUN_JSON='in line)
    train=[i for i,line in enumerate(lines)if 'BAPO_TRAIN_METRICS_JSON='in line]
    final=[i for i,line in enumerate(lines)if 'SEARCH_R1_FINAL_METRICS_JSON='in line]
    require(train and len(final)==1 and train[-1]<completed<final[0],'completed200 marker sequence differs')
    return records[0]


def replay_accounting(attempts,events,run,config,driver_pid,read_record,verify_em,shared_inputs,source_hashes):
    from search_r1.baselines.fresh200_driver import Fresh200Progress
    from search_r1.baselines.fresh200_contract import fresh200_lr_factor
    from search_r1.baselines.aligned_outer_commit import _digest
    require(len(attempts)==len(events)==200,'exact200 attempts and summaries required')
    progress=Fresh200Progress(baseline_aligned=True);em_reports=[]
    binding_expected=dict(data_manifest_path=config['baselines']['holdout']['manifest_path'],
        data_manifest_sha256=config['baselines']['holdout']['manifest_sha256'],
        retrieval_manifest_path=config['retriever']['identity_manifest'],retrieval_manifest_sha256=config['retriever']['identity_manifest_sha256'],
        order_manifest_path=config['baselines']['alignment']['order_manifest'],order_manifest_sha256=config['baselines']['alignment']['order_sha256'],
        common_input_binding=shared_inputs['common_input_binding'])
    for outer,(attempt,event)in enumerate(zip(attempts,events),1):
        require(attempt.get('attempt_id')==outer and attempt.get('candidate_outer')==outer and attempt.get('driver_pid')==driver_pid
            and attempt.get('source_commit')==SOURCE and attempt.get('status')=='COMPLETE'and attempt.get('accepted_complete_outer')is True,'same-driver complete no-skip attempt differs')
        ledger=run/'checkpoints/diagnostics'/f'attempt_{outer}_candidate_outer_{outer}/all-rank-write-ledger.json'
        require(attempt.get('ledger_path')==str(ledger)and read_record(ledger)==attempt,'actual immutable driver ledger differs')
        require(progress.begin_attempt(outer)==outer and progress.record(attempt['rank_write_ledger'],outer),'200 complete rank-ledger replay failed')
        require(attempt.get('progress')==progress.state_dict(),'original progress differs from actual rank-write replay')
        require(event.get('step')==outer and event.get('attempt_id')==outer and event.get('accepted_complete_outer')is True,'original training summary identity differs')
        metrics=event['metrics']
        require(metrics.get('actor/optimizer_steps')==20 and metrics.get('actor/scheduler_outer_steps')==outer
            and metrics.get('bapo/actual_outer_updates')==outer and metrics.get('bapo/stage_evaluated_count')==64,'actual20write/stage/scheduler metric differs')
        stage=read_record(run/'aligned-stage-proof'/f'outer_{outer:04d}-attempt_{outer:04d}.json')
        require(stage.get('sha256')==_digest({k:v for k,v in stage.items()if k!='sha256'})
            and stage.get('actual_outer')==outer and stage.get('attempt_id')==outer and type(stage.get('evaluated_count'))is int and stage.get('evaluated_count')==64
            and stage.get('driver_pid')==driver_pid and stage.get('run_root')==str(run)and stage.get('source_commit')==SOURCE
            and stage.get('source_sha256')==source_hashes and stage.get('common_input_binding')==shared_inputs['common_input_binding']
            and stage.get('stage_input_binding')==shared_inputs['stage']and stage.get('order_manifest')==binding_expected['order_manifest_path']
            and stage.get('order_manifest_sha256')==binding_expected['order_manifest_sha256']
            and type(stage.get('score'))in(int,float)and math.isfinite(stage['score'])and 0<=stage['score']<=1
            and stage['score']==metrics.get('bapo/stage_validation')and stage['sha256']==metrics.get('fresh200/stage_proof_sha256'),'actual stage64 source/input proof differs')
        prepared=read_record(run/'aligned-outer-prepared'/f'outer_{outer:04d}-attempt_{outer:04d}.json')
        require(prepared.get('status')=='WRITES_PREPARED_NOT_ACCEPTED_OUTER'and prepared.get('actual_outer')==outer
            and prepared.get('attempt_id')==outer and prepared.get('driver_pid')==driver_pid and prepared.get('source_commit')==SOURCE
            and prepared.get('previous_completed_outer')==outer-1,'prepared write identity differs')
        provisional={r.get('rank'):r for r in prepared.get('rank_write_ledger',[])}
        require(len(provisional)==8 and set(provisional)==set(range(8)),'prepared eight ranks required')
        transitions={'outer_commit_phase','accepted_complete_outer','stage_validation','scheduler_before','scheduler_after','scheduler_lr_before','scheduler_lr_after'}
        for rank_event in attempt['rank_write_ledger']:
            rank=rank_event['rank'];prior=provisional[rank]
            require(rank_event.get('outer_commit_phase')=='STAGE_COMMITTED'and rank_event.get('accepted_complete_outer')is True
                and rank_event.get('stage_validation')==stage and rank_event.get('scheduler_before')==outer-1 and rank_event.get('scheduler_after')==outer
                and prior.get('outer_commit_phase')=='WRITES_PREPARED'and prior.get('accepted_complete_outer')is False
                and prior.get('scheduler_before')==prior.get('scheduler_after')==outer-1
                and all(rank_event.get(k)==v for k,v in prior.items()if k not in transitions),'stage-before-scheduler committed transition differs')
            for key,completed in (('scheduler_lr_before',outer-1),('scheduler_lr_after',outer)):
                lr=rank_event.get(key);expected=1e-6*fresh200_lr_factor(completed)
                require(isinstance(lr,list)and lr and all(type(v)in(int,float)and math.isfinite(v)and math.isclose(v,expected,rel_tol=1e-14,abs_tol=0.)for v in lr),'warmup57 actual LR differs')
            fault_dir=Path(config['actor_rollout_ref']['actor']['fresh200_fault_dir'])
            require(run in fault_dir.parents,'foreign rank evidence directory')
            require(read_record(fault_dir/f'rank_{rank}-outer_{outer}-attempt_{outer}.prepared.json')==prior
                and read_record(fault_dir/f'rank_{rank}-outer_{outer}-attempt_{outer}.json')==rank_event,'original rank events differ')
            monitors=rank_event.get('policy_monitor',[])
            require(len(monitors)==2 and [m.get('stage')for m in monitors]==['before_adam','after_adam']
                and all(m.get('denominator_role')=='frozen_hf_old_policy'and m.get('diagnostic_only')is True
                    and m.get('monitoring_scope')=='full_frozen_rollout'for m in monitors),'frozenHF-old diagnostic roles differ')
        path=run/'training-em-audit'/f'outer_{outer:04d}-attempt_{outer:04d}.jsonl'
        report=verify_em(path,dict(source_commit=SOURCE,run_root=str(run),driver_pid=driver_pid,actual_outer=outer,attempt_id=outer,
            expected_questions=512,siblings=5,bindings=binding_expected,policy_snapshot_id=f'{run}:attempt_{outer}:before_outer_{outer}'))
        require(all(metrics.get(key)==value for key,value in report['metrics'].items()),'independently replayed EM differs from original metrics')
        em_reports.append(dict(actual_outer=outer,metrics=report['metrics']))
    require(progress.actual_adam_writes_per_rank==[4000]*8 and progress.skipped_opportunities_per_rank==[0]*8
        and progress.completed_outers==200 and progress.incomplete_attempts==0,'final full200/4000write no-skip accounting differs')
    return dict(progress=progress.state_dict(),training_em=em_reports,scope='RAW_PRODUCER_LEDGER_AND_AUTHORED_ANSWER_REPLAY_NOT_STATE_RELOAD')


def native_metadata(side,rank,outer,event,path):
    require(type(side.get('rank'))is int and side['rank']==rank and type(side.get('step'))is int and side['step']==outer
        and type(side.get('world_size'))is int and side['world_size']==8
        and type(side.get('scheduler_last_epoch'))is int and side['scheduler_last_epoch']==outer
        and side.get('native_state_version')==3 and side.get('attempt_id')==outer
        and side.get('status')==('EXACT_FULL_STATE_RELOAD_OK'if outer==5 else'FULL_STATE_SERIALIZED_OK')
        and side.get('roundtrip')is(outer==5) and side.get('path')==str(path)
        and 'aligned_pending_outer'in side and side['aligned_pending_outer']is None
        and side.get('latest_rank_event')==event,'original native checkpoint metadata differs')


def collect_phase1(submit,output,gate,query=execute):
    who=identity(submit);run=who['run'];records=[];archived={};external={ROOT/'scheduler-head/formal-rayip-v2/formal-submit-receipt.json'}
    def retaining_query(argv):
        raw=query(argv);write_new(output/'scheduler-original-query.json',raw);return raw
    observation=terminal_observation(submit,retaining_query)
    write_new(output/'scheduler-terminal-observation.json',observation)
    require(observation['terminal_verified'],'actual strongly bound terminal scheduler evidence required before RUN reads')
    def archive(path):
        path=regular(path,ROOT)
        require(run in path.parents or path in external,'foreign raw evidence')
        if str(path)in archived:
            saved,original=archived[str(path)]
            verify_original_bindings([original]);return saved
        relative=path.relative_to(ROOT);record=copy_regular(path,output/'raw'/relative)
        records.append(dict(path=str(path),sha256=record['sha256'],bytes=record['bytes']))
        saved=Path(record['archive']);archived[str(path)]=(saved,records[-1])
        write_new(output/f'raw-binding-{len(records):05d}.json',record)
        return saved
    def read_record(path):return json.loads(archive(path).read_text())
    # Cache repeated ledger accesses without overwriting archived original files.
    cache={}
    def read_cached(path):
        if str(path)not in cache:cache[str(path)]=read_record(path)
        return cache[str(path)]
    for name in ('source-commit.txt','formal-supervisor.log','resource-watch.jsonl','own-process-cleanup.json','own-process-cleanup.err',
                 'gpu-exit.csv','gpu-exit.err','original-exit-code.txt','cleanup-exit-code.txt'):
        archive(run/name)
    require((run/'source-commit.txt').read_text().strip()==SOURCE,'original source marker differs')
    require((run/'original-exit-code.txt').read_text().strip()=='0'and (run/'cleanup-exit-code.txt').read_text().strip()=='0'
        and observation['completed_successfully'],'formal job failed; terminal raw evidence preserved')
    resource_rows=[json.loads(line)for line in archive(run/'resource-watch.jsonl').read_text().splitlines()]
    resources=resource_metadata(resource_rows,run,who['job'])
    cleanup=read_cached(run/'own-process-cleanup.json')
    require(cleanup.get('own_marker')==str(run)and cleanup.get('remaining')==[],'own process cleanup differs')
    log_path=archive(run/'train.log');text=log_path.read_text();require(text.endswith('\n'),'terminal log partial line')
    attempts=marked(text,'FRESH200_ATTEMPT_JSON=');events=marked(text,'BAPO_TRAIN_METRICS_JSON=')
    require(len(attempts)==200,'actual200 attempts missing');pid=attempts[0]['driver_pid'];require(type(pid)is int and pid>1,'original driver PID required')
    config=read_cached(run/'checkpoints/fresh200_state/outer_5_COMPLETE/resolved-config.json')
    require(gate.validate_gate_config(config)==(200,57)and config['baselines']['algorithm_source_commit']==SOURCE
        and config['baselines']['alignment']['common_manifest_sha256']==COMMON_SHA,'source080 common0b production configuration differs')
    external.add(regular(config['trainer']['fresh200_gate_receipt'],ROOT))
    release=fifth_release_binding(text,config,run,pid,read_cached)
    factual=read_cached(run/'checkpoints/fresh200_state/outer_5_COMPLETE/aligned-step5-bindings.json')
    from search_r1.baselines.fresh200_contract import critical_source_hashes
    require(factual.get('source_commit')==SOURCE and factual.get('source_sha256')==critical_source_hashes()
        and factual.get('driver_pid')==pid and factual.get('run_root')==str(run),'original shared source/run/PID bindings differ')
    from search_r1.baselines.training_em import verify_training_em_audit
    def em(path,expected):
        saved=archive(path)
        return verify_training_em_audit(saved,records[-1]['sha256'],expected)
    replay=replay_accounting(attempts,events,run,config,pid,read_cached,em,factual['shared_inputs'],factual['source_sha256'])
    completed=completed_marker(text,replay['progress'])
    finals=marked(text,'SEARCH_R1_FINAL_METRICS_JSON=')
    require(len(finals)==1,'one actual terminal final-evaluation summary required')
    from search_r1.baselines.holdout import E1_FINAL_COUNTS
    aggregate=final_aggregate(finals[0],E1_FINAL_COUNTS)
    dev=dev_schedule(events)
    require(text.rfind('SEARCH_R1_FINAL_METRICS_JSON=')>text.rfind('BAPO_TRAIN_METRICS_JSON='),'final aggregate precedes actual outer200 metrics')
    # Inspect only small checkpoint metadata now; large native/HF bytes are pending.
    checkpoint_metadata=[]
    for outer in (5,50,100,150,200):
        folder=run/'checkpoints/fresh200_state'/f'outer_{outer}_COMPLETE'
        status=read_cached(folder/'status.json');driver=read_cached(folder/'driver-reload.json')
        require(status.get('status')=='COMPLETE'and status.get('actual_outer')==status.get('completed_outer_updates')==outer
            and status.get('attempt_id')==outer and driver.get('step')==outer and driver.get('pid')==pid
            and driver.get('status')==('EXACT_DRIVER_RELOAD_OK'if outer==5 else'DRIVER_SERIALIZED_OK'),'actual checkpoint metadata differs')
        require(status.get('progress')==attempts[outer-1]['progress'],'checkpoint progress differs from replayed attempts')
        ranks=[]
        for rank in range(8):
            side=read_cached(folder/'ranks'/f'rank_{rank}.json')
            native_metadata(side,rank,outer,attempts[outer-1]['rank_write_ledger'][rank],folder/'ranks'/f'rank_{rank}.pt')
            ranks.append(side)
        checkpoint_metadata.append(dict(actual_outer=outer,driver=driver,ranks=ranks,status=status))
    verify_original_bindings(records)
    return dict(status='TERMINAL_RAW_200_LEDGER_EM_FINAL_AGGREGATE_COLLECTED_PARTIAL_NOT_ACCEPTANCE',
        scheduler=observation,source_commit=SOURCE,run_root=str(run),job=who['job'],driver_pid=pid,
        accounting=replay,original_completed_marker=completed,resources=resources,own_cleanup=cleanup,dev512=dev,fifth_release=release,final_aggregate=aggregate,checkpoint_metadata=checkpoint_metadata,original_bindings=records,
        supported_scope='SINGLE_ORIGINAL_FORMAL_JOB_NO_INFRASTRUCTURE_FAULT_NO_RESUME_ONLY',
        recovery_limitation='Fault/resume chains rejected here; retained partial writes require future cross-job accounting and must never be discarded.',
        native_scope='PRODUCER_METADATA_ONLY_NO_CPU_OR_GPU_RELOAD',pending=PENDING,acceptance=False,goal_complete=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();output=args.output
    require(output.is_absolute()and output.parent==ROOT/'evidence'and output.parent.resolve()==output.parent,'fixed own exclusive evidence directory required')
    output.mkdir();start=time.monotonic();cpu=time.process_time()
    result=dict(schema='aligned-final-evidence-candidate-phase1-v1',candidate=True,acceptance=False,goal_complete=False,
        release_receipt=False,pending=PENDING,status='CANDIDATE_UNEXECUTED_OR_UNBOUND_NOT_ACCEPTANCE',source_commit=SOURCE,
        root=str(ROOT),helper_files=[binding(HERE/name)for name in('aligned_final_evidence.py','aligned_first5_cpu_evidence.py','aligned_first5_math.py','aligned-first5-math-provenance.json','collect_hardware_evidence.py')])
    try:
        cpu_only()
        for name in ('aligned_first5_cpu_evidence','aligned_first5_math','collect_hardware_evidence'):
            require(Path(sys.modules[name].__file__).resolve()==HERE/(name+'.py'),'foreign deployed evidence dependency imported')
        gate=bind_source();receipt=regular(ROOT/'scheduler-head/formal-rayip-v2/formal-submit-receipt.json',ROOT)
        original=binding(receipt);submit=json.loads(receipt.read_text());identity(submit)
        write_new(output/'original-formal-submit-receipt.json',submit)
        require(submit.get('files')and isinstance(submit['files'],dict),'actual formal tools bytes missing')
        tools=[binding(regular(ROOT/'tools/deployment/formal-rayip-v2'/name,ROOT))for name in submit['files']]
        require(all(r['sha256']==submit['files'][Path(r['path']).name]for r in tools),'formal tools changed')
        result['actual_submission']=original;result['formal_tools']=tools
        result['phase1']=collect_phase1(submit,output,gate)
        verify_original_bindings([original]+tools+result['helper_files'])
        result['status']=result['phase1']['status']
    except Exception as error:
        result['status']='PARTIAL_INCOMPLETE_OR_UNAVAILABLE_NOT_ACCEPTANCE'
        result['issue']=dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
    result['cost']=dict(wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu,
        peak_rss_linux_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='phase1 CPU archive/replay only',monetary_price_assumed=False)
    write_new(output/'phase1-evidence.json',result)
    print(json.dumps(dict(status=result['status'],acceptance=False,goal_complete=False,path=str(output/'phase1-evidence.json'))))

if __name__=='__main__':main()
