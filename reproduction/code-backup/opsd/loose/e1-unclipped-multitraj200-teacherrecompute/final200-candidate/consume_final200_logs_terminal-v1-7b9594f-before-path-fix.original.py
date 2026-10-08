"""Read-only fcec200 log/terminal layers. Never signs, launches or restores."""
import argparse
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import stat
import types

SOURCE = 'fcecbea5d939e41dcfb3b118a26ef049a9bda3d6'
CONFIG = '0f0fc3a6aff4648ce0d2c8ec28eb104e27678315ad09824c90da043956a0b990'
SPEC_KIND = 'OBSERVED_MULTITRAJ200_LOG_TERMINAL_INPUT_NOT_COMPLETION_RECEIPT'
OUTER = 'OPSD_E1_OUTER_JSON='
EVAL = 'BASELINE_EVALUATION_METRICS_JSON='
FINAL = 'SEARCH_R1_FINAL_METRICS_JSON='
PAUSE = 'OPSD_E1_STEP5_PAUSED_JSON='
RELEASE = 'OPSD_E1_STEP5_RELEASED_JSON='
MARKERS = (OUTER, EVAL, FINAL, PAUSE, RELEASE)
TERMINAL_ROLES = {'training_child_exit', 'wrapper_exit', 'scheduler_original',
                  'sampler_stop', 'owned_cleanup', 'release_observation'}
FINAL_COUNTS = dict(nq=3610,triviaqa=11313,popqa=14267,hotpotqa=7405,
                    **{'2wikimultihopqa':12576},musique=2417,bamboogle=125)
DEV_COUNTS = dict(nq=256,hotpotqa=256)
STAGE_COUNTS = dict(nq=32,hotpotqa=32)
MEANS = ('f1','em','answer_format','format_rate','generated_tokens','search_calls')


def require(ok, message):
    if not ok: raise ValueError(message)


def exact(value, target, message):
    require(type(value) is type(target) and value == target, message)


def number(value, message, nonnegative=False):
    require(type(value) in (int,float) and math.isfinite(value) and
            (not nonnegative or value >= 0), message)
    return value


def numeric_equal(value,target,message):
    require(number(value,message)==target,message)


def close(value,target,message):
    require(math.isclose(number(value,message),target,rel_tol=1e-12,abs_tol=1e-18),message)


def load_common(expected_sha):
    p=Path(__file__).resolve().parent/'common.py'
    require(re.fullmatch('[0-9a-f]{64}',expected_sha) is not None and p.resolve(strict=True)==p,'Exact physical common source required')
    before=p.stat();fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        opened=os.fstat(fd);raw=b''
        while True:
            chunk=os.read(fd,1024**2)
            if not chunk:break
            raw+=chunk
        after=os.fstat(fd)
    finally:os.close(fd)
    fields=('st_dev','st_ino','st_size','st_uid','st_mode','st_mtime_ns','st_ctime_ns')
    require(stat.S_ISREG(opened.st_mode) and opened.st_uid==os.getuid() and all(tuple(getattr(x,k) for k in fields)==tuple(getattr(before,k) for k in fields) for x in (opened,after,p.stat())) and hashlib.sha256(raw).hexdigest()==expected_sha,'Pinned common changed')
    module=types.ModuleType('final200_logs_exact_common');module.__file__=str(p)
    exec(compile(raw,str(p),'exec',dont_inherit=True,optimize=0),module.__dict__)
    module.read_file(p,expected_sha)
    return module


def parse_log(raw,context,gate,fsdp_source):
    """Full, unmodified complete log bytes. Parsed records retain raw provenance."""
    require(type(raw) is bytes and raw.endswith(b'\n'),'Closed full log must end in newline')
    records=[];consoles=[];offset=0
    driver=context['manifest']['driver_pid'];workers=[w['pid'] for w in context['workers']]
    for n,line_raw in enumerate(raw.splitlines(keepends=True),1):
        line=line_raw.decode('utf-8',errors='strict').rstrip('\r\n')
        found=[m for m in MARKERS if m in line]
        if found:
            require(len(found)==1,'Different marker events on one physical line')
            marker=found[0]
            value=gate.parse_log_marker(line,marker,driver_pid=driver,job_id='3304',worker_pids=workers,
                checkpoint_warning_path='/mnt/data/shaoshitong/search-r1-rl/opsd/env/lib/python3.10/site-packages/torch/utils/checkpoint.py',fsdp_warning_source_bytes=fsdp_source)
            records.append(dict(marker=marker,value=value,line_number=n,offset=offset,bytes=len(line_raw),sha256=hashlib.sha256(line_raw).hexdigest(),raw_line=line))
        # Only the original driver's console transport is accepted.
        prefix=re.fullmatch(r'(?:(?:\x1b\[36m)?\(main_task pid=([1-9][0-9]*)\)(?:\x1b\[0m)?\s+)?(step:[0-9]+(?: - .*)?)',line)
        if prefix:
            require(prefix.group(1) is None or int(prefix.group(1))==driver,'Console driver PID differs')
            chunks=prefix.group(2).split(' - ');step=int(chunks[0].split(':')[1]);values={}
            for chunk in chunks[1:]:
                match=re.fullmatch(r'([^:]+):(-?[0-9]+\.[0-9]{3})',chunk)
                require(match is not None and match.group(1) not in values,'Malformed/duplicate console key')
                values[match.group(1)]=match.group(2)
            consoles.append(dict(step=step,values=values,line_number=n,offset=offset,sha256=hashlib.sha256(line_raw).hexdigest(),raw_line=line))
        elif re.search(r'(?:^|\s)step:[0-9]+(?: - |$)',line):
            raise ValueError('Unknown console transport/frame')
        offset+=len(line_raw)
    return dict(records=records,consoles=consoles,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def evaluation_math(metrics,counts,gate):
    require(type(metrics) is dict,'Evaluation must be an object')
    total=sum(counts.values());gate.validate_evaluation(metrics,total)
    for family in ('qa_count','evaluated_count'):
        expected=set(counts)|({'all'} if family=='qa_count' else set())
        require({k.removeprefix('val/'+family+'/') for k in metrics if k.startswith('val/'+family+'/')}==expected,'Exact evaluation source count keys differ')
        for source,count in counts.items():exact(metrics['val/'+family+'/'+source],count,'Typed evaluation source count differs')
    exact(metrics['val/qa_count/all'],total,'Typed evaluation total differs')
    for name in MEANS:
        require({k.removeprefix('val/'+name+'/') for k in metrics if k.startswith('val/'+name+'/')}==set(counts)|{'all'},'Evaluation mean source keys differ')
        for source in list(counts)+['all']:
            value=number(metrics['val/'+name+'/'+source],'Invalid evaluation mean',True)
            if name in ('f1','em','answer_format','format_rate'):require(value<=1,'Evaluation probability outside [0,1]')
        weighted=sum(metrics['val/'+name+'/'+s]*n for s,n in counts.items())/total
        close(metrics['val/'+name+'/all'],weighted,'Evaluation all/source weighted mean differs')
    for source,count in counts.items():
        for name in ('em','answer_format','format_rate'):
            hits=metrics['val/'+name+'/'+source]*count
            require(math.isclose(hits,round(hits),abs_tol=1e-7,rel_tol=0),'Evaluation integer hits differ')
        close(metrics['val/test_score/'+source],metrics['val/em/'+source],'Evaluation reward/EM differs')
    require({k.removeprefix('val/test_score/') for k in metrics if k.startswith('val/test_score/')}==set(counts),'Evaluation reward sources differ')
    return dict(count=total,source_counts=counts,metrics=metrics,
                per_question_raw_evidence_available=False,base_same_final_dataset_available=False)


def console_matches(console,metrics,prefix=''):
    for key,value in metrics.items():
        require(console['values'].get(prefix+key)==f'{number(value,"Console source value not numeric"):.3f}','Original console/JSON value differs: '+prefix+key)


def check_log(parsed,context,gate):
    events=parsed['records'];outer=[r for r in events if r['marker']==OUTER]
    require(len(outer)==200 and [r['value'].get('actual_outer_updates') for r in outer]==list(range(1,201)),'Exactly actual1..200 required; missing/duplicate/201')
    manifest=context['manifest'];result=[]
    for step,record in enumerate(outer,1):
        row=record['value'];require(set(row)=={'actual_outer_updates','global_step','driver_pid','job_id','source_commit','config_sha256','metrics'},'Outer schema differs')
        for key,target in dict(actual_outer_updates=step,global_step=step,driver_pid=manifest['driver_pid'],job_id='3304',source_commit=SOURCE,config_sha256=CONFIG).items():exact(row.get(key),target,'Outer identity differs: '+key)
        metrics=row['metrics'];gate.validate_outer(metrics,profile='multitraj200')
        for key,target in [('training/actual_outer_updates',step),('rollout/trajectories',2560),('diagnostic/student_archived_count',2560),('diagnostic/teacher_archived_count',512),('diagnostic/paired_archive_complete',1)]:numeric_equal(metrics.get(key),target,'Outer workload/archive summary differs')
        close(metrics.get('actor/lr'),1e-6*min(step/57,1),'Outer scheduler post LR differs')
        for rank in range(8):
            close(metrics.get(f'actor/optimizer_lr_used_mean/rank_{rank}'),1e-6*min((step-1)/57,1),'Rank actual Adam used LR differs')
            numeric_equal(metrics.get(f'actor/optimizer_lr_used_finite/rank_{rank}'),1,'Rank Adam LR finite flag differs')
            numeric_equal(metrics.get(f'teacher/cuda_update_entries_cumulative_per_rank/rank_{rank}'),step,'Fixed teacher outer counter differs')
            number(metrics.get(f'actor/opsd_loss_mean/rank_{rank}'),'Nonfinite per-rank distillation loss')
            number(metrics.get(f'actor/grad_norm_mean/rank_{rank}'),'Nonfinite per-rank gradient norm',True)
        number(metrics.get('actor/opsd_loss'),'Nonfinite logged distillation loss')
        number(metrics.get('actor/grad_norm'),'Nonfinite logged gradient norm',True)
        # PG/reference objectives are absent. Zero diagnostic advantages are expected.
        require(not any(k.startswith(('actor/pg_loss','actor/kl_loss','actor/entropy_loss','actor/ref_kl')) for k in metrics),'Unexpected PG/reference KL/entropy objective')
        for k in ('critic/advantages/min','critic/advantages/mean','critic/advantages/max'):numeric_equal(metrics.get(k),0,'PureOPSD diagnostic advantages must be zero')
        matches=[c for c in parsed['consoles'] if c['step']==step and c['line_number']<record['line_number']]
        require(len(matches)==1,'Missing/duplicate original outer console')
        console_matches(matches[0],metrics)
        result.append(dict(step=step,adam_calls_per_rank=10,student_summary_count=2560,teacher_summary_count=512,
            reward=metrics.get('critic/rewards/mean'),loss_rank0=metrics['actor/opsd_loss'],grad_norm_rank_mean=metrics['actor/grad_norm'],line_sha256=record['sha256'],console_sha256=matches[0]['sha256']))
    pauses=[r for r in events if r['marker']==PAUSE];released=[r for r in events if r['marker']==RELEASE]
    require(len(pauses)==len(released)==1,'One real first5 pause and signed release required')
    expected=dict(driver_pid=manifest['driver_pid'],job_id='3304',manifest_path=context['accepted5']['manifest_path'],manifest_sha256=context['accepted5']['manifest_sha256'],pause_nonce=manifest['pause_nonce'])
    require(pauses[0]['value']==expected,'Actual pause differs from signed5')
    require(released[0]['value']==dict(driver_pid=manifest['driver_pid'],actual_outer_updates=5,next_global_step=6,receipt_sha256=context['accepted5']['receipt_sha256']),'Actual release differs from historical signature')
    require(outer[4]['line_number']<pauses[0]['line_number']<released[0]['line_number']<outer[5]['line_number'],'First5 pause/release order differs')
    evaluations=[r for r in events if r['marker']==EVAL];final=[r for r in events if r['marker']==FINAL]
    expected_events=[('validation',0,DEV_COUNTS),('stage',5,STAGE_COUNTS)]+[(kind,step,counts) for step in (50,100,150,200) for kind,counts in [('validation',DEV_COUNTS),('stage',STAGE_COUNTS)]]+[('final',200,FINAL_COUNTS)]
    require(len(evaluations)==11 and len(final)==1,'Exact initialization/periodic/stage/final evaluation event count required')
    eval_results=[]
    for record,(kind,step,counts) in zip(evaluations,expected_events):
        m=record['value'];info=evaluation_math(m,counts,gate)
        if kind=='final':
            require(record['line_number']>outer[-1]['line_number'] and final[0]['line_number']>record['line_number'] and final[0]['value']==m,'Final baseline/final marker differs or precedes actual200')
            candidates=[c for c in parsed['consoles'] if c['step']==200 and c['line_number']>final[0]['line_number']]
        elif step==0:
            require(record['line_number']<outer[0]['line_number'],'Initialization evaluation followed training')
            candidates=[c for c in parsed['consoles'] if c['step']==0]
        else:
            current=outer[step-1];previous=outer[step-2]['line_number']
            require(previous<record['line_number']<current['line_number'],'Evaluation occurs outside intended outer')
            embedded={k.removeprefix('opsd_stage/'):v for k,v in current['value']['metrics'].items() if k.startswith('opsd_stage/')} if kind=='stage' else {k:v for k,v in current['value']['metrics'].items() if k.startswith('val/')}
            require(embedded==m,'Evaluation JSON/outer full metrics differ')
            candidates=[c for c in parsed['consoles'] if c['step']==step and c['line_number']<current['line_number']]
        require(len(candidates)==1,'Exact evaluation console pairing absent')
        console_matches(candidates[0],m,'opsd_stage/' if kind=='stage' else '')
        info.update(kind=kind,outer_step=step,event_line_sha256=record['sha256'],console_line_sha256=candidates[0]['sha256']);eval_results.append(info)
    require(len(parsed['consoles'])==202,'Unexpected console update/extra executed201')
    return dict(status='VERIFIED_LOG_EVENTS_ONLY_NOT_FINAL_ACCEPTANCE',actual_outer_updates=200,adam_calls_per_rank=2000,outer=result,evaluations=eval_results,final_metrics=final[0]['value'],first5_pause_sha256=pauses[0]['sha256'],first5_release_sha256=released[0]['sha256'],no_executed_outer201=True,scope='Original logs only; archive summary counters are not raw archive content qualification')


def scheduler(raw):
    text=raw.decode('utf-8');require(len(re.findall(r'\bJobId=',text))==1,'Scheduler must contain one actual job')
    require(re.search(r'\bJobId=3304(?:\s|$)',text) and re.search(r'\bUserId=[^\s(]+\(0\)(?:\s|$)',text),'Actual scheduler job/UID differs')
    state=re.search(r'\bJobState=([^\s]+)',text);exit_=re.search(r'\bExitCode=([0-9]+:[0-9]+)(?:\s|$)',text)
    require(state is not None and exit_ is not None,'Actual scheduler terminal/exit missing')
    require(state.group(1) in {'COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'},'Scheduler is not terminal')
    return dict(status='OBSERVED_LITERAL_SCHEDULER_TERMINAL',state=state.group(1),exit_code=exit_.group(1),original=text)


def terminal_layers(raws,c):
    """Null is BLOCKED; literal exits remain independent even when child succeeded."""
    out={}
    for name in ('training_child_exit','wrapper_exit'):
        raw=raws.get(name)
        if raw is None:out[name]=dict(status='BLOCKED_MISSING_ACTUAL_ORIGINAL',returncode=None);continue
        if name=='training_child_exit':
            value=c.strict_json(raw);require(type(value) is dict and set(value)=={'returncode'},'Training child exit original schema differs');rc=value['returncode']
        else:
            require(re.fullmatch(rb'-?[0-9]+\n?',raw) is not None,'Literal wrapper job-exit differs');rc=int(raw)
        require(type(rc) is int,'Exit bool/type differs');out[name]=dict(status='OBSERVED_LITERAL_EXIT',returncode=rc)
    out['scheduler']=dict(status='BLOCKED_MISSING_ACTUAL_ORIGINAL',state=None,exit_code=None) if raws.get('scheduler_original') is None else scheduler(raws['scheduler_original'])
    out['sampler_stop']=dict(status='BLOCKED_MISSING_ACTUAL_ORIGINAL')
    if raws.get('sampler_stop') is not None:
        value=c.strict_json(raws['sampler_stop']);require(type(value) is dict and set(value)=={'pid','returncode','scope'} and type(value['pid']) is int and value['pid']>0 and type(value['returncode']) is int and value['scope']=='Stopped our sampler before final NDJSON read','Sampler stopped original differs');out['sampler_stop']=dict(status='OBSERVED_SAMPLER_STOP_NOT_RESOURCE_RELEASE',**value)
    out['owned_cleanup']=dict(status='BLOCKED_MISSING_ACTUAL_ORIGINAL')
    if raws.get('owned_cleanup') is not None:
        value=c.strict_json(raws['owned_cleanup']);require(type(value) is dict and value.get('scope')=='Not evidence of resource release; capture terminal scontrol separately' and type(value.get('children')) is list,'Owned cleanup original scope differs');out['owned_cleanup']=dict(status='OBSERVED_CLEANUP_NOT_RESOURCE_RELEASE',original=value)
    # A separate independently reviewed raw release consumer is required. Never
    # elevate a JSON PASS or absence of NVML processes to whole-job release.
    out['resource_release']=dict(status='BLOCKED_REQUIRE_INDEPENDENT_PID_START_NVML_PORT_CGROUP_LEASE_LOCK_RAW_REVIEW',actual=None,observation_present=raws.get('release_observation') is not None)
    if raws.get('release_observation') is not None:out['resource_release']['raw_observation']=c.strict_json(raws['release_observation'])
    return out


def resource_observation(lines,c,uuids):
    """Keep all bad rows; never turns pretraining failures into whole PASS."""
    if type(lines) is bytes:lines=lines.splitlines(keepends=True)
    previous=None;bad=[];query_errors=[];count=0;bad66=None
    for line in lines:
        require(type(line) is bytes and line.endswith(b'\n'),'Closed resource original needs complete final line')
        count+=1;v=c.strict_json(line);exact(v.get('job_id'),'3304','Resource job differs')
        now=datetime.datetime.fromisoformat(v['time']);require(now.tzinfo is not None and now.utcoffset()==datetime.timedelta(0),'Resource time not UTC')
        require(previous is None or now>previous,'Resource time duplicated/backwards');previous=now
        issues=[]
        for a in v.get('gpu_processes',[]):
            if a.get('uuid') not in uuids or (not a.get('unavailable') and (a.get('in_slurm_job') is not True or str(a.get('uid'))!='0')):issues.append(a)
            if a.get('unavailable'):issues.append(a)  # Do not independently infer an exit-race exception here.
        if issues:bad.append(dict(line_number=count,line_sha256=hashlib.sha256(line).hexdigest(),time=v['time'],applications=issues))
        if v.get('query_error'):query_errors.append(dict(line_number=count,line_sha256=hashlib.sha256(line).hexdigest(),original=v))
        if count==66:
            hits=[a for a in v.get('gpu_processes',[]) if a.get('pid')==2544824 and a.get('process_start_ticks')=='1834772278' and a.get('in_slurm_job') is False and a.get('uid')=='0']
            require(len(hits)==1 and v['time']=='2026-10-06T11:47:26.494869+00:00','Whole unfiltered resource bad66 missing/changed')
            bad66=dict(line_number=66,line_sha256=hashlib.sha256(line).hexdigest(),original=v)
    require(count>=66 and bad66 is not None,'Whole resource original truncated')
    return dict(status='FAILED_WHOLE_RESOURCE_OWNERSHIP_NOT_UPGRADED_BY_TRAINING_PHASE',whole_rows=count,bad66=bad66,unresolved_or_bad_rows=bad,query_error_rows=query_errors,writer_closed_qualification='BLOCKED_SEPARATE_SAMPLER_EXIT_FD_PROVENANCE_REQUIRED',scope='Raw observation only; readable cgroupFalse is preserved, not reclassified as own/exited')


def resource_file(record,c,uuids):
    """Stream whole resources using the same FD and parsed-byte full SHA."""
    p=c.verify_record(record);before=p.stat();fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
    digest=hashlib.sha256();size=0
    try:
        opened=os.fstat(fd);require(c.identity(opened)==c.identity(before),'Resource open identity differs')
        require(opened.st_size<=3*1024**3,'Whole resource read exceeds engineering bound')
        with os.fdopen(fd,'rb',closefd=False) as handle:
            def lines():
                nonlocal size
                for line in handle:
                    require(len(line)<=32*1024**2,'Resource physical line exceeds bound')
                    digest.update(line);size+=len(line);yield line
            result=resource_observation(lines(),c,uuids)
        finished=os.fstat(fd)
    finally:os.close(fd)
    require(c.identity(before)==c.identity(finished)==c.identity(p.stat()) and size==record['bytes'] and digest.hexdigest()==record['sha256'],'Parsed resource bytes or physical identity changed')
    c.verify_record(record)
    return result


def consume(a,c):
    context=c.accepted_context(a.accepted5_receipt_sha256)
    spec,srec=c.read_json(a.evidence_spec,a.evidence_spec_sha256);records=dict(context['records']);records[srec['physical_path']]=srec
    for k,target in dict(version=1,kind=SPEC_KIND,source_commit=SOURCE,config_sha256=CONFIG,job_id='3304',run_root=str(c.RUN)).items():exact(spec.get(k),target,'Log/terminal spec identity differs: '+k)
    require(set(spec)=={'version','kind','source_commit','config_sha256','job_id','run_root','logs','terminal','resources'},'Unknown evidence spec key')
    require(set(spec['logs'])=={'driver','aggregate'} and set(spec['terminal'])==TERMINAL_ROLES,'Log/terminal role inventory differs')
    fsdp_path=Path('/mnt/data/shaoshitong/search-r1-rl/opsd/env/lib/python3.10/site-packages/torch/distributed/fsdp/fully_sharded_data_parallel.py')
    fsdp,fsrec=c.read_file(fsdp_path,context['gate'].FSDP_WARNING_SHA256,keep_bytes=True);records[str(fsdp_path)]=fsrec
    parsed={}
    for role,entry in spec['logs'].items():
        require(type(entry) is dict and set(entry)=={'file','origin_path','writer_pid'},'Log input role schema differs')
        exact(entry['writer_pid'],context['manifest']['driver_pid'],'Historical log writer PID differs')
        # Aggregate is wrapper tee, containing the exact old driver transport.
        if role=='aggregate':exact(entry['origin_path'],str(c.RUN/'train.log'),'Aggregate log origin differs')
        else:require(type(entry['origin_path']) is str and entry['origin_path'].startswith('/tmp/o4e-3304/ray/session_') and entry['origin_path'].endswith('-'+str(context['manifest']['driver_pid'])+'.out'),'Actual Ray driver origin differs')
        p=c.verify_record(entry['file']);raw,rec=c.read_file(p,entry['file']['sha256'],keep_bytes=True,limit=256*1024**2);records[str(p)]=rec
        parsed[role]=parse_log(raw,context,context['gate'],fsdp)
    log_results={k:check_log(v,context,context['gate']) for k,v in parsed.items()}
    left=parsed['driver']['records'];right=parsed['aggregate']['records']
    require([(r['marker'],r['value']) for r in left]==[(r['marker'],r['value']) for r in right],'Complete driver/aggregate evidence streams differ')
    manifest=context['manifest'];data_rec=manifest['files']['data_manifest'];data,drec=c.read_json(data_rec['path'],c.COMMON_DATA);records[drec['physical_path']]=drec
    dataset={}
    for split,counts in [('validation',DEV_COUNTS),('stage',STAGE_COUNTS),('final',FINAL_COUNTS)]:
        v=data['splits'][split];exact(v['rows'],sum(counts.values()),'Actual dataset rows differ');require(v['source_counts']==counts,'Actual seven/dev/stage source counts differ')
        cfgpath=context['config']['data']['val_files' if split=='validation' else 'final_val_files'] if split!='stage' else context['config']['baselines']['opsd_e1']['stage_files']
        exact(cfgpath,str(Path(data_rec['path']).parent/v['path']),'Actual config/data file path differs')
        _,rec=c.read_file(cfgpath,v['sha256']);records[rec['physical_path']]=rec;dataset[split]=dict(file=rec,rows=v['rows'],source_counts=counts,question_ids_sha256=v['question_ids_sha256'])
    raw_terminal={}
    for role,record in spec['terminal'].items():
        if record is None:raw_terminal[role]=None;continue
        p=c.verify_record(record);raw,rec=c.read_file(p,record['sha256'],keep_bytes=True,limit=32*1024**2);records[str(p)]=rec;raw_terminal[role]=raw
    terminal=terminal_layers(raw_terminal,c);resources=dict(status='BLOCKED_MISSING_ACTUAL_WHOLE_RESOURCE_ORIGINAL',actual=None)
    if spec['resources'] is not None:
        # Physical allocation UUIDs are anchored in signed first5 preparation.
        audit=context.get('preparation');require(type(audit) is dict,'Signed first5 preparation identity anchor required')
        uuids=set(audit['resource_native_join']['uuid_inventory']);require(len(uuids)==8,'Actual signed allocation UUID inventory differs')
        resources=resource_file(spec['resources'],c,uuids);records[spec['resources']['physical_path']]=spec['resources']
    own=c.read_file(Path(__file__).resolve())[1];records[own['physical_path']]=own
    common=c.read_file(c.__file__,a.common_sha256)[1];records[common['physical_path']]=common
    c.recheck(records)
    frames={}
    for role,stream in parsed.items():
        frames[role]=[]
        for row in stream['records']:
            body=row['raw_line'].split(row['marker'],1)[1].lstrip()
            _,end=json.JSONDecoder().raw_decode(body)
            if body[end:].strip():
                is_fsdp='fully_sharded_data_parallel.py:689: FutureWarning:' in body[end:]
                source_path=str(fsdp_path) if is_fsdp else '/mnt/data/shaoshitong/search-r1-rl/opsd/env/lib/python3.10/site-packages/torch/utils/checkpoint.py'
                frames[role].append({**row,'raw_json':body[:end],'raw_suffix':body[end:],'json_end_character':end,'warning_type':'OWNED_PINNED_FSDP689' if is_fsdp else 'OWNED_CHECKPOINT_AMP1399','warning_source_record':records.get(source_path),'scope':'Original frame accepted by frozen exact parser; unchanged whole log retained'})
    return dict(version=1,kind='FINAL200_LOG_TERMINAL_LAYER_REPORT_NOT_COMPLETION_RECEIPT',status='VERIFIED_LOGS_WITH_INDEPENDENT_TERMINAL_LAYERS_FINAL_ACCEPTANCE_BLOCKED',source_commit=SOURCE,config_sha256=CONFIG,stored_training_job_id='3304',accepted5=context['accepted5'],input_spec=srec,consumer_source=own,common_source=common,
        execution=dict(pid=os.getpid(),uid=os.getuid(),hostname=socket.gethostname(),scope='Current read-only process; not the historical training driver'),log_layers=log_results,dataset=dataset,terminal_layers=terminal,whole_resource_layer=resources,
        interleaved_original_event_frames=frames,
        actual_final_completion=None,blocked_layers=['Native200','WholeHF200','Archive4000','PostsourceBinding','IndependentResourceRelease','IndependentOverallAcceptance'],
        limitation='Final metrics are aggregates only; no original final per-question trajectories or same-final-dataset base comparison are manufactured',input_records=records)


def main():
    p=argparse.ArgumentParser()
    for name in ('common-sha256','accepted5-receipt-sha256','evidence-spec','evidence-spec-sha256','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();c=load_common(a.common_sha256);result=consume(a,c);out=c.write_exclusive(a.output,result)
    print(c.canonical(dict(output=out,status=result['status'],actual_final_completion=None)).decode())


if __name__=='__main__':main()
