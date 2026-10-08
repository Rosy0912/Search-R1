"""One local read of captured JSON; only source/metadata/log joins, no PT import."""
from pathlib import Path
import base64
import datetime
import hashlib
import json
import math
import os
import re
import shlex
import stat
import types

H=Path(__file__).resolve().parents[2]
P=H/'checkpoint50_100-collection-once-v2-original'
SOURCE=H.parent/'opsd-unclipped-multitraj200-teacherrecompute'
ATTRS=('st_dev','st_ino','st_uid','st_mode','st_nlink','st_size','st_mtime_ns','st_ctime_ns')
def need(ok,message):
    if not ok:raise ValueError(message)
def pin(raw):return hashlib.sha256(raw).hexdigest()
def capture(path):
    before=path.lstat();need(path.resolve(strict=True)==path and stat.S_ISREG(before.st_mode) and before.st_uid==os.getuid() and before.st_nlink==1,'Local physical original differs')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        opened=os.fstat(fd);parts=[]
        while True:
            part=os.read(fd,1048576)
            if not part:break
            parts.append(part)
        ended=os.fstat(fd)
    finally:os.close(fd)
    after=path.lstat();need(all(getattr(before,k)==getattr(v,k) for v in (opened,ended,after) for k in ATTRS),'Local original changed during read')
    raw=b''.join(parts)
    return raw,dict(path=str(path),sha256=pin(raw),bytes=len(raw),device=before.st_dev,inode=before.st_ino,uid=before.st_uid)
def compile_bytes(raw,path,name):
    module=types.ModuleType(name);module.__file__=str(path)
    exec(compile(raw,str(path),'exec',dont_inherit=True,optimize=0),module.__dict__)
    return module
def decoded(rec):
    raw=base64.b64decode(rec['base64'],validate=True)
    need(pin(raw)==rec['sha256'] and len(raw)==rec['bytes'],'Captured embedded full bytes differ')
    return raw
def whole_record(key,rec):
    need(type(rec) is dict and rec['physical_path'].startswith('/'),'Actual member path missing')
    for k in ('bytes','uid','device','inode','mode','nlink','mtime_ns','ctime_ns'):
        need(type(rec[k]) is int and rec[k]>=0,'Exact typed physical record differs: '+k)
    need(rec['bytes']>0 and rec['uid']==0 and rec['nlink']==1,'Owned whole member empty/alias differs')
    need(type(rec['sha256']) is str and re.fullmatch('[0-9a-f]{64}',rec['sha256']),'Actual fullSHA missing')
    need(key==rec['physical_path'],'Whole member key/path differs')
    if 'base64' in rec:decoded(rec)
    return {k:v for k,v in rec.items() if k!='base64'}
def process(record,expected):
    pid,start=expected
    need(type(record['pid']) is int and type(record['start_ticks']) is int and record['pid']==pid and record['start_ticks']==start and type(record['uid']) is int and record['uid']==0,'Original PID/start/UID mismatch')
    stat_raw=record['stat_original'];tail=stat_raw.rsplit(')',1)[1].split()
    need(int(stat_raw.split(' ',1)[0])==pid and tail[19]==str(start),'Actual proc stat clock differs')
    need(re.search(r'^Uid:\s+0\s+0\s+0\s+0\s*$',record['status_original'],re.M),'Actual four UIDs differ')
    need(re.search(r'^State:\s+[RSDI]',record['status_original'],re.M),'Actual original process not live')
    need(any(re.search(r'(?:^|/)job[_-]3304(?:\.scope)?/step[_-]0(?:\.scope)?(?:/|$)',x.split(':',2)[-1]) for x in record['cgroup'].splitlines()),'Actual cgroup identity differs')
    return dict(pid=pid,start_ticks=start,uid=0,cgroup=record['cgroup'])
def save(path,value):
    raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o444)
    with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    return dict(path=str(path),sha256=pin(raw),bytes=len(raw))

originals={};raws={}
for name in ('caller-source.original.py','collector-source.original.py','bootstrap-source.original.py','transport-source.original.py','software-peer.original.json','intent.original.json','popen.original.json','stdin.original.json','wait.original.json','stderr.original','stdout.original'):
    raws[name],originals[name]=capture(P/name)
need(originals['stdout.original']['bytes']==59981954 and originals['stdout.original']['sha256']=='4951d2635924679083b2bcdbd802100d45c37bb00eaa8756eedf6942b0cf8673','Full actual stdout identity differs')
need(raws['stderr.original']==b'','Actual collector stderr nonempty')
need(originals['collector-source.original.py']['sha256']=='61368d54b0e6732c9fef424b18df1903cce3bd50ed2ba3582c8c122b8426deba','Collector executed source differs')
need(originals['transport-source.original.py']['sha256']=='a64bf52a97cbef6e3b0cc7948e095546687c8f1690094469c4937180f9449a85','Transport original differs')
c=compile_bytes(raws['collector-source.original.py'],P/'collector-source.original.py','independent_actual_collector_pure_helpers')
intent=c.parse(raws['intent.original.json']);wait=c.parse(raws['wait.original.json']);stdin=c.parse(raws['stdin.original.json']);popen=c.parse(raws['popen.original.json']);peer=c.parse(raws['software-peer.original.json'])
need(wait['waited'] is True and type(wait['returncode']) is int and wait['returncode']==0 and wait['stdin_complete'] is True and wait['error'] is None,'Actual SSH wait not true0')
need(wait['pid']==popen['pid'] and popen['argv']==intent['argv'],'Popen/wait identity mismatch')
need(stdin['complete'] is True and stdin['bytes']==28100 and stdin['sha256']==c.sha(raws['collector-source.original.py']),'Actual complete stdin differs')
need(intent['collector_sha256']==stdin['sha256'] and intent['caller_sha256']==originals['caller-source.original.py']['sha256'] and intent['peer_sha256']==originals['software-peer.original.json']['sha256'] and intent['requested_steps']==[50,100],'Actual authority binding differs')
need(peer['status']=='PASS_INDEPENDENT_CHECKPOINT50_100_COLLECTOR_V2_DELTA_SOFTWARE_REVIEW' and peer['runtime_sha256']==stdin['sha256'],'Peer software scope/runtime differs')
outer=shlex.split(intent['argv'][-1]);middle=shlex.split(outer[-1]);inner=shlex.split(middle[-1]);need(inner[:3]==['python3','-B','-c'] and inner[3].encode()==raws['bootstrap-source.original.py'],'Exact stdin bootstrap invocation differs')
result=c.parse(raws['stdout.original'])
need(type(result['version']) is int and result['version']==1 and result['kind']=='TRAINING3304_CHECKPOINT50_100_READONLY_ORIGINALS' and result['status']=='READONLY_ACTUAL_SAVED_CHECKPOINT_LOG_AND_FILE_SNAPSHOTS_REQUIRES_INDEPENDENT_REVIEW','Actual collection is incomplete/different scope')
need(result['job_id']=='3304' and result['source_commit']==c.SOURCE and result['requested_steps']==[50,100] and 'error' not in result,'Actual stored experiment mismatch')
need(set(result['source_files'])==set(c.SOURCE_FILES) and len(result['source_files'])==19,'Actual original source19 inventory differs')
for relative,wanted in c.SOURCE_FILES.items():
    rec=result['source_files'][relative];code=decoded(rec)
    whole_record(str(c.REPO/relative),rec)
    need(rec['sha256']==wanted and pin((SOURCE/relative).read_bytes())==wanted,'Actual/local frozen source identity differs')
for label in ('git_head','git_clean','slurm','git_head_after','git_clean_after'):
    cmd=result['commands'][label];need(cmd['waited'] is True and type(cmd['returncode']) is int and cmd['returncode']==0 and cmd['stderr']=='','Read-only command literal wait/status differs')
need(result['commands']['git_head']['stdout'].strip()==result['commands']['git_head_after']['stdout'].strip()==c.SOURCE,'Actual source HEAD before/after differs')
need(not result['commands']['git_clean']['stdout'].strip() and not result['commands']['git_clean_after']['stdout'].strip(),'Actual source clean before/after differs')
slurm=result['commands']['slurm']['stdout']
for token in ('JobId=3304 ','UserId=root(0)','JobState=RUNNING','Restarts=0','NodeList=h100-10','gres:gpu:h100:8'):need(token in slurm,'Actual live Slurm identity differs')
identities={}
for phase in ('before','after'):
    identities['driver_'+phase]=process(result['driver_'+phase],c.DRIVER)
    need(set(result['workers_'+phase])==set(map(str,range(8))),'Actual eight worker ranks differ')
    identities['workers_'+phase]={str(r):process(result['workers_'+phase][str(r)],c.WORKERS[r]) for r in range(8)}
need(identities['driver_before']==identities['driver_after'] and identities['workers_before']==identities['workers_after'],'Actual original worker continuity differs')
files=result['files'];need(len(files)==68,'Actual file graph68 differs')
anchors={str(c.RUN/'full_state/global_step_5/resolved-config.json'):c.CONFIG,str(c.RUN/'bindings/data_manifest.json'):c.COMMON,str(c.RUN/'acceptance-5-receipt.json'):c.RECEIPT,str(c.FSDP_PATH):c.FSDP_SHA}
for path,wanted in anchors.items():need(files[path]['sha256']==wanted,'Actual anchor content SHA differs');decoded(files[path])
gate=c.load_gate(decoded(result['source_files']['search_r1/baselines/checkpoint_gate.py']))
envelope=c.parse(decoded(files[str(c.RUN/'acceptance-5-receipt.json')]))
public=c.parse((H/'job3304-first5-complete-originals-v3/raw/bindings/public.json').read_bytes());gate.verify_rsa_signature(envelope['payload'],envelope['signature_hex'],public)
config=c.parse(decoded(files[str(c.RUN/'full_state/global_step_5/resolved-config.json')]))
need(config['baselines']['opsd_e1']['profile']=='multitraj200' and config['baselines']['distillation']['clip_coef'] is None,'Stored pure unclipped profile differs')
warnings=decoded(files[str(c.FSDP_PATH)])
driver=decoded(files['driver-current-prefix.out']);train=decoded(files['train.log'])
need(files['driver-current-prefix.out']['physical_path']==str(c.DRIVER_PATH) and files['train.log']['physical_path']==str(c.RUN/'train.log'),'Actual log origin differs')
events,partial=c.parse_events(driver,gate,warnings);aggregate,ap=c.parse_events(train,gate,warnings)
rows=[x for x in events if x['marker']=='OPSD_E1_OUTER_JSON='];arows=[x for x in aggregate if x['marker']=='OPSD_E1_OUTER_JSON=']
need(len(rows)==result['driver_prefix_complete_outer']==128 and len(arows)==result['aggregate_prefix_complete_outer']==128,'Actual completed prefix count differs')
c.same([x['value'] for x in arows],[x['value'] for x in rows],'Actual original aggregate/driver complete prefix differs')
c.same(partial,result['driver_incomplete_eof'],'Driver partial EOF differs');c.same(ap,result['aggregate_incomplete_eof'],'Aggregate partial EOF differs')
observed_frames=[dict(log_origin=origin,event=x) for origin,items in [('driver',events),('aggregate',aggregate)] for x in items if 'native_repeat_transport' in x]
c.same(observed_frames,result['native_repeat_transport_frames'],'Exact native repeat transport proof differs')
need(len(observed_frames)==2 and sorted(x['event']['native_repeat_transport']['repeated_count'] for x in observed_frames)==[2,5],'Actual two known repeat frames differ')
saved_graph={};snapshots={};all_members=set()
for step in (50,100):
    actual=result['snapshots'][str(step)];metrics,_=c.progress(events,gate,step);evaluation=c.evaluations_and_console(driver,events,rows,gate,step)
    c.same(metrics,actual['metrics'],'Actual strict progress join differs');c.same(evaluation,actual['evaluations'],'Actual fixed evaluation/console original join differs')
    st=actual['stored'];need(st['member_count']==31 and all(st[k] is None for k in ('native_reload','whole_hf_load','rng_content_readback','sampler_content_readback')),'Stored snapshot scope upgraded')
    actor=c.RUN/('checkpoints/actor/global_step_'+str(step));driverdir=c.RUN/('full_state/global_step_'+str(step));algopath=c.RUN/('checkpoints/algorithm/global_step_'+str(step))/'state.json'
    index=c.parse(decoded(files[str(actor/'model.safetensors.index.json')]))
    c.same(index,st['hf435_index'],'Actual HF435 index differs');need(len(index['weight_map'])==435,'HF index435 cardinality differs')
    shards=set(index['weight_map'].values());need(len(shards)==3 and all(re.fullmatch('model-[0-9]{5}-of-00003.safetensors',x) for x in shards),'HF3shard map differs')
    paths={str(actor/n) for n in c.HF_SMALL|shards}|{str(actor/'training_state'/('rank_'+str(r)+ext)) for r in range(8) for ext in ('.json','.pt')}|{str(driverdir/n) for n in ('driver.json','driver.pt')}|{str(algopath)}
    need(len(paths)==31 and paths<=set(files),'Stored exact31 role graph absent');need(not all_members&paths,'Stored steps overlap physical paths');all_members|=paths
    saved_graph[str(step)]={p:whole_record(p,files[p]) for p in sorted(paths)}
    need(len({(files[p]['device'],files[p]['inode']) for p in paths})==31,'Saved31 physical inode aliases')
    reports=[]
    for r in range(8):
        report=c.parse(decoded(files[str(actor/'training_state'/('rank_'+str(r)+'.json'))]));c.same(report,st['rank_reports'][r],'Raw rank JSON vs claimed snapshot differs')
        for k,v in dict(version=3,rank=r,world_size=8,step=step,pid=c.WORKERS[r][0],scheduler_last_epoch=step,optimizer_state_entries=37,model_dtensor_entries=435).items():need(type(report[k]) is int and report[k]==v,'Typed actual stored rank header differs: '+k)
        need(report['status']=='FULL_STATE_SERIALIZED_OK' and report['algorithm']=='pure_opsd' and report['path']==str(actor/'training_state'/('rank_'+str(r)+'.pt')),'Stored rank scope/path differs')
        need(report['state_sha256']==report['sha256']==files[report['path']]['sha256'],'Stored PT fullSHA not joined')
        need(type(report['teacher']['counters']['cuda_update_entries']) is int and report['teacher']['counters']['cuda_update_entries']==step,'Stored teacher actual CUDA counter differs');reports.append(report)
    driver_report=c.parse(decoded(files[str(driverdir/'driver.json')]));c.same(driver_report,st['driver_report'],'Raw driver JSON differs')
    for k,v in dict(step=step,pid=c.DRIVER[0],uid=0).items():need(type(driver_report[k]) is int and driver_report[k]==v,'Typed driver header differs')
    need(driver_report['status']=='DRIVER_STATE_SERIALIZED_OK' and driver_report['job_id']=='3304' and driver_report['state_sha256']==files[str(driverdir/'driver.pt')]['sha256'],'Driver PT hash/scope differs')
    algorithm=c.parse(decoded(files[str(algopath)]));c.same(algorithm,st['algorithm'],'Raw algorithm JSON differs')
    need(type(algorithm['version']) is int and algorithm['version']==1 and type(algorithm['global_step']) is int and algorithm['global_step']==step and algorithm['method']=='pure_opsd' and algorithm['outcome_rewards_train_policy'] is False,'Stored algorithm identity differs')
    need(type(algorithm['trajectory_archive']['completed_outer']) is int and algorithm['trajectory_archive']['completed_outer']==step,'Stored archive cursor differs')
    observed=[dict(log_origin=origin,event=x) for origin,items in [('driver',events),('aggregate',aggregate)] for x in items if x['marker']=='FULL_STATE_RANK_JSON=' and x['value'].get('step')==step]
    c.same(observed,st['actual_observed_rank_console_records'],'Visible native rank console original set differs')
    for item in observed:
        r=item['event']['value']['rank'];c.same(item['event']['value'],reports[r],'Visible native console vs rank JSON differs')
    dev=evaluation['dev_at_step']['value'];stage=evaluation['stage_at_step']['value'];snapshots[str(step)]=dict(saved_members31=True,saved_member_bytes=sum(files[p]['bytes'] for p in paths),rank_status=[x['status'] for x in reports],optimizer_calls_per_rank=metrics['optimizer_calls_per_rank'],scheduler_clock=step,HF_keys=435,HF_shards=3,Adam_entries_per_rank=37,model_dtensor_entries_per_rank=435,expected_consumed_questions=512*step,dev=dict(count=dev['val/qa_count/all'],EM=dev['val/em/all'],F1=dev['val/f1/all']),stage=dict(count=stage['val/qa_count/all'],EM=stage['val/em/all'],F1=stage['val/f1/all']),visible_original_rank_console_count=len(observed),native_reload=None,whole_hf_load=None,rng_content_readback=None,sampler_content_readback=None)
need(len(all_members)==62 and len({(files[p]['device'],files[p]['inode']) for p in all_members})==62,'Two stored checkpoints exact62 physical graph differs')
report=dict(version=1,kind='INDEPENDENT_ACTUAL3304_CHECKPOINT50_100_STORED_BYTE_AND_SERIALIZED_OBSERVATION_REVIEW',status='PASS_STORED_BYTES_SERIALIZED_EVALUATION_JOIN_ONLY',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),originals=originals,source_commit=c.SOURCE,config_sha256=c.CONFIG,common_manifest_sha256=c.COMMON,signed_first5_sha256=c.RECEIPT,original_process_identity_before_after=identities,source19={n:{k:v for k,v in rec.items() if k!='base64'} for n,rec in result['source_files'].items()},commands=result['commands'],file_records68=len(files),saved_records62=saved_graph,snapshots=snapshots,completed_log_prefix=dict(driver=128,aggregate=128,scope='Captured growing complete prefixes only; not closed writers'),native_repeat_transport_frames=observed_frames,scope='Whole saved bytes SHA/stat observed on own h10010 plus serialized rank/driver/algorithm and fixed eval-console joins. No local PT/HF import/reload, tensor moments/RNG/sampler inspection or qualification.',retained_failure=dict(prior_once_v1='Actual collector exit1 due old parser repeat frames; no saved31 accepted; not training failure and not retroactive PASS'),limitations=['Large saved bodies were only full-hashed remotely in approved actual read-only collector; they were not copied or re-hashed locally.','Signed first5 authenticates prior5 only, not50/100. No pause, save manifest or signed50/100 receipt fabricated.','Only visibly captured native console rank rows are joined; repeated5/2 tags are not expanded into hidden ranks.','Dev512 and stage64 are independent fixed cohorts; no cross-cohort improvement interpretation.','68 captured file records include62 checkpoint members plus4 anchors/native-warning source and2 growing logs.','No final200/wholeHF/native state reload/RNG restoration/scheduler terminal/resource release proof.'])
output=save(H/'ACTUAL3304_CHECKPOINT50_100_STORED_SERIALIZED_INDEPENDENT_REVIEW.json',report)
print(json.dumps(dict(output=output,snapshots=snapshots,large_files_rehashed_locally=False,models_loaded=False),sort_keys=True))
