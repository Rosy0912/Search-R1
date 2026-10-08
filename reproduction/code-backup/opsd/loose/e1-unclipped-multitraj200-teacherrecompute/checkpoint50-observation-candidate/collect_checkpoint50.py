#!/usr/bin/env python3
"""Fixed read-only snapshot of training3304 checkpoint50, not native reload.

Only a future actual same-node invocation after completed OUTER50 may call main.
The script creates no remote files, submits no job and imports no model/Torch.
Stdout preserves small originals plus full hashes/physical metadata of large PT/HF.
"""
import base64
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys
import types

SOURCE='fcecbea5d939e41dcfb3b118a26ef049a9bda3d6'
CONFIG='0f0fc3a6aff4648ce0d2c8ec28eb104e27678315ad09824c90da043956a0b990'
COMMON='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
RUN=Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-unclipped-multitraj200-20261006/runs/formal-3304')
REPO=RUN.parent.parent/'repo-fcecbea5d939'
DRIVER=(2548319,1834784786)
WORKERS={0:(2548974,1834785562),1:(2549227,1834786108),2:(2549228,1834786110),3:(2549229,1834786111),
         4:(2549231,1834786113),5:(2549232,1834786114),6:(2549230,1834786112),7:(2549233,1834786115)}
DRIVER_PATH=Path('/tmp/o4e-3304/ray/session_2026-10-06_19-44-37_985734_2537562/logs/worker-0cdc9d8698e23448dcc06d212bbfba65513b4b4f2dfc726c2144806d-02000000-2548319.out')
TORCH_ROOT=Path('/mnt/data/shaoshitong/search-r1-rl/opsd/env/lib/python3.10/site-packages/torch')
FSDP_PATH=TORCH_ROOT/'distributed/fsdp/fully_sharded_data_parallel.py'
FSDP_SHA='ca839d30fda12e0b12dea934e1f1bf998bd80ed716e8e4c88927691a9bb1c6a3'
GATE_SHA='09561719673a7c273ad39b3c9363b1a250eab0a94c20c80f40dd1dfc05589e16'
RECEIPT='d1db79938fd1a3b3520e4d1da673525dd260302d2402e063fd7734fcb11721fc'
STEP=50
ATTRS=('st_dev','st_ino','st_uid','st_mode','st_nlink','st_size','st_mtime_ns','st_ctime_ns')
HF_SMALL={'added_tokens.json','config.json','generation_config.json','merges.txt','model.safetensors.index.json',
          'special_tokens_map.json','tokenizer.json','tokenizer_config.json','vocab.json'}
MAX_PREFIX=64*1024**2
MAX_SMALL=32*1024**2
MAX_MEMBER=32*1024**3


def need(ok,message):
    if not ok:raise ValueError(message)


def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def sha(v):return hashlib.sha256(v).hexdigest()
def sig(v):return tuple(getattr(v,k) for k in ATTRS)
def same(a,b,message):need(canonical(a)==canonical(b),message)


def parse(raw):
    def pairs(items):
        v={}
        for k,x in items:need(k not in v,'Duplicate JSON key');v[k]=x
        return v
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def opened(path,limit,installed_fsdp=False):
    path=Path(path);need(path.is_absolute() and '..' not in path.parts and path.resolve(strict=True)==path,'Physical absolute path required')
    s=path.lstat();allowed=2 if installed_fsdp and path==FSDP_PATH else 1
    need(stat.S_ISREG(s.st_mode) and s.st_uid==os.getuid() and s.st_nlink in (1,allowed) and s.st_size<=limit,'Owned regular file/link/size differs')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    try:need(sig(os.fstat(fd))==sig(s),'Opened identity differs')
    except BaseException:os.close(fd);raise
    return path,s,fd


def record(path,s,pin,size):
    return dict(physical_path=str(path),sha256=pin,bytes=size,uid=s.st_uid,device=s.st_dev,inode=s.st_ino,
        mode=stat.S_IMODE(s.st_mode),nlink=s.st_nlink,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)


def read_member(path,expected=None,keep=False,installed_fsdp=False):
    path,before,fd=opened(path,MAX_SMALL if keep else MAX_MEMBER,installed_fsdp);h=hashlib.sha256();n=0;parts=[]
    try:
        while True:
            chunk=os.read(fd,1024**2)
            if not chunk:break
            h.update(chunk);n+=len(chunk)
            if keep:parts.append(chunk)
        end=os.fstat(fd)
    finally:os.close(fd)
    after=path.lstat();need(sig(before)==sig(end)==sig(after) and path.resolve(strict=True)==path and n==before.st_size,'Immutable member changed during full read')
    need(expected is None or h.hexdigest()==expected,'Full member SHA differs')
    rec=record(path,before,h.hexdigest(),n)
    if keep:rec['base64']=base64.b64encode(b''.join(parts)).decode()
    return (b''.join(parts) if keep else None),rec


def log_prefix(path,after_first_read=None):
    """One fixed complete byte prefix; later append is permitted, never normalized."""
    path,before,fd=opened(path,MAX_PREFIX);size=before.st_size
    def read_fixed():
        os.lseek(fd,0,os.SEEK_SET);parts=[];left=size
        while left:
            block=os.read(fd,min(left,1024**2));need(block,'Log prefix shortened');parts.append(block);left-=len(block)
        return b''.join(parts)
    try:
        raw=read_fixed()
        if after_first_read is not None:after_first_read()
        second=read_fixed();end=os.fstat(fd)
    finally:os.close(fd)
    after=path.lstat();stable=('st_dev','st_ino','st_uid','st_mode','st_nlink')
    need(all(getattr(before,k)==getattr(v,k) for v in (end,after) for k in stable) and end.st_size>=size and after.st_size>=size
         and path.resolve(strict=True)==path and raw==second,'Log prefix replaced/overwritten/shortened')
    rec=record(path,before,sha(raw),size);rec.update(base64=base64.b64encode(raw).decode(),observed_size_after=after.st_size,
        prefix_scope='EXACT_CAPTURED_BYTES_DOUBLE_READ_APPEND_ALLOWED_NOT_WRITER_LIFETIME',last_lf_offset=raw.rfind(b'\n')+1)
    return raw,rec


def proc_identity(pid,start):
    need(type(pid) is int and type(start) is int,'Typed actual identity required')
    p=Path('/proc')/str(pid);a=(p/'stat').read_bytes();s=(p/'status').read_bytes();c=(p/'cgroup').read_bytes();b=(p/'stat').read_bytes()
    def clock(raw):return raw.decode().rsplit(')',1)[1].split()[19]
    need(clock(a)==clock(b)==str(start) and int(a.split(b' ',1)[0])==pid,'Original PID/start differs')
    need(re.search(rb'^Uid:\s+0\s+0\s+0\s+0\s*$',s,re.M),'Actual process UID differs')
    need(re.search(rb'^State:\s+[RSDI]',s,re.M),'Original process no longer live')
    need(any(re.search(r'(?:^|/)job[_-]3304(?:\.scope)?/step[_-]0(?:\.scope)?(?:/|$)',x.split(':',2)[-1]) for x in c.decode().splitlines()),'Original job/step differs')
    return dict(pid=pid,start_ticks=start,stat_original=a.decode(),status_original=s.decode(),cgroup=c.decode(),uid=0)


def command(argv):
    p=subprocess.Popen(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:out,err=p.communicate(timeout=60);waited=True
    except subprocess.TimeoutExpired:raise TimeoutError('Read-only command pending; no automatic retry')
    value=dict(argv=argv,pid=p.pid,waited=waited,returncode=p.returncode,stdout=out.decode(),stderr=err.decode())
    need(type(p.returncode) is int and p.returncode==0,'Read-only command did not truly wait0')
    return value


def load_gate(code):
    need(sha(code)==GATE_SHA,'Frozen gate bytes differ');m=types.ModuleType('checkpoint50_exact_gate')
    exec(compile(code,str(REPO/'search_r1/baselines/checkpoint_gate.py'),'exec',dont_inherit=True,optimize=0),m.__dict__);return m


def parse_events(raw,gate,warnings):
    events=[];partial=[];offset=0
    markers=('OPSD_E1_OUTER_JSON=','BASELINE_EVALUATION_METRICS_JSON=','FULL_STATE_RANK_JSON=','OPSD_E1_STEP5_RELEASED_JSON=')
    for line in raw.splitlines(keepends=True):
        for marker in markers:
            if marker.encode() not in line:continue
            try:v=gate.parse_log_marker(line.decode(),marker,driver_pid=DRIVER[0],job_id='3304',worker_pids=tuple(x[0] for x in WORKERS.values()),
                checkpoint_warning_path=TORCH_ROOT/'utils/checkpoint.py',fsdp_warning_source_bytes=warnings)
            except json.JSONDecodeError:
                need(not line.endswith(b'\n') and offset+len(line)==len(raw),'Malformed complete structured record')
                partial.append(dict(marker=marker,offset=offset,bytes=len(line),sha256=sha(line)));continue
            events.append(dict(marker=marker,value=v,offset=offset,bytes=len(line),sha256=sha(line),raw_base64=base64.b64encode(line).decode()))
        offset+=len(line)
    return events,partial


def progress(events,gate):
    rows=[e for e in events if e['marker']=='OPSD_E1_OUTER_JSON='];need(len(rows)>=STEP,'Actual completed OUTER50 absent')
    need(len(rows)<=200,'Unexpected outer budget');sums=[0]*8
    for i,event in enumerate(rows,1):
        v=event['value'];same({k:v.get(k) for k in ('actual_outer_updates','global_step','driver_pid','job_id','source_commit','config_sha256')},
            dict(actual_outer_updates=i,global_step=i,driver_pid=DRIVER[0],job_id='3304',source_commit=SOURCE,config_sha256=CONFIG),'Outer source/typed header/sequence differs')
        m=v['metrics'];gate.validate_outer(m,profile='multitraj200')
        need(math.isclose(m['actor/lr'],min(i/57,1)*1e-6,rel_tol=1e-12) and math.isclose(m['actor/optimizer_lr_used'],min((i-1)/57,1)*1e-6,rel_tol=1e-12,abs_tol=1e-18),'Actual warmup clock differs')
        for r in range(8):sums[r]+=m[f'actor/optimizer_steps/rank_{r}']
        if i==STEP:at50=dict(event=event,optimizer_calls_per_rank=sums.copy(),expected_questions_consumed=512*STEP,
            data_cursor_scope='SOURCE_COMMIT_CONSUMED_512_PER_OUTER_NOT_NATIVE_DRIVER_PT_READBACK',actual_rng_contents=None)
    same(at50['optimizer_calls_per_rank'],[500.0]*8,'Expected actual500 optimizer calls per rank differs')
    released=[e['value'] for e in events if e['marker']=='OPSD_E1_STEP5_RELEASED_JSON=']
    same(released,[dict(driver_pid=DRIVER[0],actual_outer_updates=5,next_global_step=6,receipt_sha256=RECEIPT)],'Unique actual first5 release differs')
    return at50,rows


def evaluations_and_console(raw,events,rows,gate):
    before50=[e for e in events if e['offset']<rows[STEP-1]['offset'] and e['marker']=='BASELINE_EVALUATION_METRICS_JSON=']
    need(len(before50)==4,'Expected exact dev0/stage5/dev50/stage50 event sequence')
    for e,count in zip(before50,[512,64,512,64]):gate.validate_evaluation(e['value'],count)
    m=rows[STEP-1]['value']['metrics'];dev=before50[2]['value'];stage=before50[3]['value']
    same({k:v for k,v in m.items() if k.startswith('val/')},dev,'Actual dev50 vsOUTER50 payload differs')
    same({k[len('opsd_stage/'):]:v for k,v in m.items() if k.startswith('opsd_stage/')},stage,'Actual stage50 vsOUTER50 differs')
    console=[];offset=0
    for line in raw.splitlines(keepends=True):
        if line.startswith(b'step:50 - '):
            need(line.endswith(b'\n'),'Console50 record must be a complete LF line');parts=line.decode().strip().split(' - ');view={}
            for item in parts[1:]:
                k,sep,value=item.rpartition(':');need(sep and k not in view,'Malformed/duplicate console metric');view[k]=value
            for k,v in {**dev,**{'opsd_stage/'+k:v for k,v in stage.items()}}.items():
                need(view.get(k)==format(float(v),'.3f'),'Console50 rounded dev/stage metric differs')
            need(offset<rows[STEP-1]['offset'] and offset>before50[3]['offset'],'Console50 order differs')
            console.append(dict(offset=offset,bytes=len(line),sha256=sha(line),raw_base64=base64.b64encode(line).decode(),comparison='ORIGINAL_CONSOLE_3_DECIMAL_DISPLAY_NOT_FULL_PRECISION'))
        offset+=len(line)
    need(len(console)==1,'Exactly one console50 record required')
    return dict(dev0=before50[0],stage5=before50[1],dev50=before50[2],stage50=before50[3],console50=console[0],
        comparison_scope='DEV0_VS_DEV50_SAME512_AND_STAGE5_VS_STAGE50_SAME64_DISTINCT_COHORTS_NO_CROSS_COMPARISON')


def saved_members(files):
    actor=RUN/'checkpoints/actor/global_step_50';need(actor.resolve(strict=True)==actor and actor.is_dir(),'Actual HF50 directory absent')
    entries=list(actor.iterdir());regular=set();directories=set()
    for entry in entries:
        s=entry.lstat();need(entry.resolve(strict=True)==entry and s.st_uid==os.getuid(),'Checkpoint symlink/ownership differs')
        if stat.S_ISREG(s.st_mode):regular.add(entry.name)
        elif stat.S_ISDIR(s.st_mode):directories.add(entry.name)
        else:raise ValueError('Unsupported checkpoint entry')
    same(sorted(directories),['training_state'],'Unexpected HF50 subdirectory')
    index,rec=read_member(actor/'model.safetensors.index.json',keep=True);files[str(actor/'model.safetensors.index.json')]=rec
    idx=parse(index);need(type(idx['weight_map']) is dict and len(idx['weight_map'])==435,'HF50 index435 keys absent')
    shards=set(idx['weight_map'].values());need(len(shards)==3 and all(re.fullmatch(r'model-[0-9]{5}-of-00003\.safetensors',n) for n in shards),'Actual HF50 shard names differ')
    same(sorted(regular),sorted(HF_SMALL|shards),'Actual HF50 immediate member set differs')
    for n in sorted(regular-{'model.safetensors.index.json'}):files[str(actor/n)]=read_member(actor/n,keep=n in HF_SMALL)[1]
    same(sorted(p.name for p in (actor/'training_state').iterdir()),sorted(f'rank_{r}{ext}' for r in range(8) for ext in ('.json','.pt')),'Exact eight rank PT/JSON members required')
    reports=[]
    for r in range(8):
        directory=actor/'training_state';raw,rec=read_member(directory/f'rank_{r}.json',keep=True);report=parse(raw);files[str(directory/f'rank_{r}.json')]=rec
        same({k:report.get(k) for k in ('version','algorithm','status','rank','world_size','step','pid','scheduler_last_epoch','optimizer_state_entries','model_dtensor_entries')},
            dict(version=3,algorithm='pure_opsd',status='FULL_STATE_SERIALIZED_OK',rank=r,world_size=8,step=50,pid=WORKERS[r][0],scheduler_last_epoch=50,optimizer_state_entries=37,model_dtensor_entries=435),'Actual50 rank serialized report differs')
        same(report.get('path'),str(directory/f'rank_{r}.pt'),'Rank PT physical report path differs')
        same(report['sha256'],report['state_sha256'],'Rank report two SHA fields differ')
        files[str(directory/f'rank_{r}.pt')]=read_member(directory/f'rank_{r}.pt',report['state_sha256'])[1]
        same(report['teacher']['counters']['cuda_update_entries'],50,'Saved teacher50 clock differs');reports.append(report)
    driver_dir=RUN/'full_state/global_step_50';same(sorted(p.name for p in driver_dir.iterdir()),['driver.json','driver.pt'],'Exact driver saved member set required')
    raw,rec=read_member(driver_dir/'driver.json',keep=True);driver=parse(raw);files[str(driver_dir/'driver.json')]=rec
    same({k:driver.get(k) for k in ('status','step','pid','uid','job_id')},dict(status='DRIVER_STATE_SERIALIZED_OK',step=50,pid=DRIVER[0],uid=0,job_id='3304'),'Actual driver50 serialized report differs')
    files[str(driver_dir/'driver.pt')]=read_member(driver_dir/'driver.pt',driver['state_sha256'])[1]
    algorithm_path=RUN/'checkpoints/algorithm/global_step_50/state.json';raw,rec=read_member(algorithm_path,keep=True);algorithm=parse(raw);files[str(algorithm_path)]=rec
    same({k:algorithm.get(k) for k in ('version','method','global_step','outcome_rewards_train_policy')},dict(version=1,method='pure_opsd',global_step=50,outcome_rewards_train_policy=False),'Saved algorithm50 identity differs')
    same(algorithm['trajectory_archive']['completed_outer'],50,'Algorithm archive50 progress differs')
    return dict(member_count=31,hf435_index=idx,rank_reports=reports,driver_report=driver,algorithm=algorithm,
        native_reload=None,whole_hf_load=None,rng_content_readback=None,sampler_content_readback=None,
        scope='ACTUAL_STORED_MEMBERS_FULL_BYTE_HASH_AND_SERIALIZED_REPORT_JOIN_NOT_NATIVE_RELOAD_OR_TENSOR_VERIFICATION')


def collect(result):
    need(sys.platform=='linux' and os.getuid()==0 and socket.gethostname().split('.')[0]=='h100-10','Actual same-node UID0 collector only')
    result['driver_before']=proc_identity(*DRIVER);result['workers_before']={str(r):proc_identity(*v) for r,v in WORKERS.items()}
    need(Path(os.readlink(f'/proc/{DRIVER[0]}/fd/1'))==DRIVER_PATH,'Actual own driver stdout origin differs')
    for label,argv in [('git_head',['git','-C',str(REPO),'rev-parse','HEAD']),('git_clean',['git','-C',str(REPO),'status','--porcelain','--untracked-files=all']),('slurm',['scontrol','show','job','3304','-o'])]:result['commands'][label]=command(argv)
    same(result['commands']['git_head']['stdout'].strip(),SOURCE,'Actual source HEAD differs');need(not result['commands']['git_clean']['stdout'].strip(),'Actual source is dirty')
    for token in ['JobId=3304 ','UserId=root(0)','JobState=RUNNING','Restarts=0','NodeList=h100-10','gres:gpu:h100:8']:need(token in result['commands']['slurm']['stdout'],'Literal own Slurm identity differs')
    for relative,pin in SOURCE_FILES.items():result['source_files'][relative]=read_member(REPO/relative,pin,keep=True)[1]
    code=base64.b64decode(result['source_files']['search_r1/baselines/checkpoint_gate.py']['base64'],validate=True);gate=load_gate(code)
    warnings,wrec=read_member(FSDP_PATH,FSDP_SHA,keep=True,installed_fsdp=True);need(wrec['bytes']==99345,'Exact native warning source length differs');result['files'][str(FSDP_PATH)]=wrec
    for path,pin in [(RUN/'resolved-config.json',CONFIG),(RUN/'bindings/data_manifest.json',COMMON),(RUN/'full_state/acceptance-5-receipt.json',RECEIPT)]:result['files'][str(path)]=read_member(path,pin,keep=True)[1]
    raw,rec=log_prefix(DRIVER_PATH);result['files']['driver-current-prefix.out']=rec;events,partial=parse_events(raw,gate,warnings);at50,rows=progress(events,gate)
    result['metrics']=at50;result['evaluations']=evaluations_and_console(raw,events,rows,gate);result['driver_prefix_complete_outer']=len(rows);result['driver_incomplete_eof']=partial
    train,tr=log_prefix(RUN/'train.log');result['files']['train.log']=tr;te,tp=parse_events(train,gate,warnings);aggregate=[x for x in te if x['marker']=='OPSD_E1_OUTER_JSON=']
    need(len(aggregate)<=len(rows),'Aggregate is ahead of captured driver prefix')
    same([x['value'] for x in aggregate],[x['value'] for x in rows[:len(aggregate)]],'Aggregate completed prefix differs from driver')
    result['aggregate_prefix_complete_outer']=len(aggregate);result['aggregate_incomplete_eof']=tp;result['aggregate_lag_is_not_training_failure']=len(rows)-len(aggregate)
    result['stored50']=saved_members(result['files'])
    # Current31 contents may be large; reread small reports/index and current inode
    # metadata after all whole hashes to prevent an inconsistent collected graph.
    for path,rec in result['files'].items():
        if path in ('driver-current-prefix.out','train.log'):continue
        p=Path(path);s=p.lstat();need(p.resolve(strict=True)==p and all(rec[k]==v for k,v in dict(device=s.st_dev,inode=s.st_ino,uid=s.st_uid,nlink=s.st_nlink,bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns).items()),'Collected immutable graph member changed')
    result['driver_after']=proc_identity(*DRIVER);result['workers_after']={str(r):proc_identity(*v) for r,v in WORKERS.items()}
    result['commands']['git_head_after']=command(['git','-C',str(REPO),'rev-parse','HEAD']);result['commands']['git_clean_after']=command(['git','-C',str(REPO),'status','--porcelain','--untracked-files=all'])
    same(result['commands']['git_head_after']['stdout'].strip(),SOURCE,'Final source HEAD differs');need(not result['commands']['git_clean_after']['stdout'].strip(),'Final source is dirty')
    result['status']='READONLY_ACTUAL50_LOG_AND_STORED_FILE_SNAPSHOT_REQUIRES_INDEPENDENT_REVIEW'


def main():
    result=dict(version=1,kind='TRAINING3304_CHECKPOINT50_READONLY_ORIGINALS',status='COLLECTION_INCOMPLETE',
        observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_commit=SOURCE,job_id='3304',step=50,
        files={},source_files={},commands={},limits=['Full file SHA and serialized reports do not prove native reload, tensor/sampler/RNG contents or HF model forward.',
        'Driver/train are fixed captured prefixes with append allowed; they are not closed terminal logs.','Dev512 and stage64 are separate cohorts; comparisons stay within each.',
        'No checkpoint50 pause, manifest receipt, optimizer change, GPU diagnostic, resource release or terminal success is created.'])
    try:collect(result)
    except BaseException as exc:
        result['error']=dict(type=type(exc).__name__,message=str(exc));print(json.dumps(result,allow_nan=False,sort_keys=True),flush=True);raise
    print(json.dumps(result,allow_nan=False,sort_keys=True),flush=True)


if __name__=='__main__':main()
