"""3315 terminal-only, read-only resource/cost/cleanup review candidate. No signals, sampling or launch."""
import datetime
import hashlib
import json
import math
import re
from pathlib import Path

SOURCE='60f566b3ddabc88ce9de56ead5ad30a3d6795dd2'
ROOT=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-controller-initfix-20261006')
NONCE='bab8981d9fd14afc98923eb4ebb89e33'
JOB=3315
RUN=ROOT/'runs'/('formal200-'+NONCE+'-3315')
COMMENT='BAPO_FRESH200:formal200:'+SOURCE+':'+NONCE
TOOLS={'guard_journal.py':'a7770927930814e1cb8c39263696da1a493d52ad74fe023a090b73574f6ad6c5',
       'resource_watch.py':'c5fd9902ac5977978349d47709ffb40008bb93424dfcc4aa2dfd801e8df70749',
       'owned_processes.py':'2c8cdb3363461b402c3155091cab89739524256b1f89ab58c0f5e62334a877bd',
       'stage.sh':'bb17782f408049778b0206c7b870cee8a64b8347b0066076add37895118fa6bf'}


def need(ok,msg):
    if not ok: raise ValueError(msg)


def digest(raw):return hashlib.sha256(raw).hexdigest()


def strict_json(raw):
    def pairs(items):
        out={}
        for key,val in items:
            need(key not in out,'duplicate JSON key');out[key]=val
        return out
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite JSON')))


def integer(x):return type(x)is int and x>=0


def number(x):return type(x)in(int,float) and math.isfinite(x) and x>=0


def terminal_slurm(record):
    need(type(record.get('job_id'))is int and record['job_id']==JOB and type(record.get('returncode'))is int and record['returncode']==0,'actual Slurm collector identity/exit')
    text=record.get('stdout');need(isinstance(text,str),'actual Slurm raw stdout')
    fields={}
    for match in re.finditer(r'(?:^|\s)([A-Za-z][A-Za-z0-9:/]*)=([^\s]*)',text):
        key,val=match.groups();need(key not in fields,'duplicate Slurm field');fields[key]=val
    expected={'JobId':'3315','JobState':'COMPLETED','ExitCode':'0:0','Restarts':'0','Requeue':'0',
              'NodeList':'h100-5','NumNodes':'1','BatchHost':'h100-5','Comment':COMMENT,
              'Command':str(ROOT/'tools/formal200.sbatch'),'WorkDir':str(ROOT),
              'SuspendTime':'None','SecsPreSuspend':'0'}
    for key,val in expected.items():need(fields.get(key)==val,'terminal Slurm mismatch '+key)
    start=datetime.datetime.fromisoformat(fields['StartTime']);end=datetime.datetime.fromisoformat(fields['EndTime'])
    need(start.tzinfo is None and end.tzinfo is None and end>start,'actual Shanghai terminal times')
    duration=fields['RunTime'];match=re.fullmatch(r'(?:(\d+)-)?(\d+):(\d\d):(\d\d)',duration)
    need(match is not None,'actual Slurm RunTime format')
    day,hour,minute,second=match.groups();minute=int(minute);second=int(second)
    need(minute<60 and second<60,'invalid elapsed fields')
    elapsed=int(day or 0)*86400+int(hour)*3600+minute*60+second
    need(elapsed==int((end-start).total_seconds()) and elapsed>0,'Slurm elapsed/date mismatch')
    # Some cluster Slurm versions omit gres in TRES. Allocation is independently
    # proved by all journal samples' actual raw inventory + UUID leases.
    for key in ('TRES','AllocTRES','TresPerNode'):
        if key in fields:
            matches=re.findall(r'(?:gres/gpu|gres:gpu)(?::[A-Za-z0-9_-]+)?(?:=|:)(\d+)',fields[key])
            need(not matches or all(int(n)==8 for n in matches),'Slurm GPU allocation disagrees')
    return dict(start_shanghai=fields['StartTime'],end_shanghai=fields['EndTime'],elapsed_seconds=elapsed,
                allocated_gpu_count=8,reserved_gpu_hours=elapsed*8/3600,
                cost_scope='SLURM_WALL_TIME_X_8_RESERVED_GPUS_NOT_UTILIZATION_GPU_HOURS_NOT_PRICE')


def check_cleanup(cleanup,epochs,guard,exit_codes,errors):
    need(cleanup.get('own_marker')==str(RUN) and cleanup.get('remaining')==[],'own cleanup scope/remaining')
    stopped=cleanup.get('stopped_pids')
    need(isinstance(stopped,list) and all(type(p)is int and p>0 for p in stopped) and len(set(stopped))==len(stopped),'cleanup PID list invalid')
    need(all(p in epochs and len(epochs[p])==1 for p in stopped),'stopped PID missing or ambiguous historical epoch')
    need(all(value.strip()=='0' for value in exit_codes.values()) and set(exit_codes)=={'original','cleanup'},'original/cleanup exit not zero')
    need(all(not raw.strip() for raw in errors.values()) and set(errors)=={'cleanup','gpu_exit'},'cleanup/GPU exit error evidence')
    need(guard.get('status')=='PASS_ACTUAL_GUARD_JOURNAL_REPLAY' and guard.get('source_commit')==SOURCE
         and guard.get('job_id')==JOB and guard.get('run_root')==str(RUN),'fixed full-journal replay identity')
    need(guard.get('unfinished_recovery')is False and guard.get('abort_events')==0,'unsafe/unfinished recovery')
    uuids=guard.get('allocated_uuids');need(isinstance(uuids,list) and len(uuids)==8 and len(set(uuids))==8,'full eight UUID allocation')
    for key in ('samples','recovery_samples','freeze_events','safe_resume_events','prelaunch_recoveries','abort_events'):
        need(integer(guard.get(key)),'missing/invalid recovery count '+key)
    need(guard['samples']>0 and number(guard.get('cost_wall_seconds')),'missing full journal/recovery duration')
    return dict(stopped_pid_epochs={str(p):next(iter(epochs[p])) for p in stopped},remaining=[],
                scope='FIXED_OWN_RUN_AND_JOB_UID_PIDFD_CLEANUP_ONLY_NOT_FOREIGN_GPU_CLEARING',allocated_uuids=uuids)


def check_leases(collection,uuids):
    need(collection.get('status')=='ACTUAL_FIRST5_COLLECTION_REQUIRES_INDEPENDENT_REVIEW'
         and collection.get('source_commit')==SOURCE and collection.get('job_id')==JOB
         and collection.get('run_root')==str(RUN),'actual first5 collection identity')
    proof=collection.get('leases',{});guard=proof.get('guard',{});records=proof.get('leases')
    need(proof.get('verification')=='read_only_proc_fd_inode_and_proc_locks_no_flock_mutation','actual historical read-only FD/FLOCK verification absent')
    need(type(guard.get('pid'))is int and guard['pid']>0 and type(guard.get('start_ticks'))is int and guard['start_ticks']>0,'actual guard PID epoch')
    env=guard.get('selected_env',{})
    need(env.get('BAPO_RUN_ID')==str(RUN) and env.get('SLURM_JOB_ID')=='3315','historical guard ownership mismatch')
    need(isinstance(records,list) and len(records)==8 and {r.get('uuid')for r in records}==set(uuids),'lease UUID set mismatch')
    need(len({r.get('fd')for r in records})==8,'duplicate lease FD')
    for rec in records:
        need(integer(rec.get('fd')) and type(rec.get('inode'))is int and rec['inode']>0,'invalid lease FD/inode')
        tokens=rec.get('exclusive_lock_record','').split()
        need(len(tokens)>=6 and tokens[1:4]==['FLOCK','ADVISORY','WRITE']
             and tokens[4]==str(guard['pid']) and tokens[5].split(':')[-1]==str(rec['inode']), 'historical lock owner/inode mismatch')
    return dict(scope='ACTUAL_FIRST5_HISTORICAL_FD_INODE_FLOCK_PROOF_NOT_POSTTERMINAL_LOCK_HOLD',
                guard_pid=guard['pid'],guard_start_ticks=guard['start_ticks'],allocated_uuids=sorted(uuids))


def check_gpu_exit(raw,uuids):
    import csv,io
    rows=list(csv.reader(io.StringIO(raw.decode())))
    need(rows and len(rows)>1,'actual GPU exit inventory missing')
    header=[h.strip()for h in rows[0]]
    need(header.count('uuid')==1,'GPU exit UUID header missing/ambiguous')
    column=header.index('uuid');observed=[]
    for row in rows[1:]:
        need(len(row)==len(header),'truncated GPU exit row')
        observed.append(row[column].strip())
    need(len(observed)==8 and set(observed)==set(uuids),'GPU exit allocation changed')
    return dict(gpu_count=8,uuid_set=sorted(observed),foreign_process_absence_inferred=False)


def historical_epochs(rows,monitor):
    epochs={}
    def add(pid,tick):
        need(type(pid)is int and pid>0 and type(tick)is int and tick>0,'invalid PID epoch')
        epochs.setdefault(pid,set()).add(tick)
    for row in rows:
        need(row.get('run')==str(RUN) and row.get('slurm_job_id')=='3315','journal ownership changed')
        for pid,proof in row.get('owned_before_query',{}).items():
            need(proof.get('run')==str(RUN) and proof.get('slurm_job_id')=='3315','own proof identity changed')
            add(int(pid),proof['start_ticks'])
        cert=row.get('certificate')
        if cert is not None:add(cert['pid'],cert['start_ticks'])
    previous=-1
    for row in monitor:
        need(number(row.get('time')) and row['time']>=previous,'monitor time invalid');previous=row['time']
        need(isinstance(row.get('processes'),list),'monitor process records missing')
        for process in row['processes']:add(process['pid'],process['start_ticks'])
    need(epochs,'no original process epoch evidence')
    return epochs


def main():
    import argparse,os,time,types
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--slurm-before',type=Path,required=True)
    parser.add_argument('--slurm-after',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--first5-collection',type=Path,required=True)
    parser.add_argument('--expected-first5-collection-sha',required=True)
    args=parser.parse_args();wall=time.perf_counter();cpu=time.process_time();raws={}
    def read(path):
        path=Path(path);need(path.is_absolute() and path.resolve()==path and path.is_file() and not path.is_symlink(),'canonical original file required')
        raw=path.read_bytes();raws[str(path)]=raw;return raw
    before=terminal_slurm(strict_json(read(args.slurm_before)))
    for name,sha in TOOLS.items():need(digest(read(ROOT/'tools'/name))==sha,'fixed tool bytes changed '+name)
    need(read(RUN/'source-commit.txt').decode().strip()==SOURCE,'run source identity mismatch')
    # Original fixed guard validates full raw monitor math, ownership, inherited
    # UUID set, freeze/resume certificates and complete state machine. No adapter
    # or last-record substitute. It also validates actual runtime/source anchor.
    module=types.ModuleType('fixed_terminal_guard');module.__file__=str(ROOT/'tools/guard_journal.py')
    exec(compile(raws[module.__file__],module.__file__,'exec'),module.__dict__)
    journal=read(RUN/'resource-watch.jsonl')
    need(journal.endswith(b'\n'),'partial terminal journal')
    guard=module.replay(RUN/'resource-watch.jsonl',JOB,str(RUN),SOURCE,ROOT/'tools/resource_watch.py')
    rows=[strict_json(line)for line in journal.splitlines()]
    monitorraw=read(RUN/'cpu-process-monitor.jsonl');need(monitorraw.endswith(b'\n'),'partial CPU monitor')
    monitor=[strict_json(line)for line in monitorraw.splitlines()]
    epochs=historical_epochs(rows,monitor)
    cleanup=strict_json(read(RUN/'own-process-cleanup.json'))
    exit_codes={'original':read(RUN/'original-exit-code.txt').decode(),'cleanup':read(RUN/'cleanup-exit-code.txt').decode()}
    errors={'cleanup':read(RUN/'own-process-cleanup.err'),'gpu_exit':read(RUN/'gpu-exit.err')}
    clean=check_cleanup(cleanup,epochs,guard,exit_codes,errors)
    collection_raw=read(args.first5_collection)
    need(digest(collection_raw)==args.expected_first5_collection_sha,'actual first5 lease collection binding differs')
    collection=strict_json(collection_raw)
    lease=check_leases(collection,guard['allocated_uuids'])
    driver=collection.get('driver_process',{})
    need(type(driver.get('pid'))is int and type(driver.get('start_ticks'))is int
         and epochs.get(driver['pid'])=={driver['start_ticks']},'same original driver epoch missing/changed')
    need(guard.get('own_uid')==os.getuid(),'reviewer own UID differs from guarded allocation')
    # Bind all original text logs without touching model/optimizer shards.
    log_suffixes={'.log','.jsonl','.csv','.txt','.err'}
    extra_logs={};log_stats={}
    for path in sorted(RUN.rglob('*')):
        if path.suffix not in log_suffixes or not path.is_file():continue
        need(not path.is_symlink() and path.resolve()==path,'original log symlink refused')
        before_stat=path.stat();h=hashlib.sha256()
        with path.open('rb')as stream:
            for chunk in iter(lambda:stream.read(1048576),b''):h.update(chunk)
        after_stat=path.stat()
        need((before_stat.st_ino,before_stat.st_size,before_stat.st_mtime_ns)==(after_stat.st_ino,after_stat.st_size,after_stat.st_mtime_ns),'terminal log changed during hashing')
        extra_logs[str(path)]={'sha256':h.hexdigest(),'bytes':after_stat.st_size}
        log_stats[path]=(after_stat.st_ino,after_stat.st_size,after_stat.st_mtime_ns)
    need(str(RUN/'train.log') in extra_logs and str(RUN/'phase.log') in extra_logs,'complete training/phase logs absent')
    # Read actual GPU-exit output as evidence; foreign GPUs/processes need not be empty.
    gpu_exit=check_gpu_exit(read(RUN/'gpu-exit.csv'),guard['allocated_uuids'])
    # Read-only postterminal own-marker absence; never signal any process.
    remaining=[];unknown=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdecimal():continue
        try:
            env=(entry/'environ').read_bytes().split(b'\0')
            if ('BAPO_RUN_ID='+str(RUN)).encode() in env and b'SLURM_JOB_ID=3315' in env:
                remaining.append(int(entry.name))
        except (OSError,PermissionError):
            if entry.exists():unknown.append(int(entry.name))
    need(not remaining and not unknown,'own-marker absence cannot be proved: remaining/inaccessible processes')
    after=terminal_slurm(strict_json(read(args.slurm_after)));need(before==after,'terminal Slurm identity changed')
    for path,raw in raws.items():need(Path(path).read_bytes()==raw,'artifact changed during terminal replay')
    for path,expected in log_stats.items():
        stat=path.stat();need((stat.st_ino,stat.st_size,stat.st_mtime_ns)==expected,'terminal text log changed after hashing')
    result=dict(status='TERMINAL_RESOURCE_COST_CLEANUP_SCOPE_PASS_NOT_WHOLE_TRAINING_ACCEPTANCE',source_commit=SOURCE,
        job_id=JOB,run_root=str(RUN),slurm=after,resource_full_journal=guard,cleanup=clean,historical_lease_proof=lease,gpu_exit_inventory=gpu_exit,
        all_original_text_logs=extra_logs,original_artifacts={p:{'sha256':digest(raw),'bytes':len(raw)}for p,raw in raws.items()},
        reviewer_cpu_seconds=time.process_time()-cpu,reviewer_wall_seconds=time.perf_counter()-wall,
        whole_training_accepted=False,native_training_accepted=False,release_authorized=False,
        limitations=['resource samples cover recorded windows, not continuous GPU utilization',
                     'cleanup recorded stopped PID list has no per-signal epoch; matched historical epochs and fixed pidfd selector code',
                     'price and utilization GPU-hours are not inferred',
                     'full 200/native/HF/sampling/first5/history/final evaluation remain separate acceptance'])
    with args.output.open('x')as handle:json.dump(result,handle,indent=2,allow_nan=False)
    print(result['status'])


if __name__=='__main__':main()
