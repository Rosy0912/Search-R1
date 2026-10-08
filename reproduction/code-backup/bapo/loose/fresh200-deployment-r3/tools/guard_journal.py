"""Read-only replay of the fixed recovery watcher's raw journal. Never samples/signals."""
import hashlib,json,math,re,types
from pathlib import Path
NEW_ROOT=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-safety-r3-20261006')
FRESH200_SOURCE='379b14c65e127cd3a71524f7783deaebafbbdb41'  # No launch/replay admission until the new skip source is bound.
WATCH_SHA='da34367b9b3c263a9bc47c5e7b7b42c0e873be7bbd87c612c8cf7dfe7fd246d7'
class JournalError(ValueError):pass
def need(value,message):
    if not value:raise JournalError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def integer(x,positive=False):return type(x)is int and (x>0 if positive else x>=0)
def finite(x):return type(x)in(int,float)and math.isfinite(x)and x>=0
def read_json(line):
    def pairs(items):
        d={}
        for k,v in items:
            need(k not in d,'Duplicate JSON key');d[k]=v
        return d
    return json.loads(line,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(JournalError('Nonfinite JSON')))
def load_bound_watcher(path):
    w=types.ModuleType('bound_journal_watcher');w.__file__=str(path)
    exec(compile(path.read_bytes(),str(path),'exec'),w.__dict__)
    return w
def replay(path,job_id,run_root,source_commit,watcher_path):
    need(integer(job_id,True),'Positive job ID');run=str(Path(run_root));need(Path(run).is_absolute()and Path(run).parent==NEW_ROOT/'runs','Exact new recovery ROOT/run required; legacy journals refused')
    need(FRESH200_SOURCE is not None and source_commit==FRESH200_SOURCE,'Exact frozen Fresh200 R3 source SHA required')
    path=Path(path);watcher_path=Path(watcher_path)
    for p in (path,watcher_path):need(p.is_absolute()and p==p.resolve()and p.is_file(),'Absolute regular nonsymlink artifact')
    need(path==Path(run)/'resource-watch.jsonl' and watcher_path==NEW_ROOT/'tools/resource_watch.py','Exact Fresh200 journal/watcher scope')
    need(sha(watcher_path)==WATCH_SHA,'Fixed reviewed watcher SHA')
    w=load_bound_watcher(watcher_path)
    initial_anchor=w.load_runtime_anchor(run,verify_current=False)
    need(initial_anchor['source_commit']==source_commit,'Journal source differs from actual runtime anchor')
    raw=path.read_bytes();lines=raw.decode().splitlines();need(lines and all(lines),'Empty/partial journal')
    rows=[read_json(line)for line in lines];need(all(isinstance(r,dict)for r in rows),'Object rows')
    uuids=None;uid=None;latest_proofs={};previous_time=-1.;phase=None;anchor=initial_anchor;title_transitions={};pending=False;active=False;prelaunch=False;validated=False;resuming=False;safe=0;held={};certificates={};resumed=set();live_frozen=set();stable=False;cost=0.;started_time=None;grace=None
    counts=dict(samples=0,recovery_samples=0,freeze_events=0,safe_resume_events=0,prelaunch_recoveries=0,abort_events=0)
    def sample(row):
        nonlocal uuids,uid,latest_proofs
        data=row
        unknown=row.get('reason')=='monitor_unknown_fail_closed'
        if unknown:
            need(isinstance(row.get('diagnostic'),dict),'Unknown requires original diagnostics')
            attempts=row['diagnostic'].get('samples');need(isinstance(attempts,list)and attempts,'Unknown needs full raw attempts');data=attempts[-1]
        required=('raw_inventory','raw_compute','raw_pmon','raw_nvml','allocated_gpus','window_start_us','window_end_us','owned_before_query','pid_start_ticks')
        need(all(k in data for k in required),'Missing raw resource proof')
        a=data['allocated_gpus'];need(isinstance(a,dict)and len(a)==8,'Eight allocated GPUs')
        actual=w.inventory(data['raw_inventory'],list(a))
        for record in actual.values():
            n=data['raw_nvml'].get(record['uuid'],{})
            if 'whole_gpu_sm_percent'in n:record.update(whole_gpu_sm_percent=w.number(n['whole_gpu_sm_percent'],'whole SM'),whole_gpu_sm_source='nvml_query',whole_gpu_sm_query_us=n.get('whole_gpu_sm_query_us'))
            else:record['whole_gpu_sm_source']='nvidia_smi_inventory_query'
        need(json.dumps(actual,sort_keys=True,allow_nan=False)==json.dumps(a,sort_keys=True,allow_nan=False),'Declared inventory differs from raw')
        current={x['uuid']for x in a.values()};need(len(current)==8,'Unique UUIDs')
        if uuids is None:uuids=current
        need(current==uuids,'GPU UUID set changed')
        need(integer(data['window_start_us'],True)and integer(data['window_end_us'],True)and data['window_end_us']>=data['window_start_us'],'Invalid window')
        proofs=data['owned_before_query'];need(isinstance(proofs,dict)and proofs,'Own identity proofs required')
        for p,proof in proofs.items():
            need(p.isdecimal()and integer(proof.get('start_ticks'),True)and integer(proof.get('uid')),'Own PID epoch/UID')
            need(proof.get('run')==run and proof.get('slurm_job_id')==str(job_id)and isinstance(proof.get('lease_uuids'),list)and len(proof['lease_uuids'])==8 and set(proof['lease_uuids'])==uuids,'Own allocation proof')
            if uid is None:uid=proof['uid']
            need(proof['uid']==uid,'Own UID changed')
        latest_proofs=proofs
        sm=w.parse_pmon(data['raw_pmon'],a);mem=w.parse_memory(data['raw_compute'],a);fresh,_=w.fresh_nvml(a,data['raw_nvml'],data['window_start_us'])
        for _,pid in set(sm)|set(mem)|set(fresh):
            tick=data['pid_start_ticks'].get(str(pid),{})
            need(integer(tick.get('before'),True)and tick['before']==tick.get('after'),'GPU PID lifecycle not stable')
            if str(pid)in proofs:
                need(proofs[str(pid)]['start_ticks']==tick['before'],'Own PID epoch differs')
                if active and not prelaunch:need(pid in held and held[pid]==tick['before'],'Recovery GPU own PID not certified frozen')
        failure=None
        try:result=w.assess_with_bounds(a,sm,mem,lambda p:str(p)in proofs,data['raw_nvml'],data['window_start_us'])
        except Exception as e:failure=e
        if unknown:
            need(failure is not None,'Unknown cannot hide a recomputed safe assessment')
            need(row.get('pause')is True and data.get('error')and str(failure)in data['error'],'Unknown diagnostic mismatch')
            return False
        need(failure is None,'Declared assessment cannot suppress raw failure')
        for key in ('external_processes','external_totals','pause','exceeded'):need(json.dumps(row.get(key),sort_keys=True,allow_nan=False)==json.dumps(result[key],sort_keys=True,allow_nan=False),'Resource result mismatch '+key)
        need(type(row.get('pause'))is bool,'Explicit pause bool')
        need(row.get('reason')==('external_occupancy_exceeds_15_percent'if result['pause']else'within_threshold'),'Reason mismatch')
        return not result['pause']
    for row in rows:
        need(row.get('run')==run and row.get('slurm_job_id')==str(job_id),'Wrong journal job/run')
        need(finite(row.get('time'))and row['time']>=previous_time,'Nonmonotonic event time');previous_time=row['time']
        p=row.get('phase_pid');need(p is None or integer(p,True),'Invalid phase PID')
        if phase is not None:need(p==phase,'Phase PID changed')
        elif p is not None:
            need(not active,'Phase cannot appear during prelaunch recovery');phase=p
        event=row.get('event')
        if title_transitions:need(event=='owned_freeze_signal'and row.get('pid')in title_transitions,'Title transition must immediately match STOP in current freeze')
        if event is None:
            need(not active and not pending,'Normal sample inside incomplete recovery')
            counts['samples']+=1;pending=not sample(row)
        elif event=='recovery_started':
            need(pending and not active and not title_transitions,'Recovery requires unsafe sample and no pending title proof');prelaunch=phase is None
            need(finite(row.get('grace_seconds'))and 1<=row['grace_seconds']<=120,'Bounded grace')
            active=True;pending=False;safe=0;held={};certificates={};resumed=set();live_frozen=set();stable=False;validated=False;resuming=False;started_time=row['time'];grace=row['grace_seconds']
        elif event=='owned_title_transition':
            need(active and not prelaunch and not validated and not resuming,'Title transition outside current unvalidated owned freeze')
            before=row.get('before_certificate');after=row.get('after_certificate')
            need(isinstance(before,dict)and isinstance(after,dict)and before.get('pid')==after.get('pid')==row.get('pid'),'Title transition PID proof')
            if anchor is None:anchor=w.load_runtime_anchor(run,verify_current=False)
            need(w.certificate_identity(before,anchor)==w.certificate_identity(after,anchor),'Title transition changes immutable identity')
            need(before.get('role')=='ray_worker'and before.get('command')!=after.get('command'),'Title transition must be Ray observation only')
            need(before['run']==run and before['slurm_job_id']==str(job_id)and before['uid']==uid and before['allocated_uuid_set']==sorted(uuids),'Title transition allocation differs')
            need(before['pid']not in held and before['pid']not in title_transitions,'Duplicate/held title transition')
            title_transitions[before['pid']]=after
        elif event=='owned_freeze_signal':
            need(active and not prelaunch and not validated and integer(row.get('pid'),True),'Freeze sequence')
            pid=row['pid'];certificate=row.get('certificate')
            if pid in title_transitions:need(certificate==title_transitions.pop(pid),'STOP does not match postpidfd title transition')
            need(isinstance(certificate,dict),'Full process certificate required')
            need(certificate.get('pid')==pid and integer(certificate.get('pid'),True)and integer(certificate.get('ppid'))and integer(certificate.get('uid'))and integer(certificate.get('start_ticks'),True),'Certificate PID/PPID/UID/epoch')
            need('parent_start_ticks'in certificate and (certificate['parent_start_ticks']is None or integer(certificate['parent_start_ticks'],True)),'Certificate parent epoch')
            if certificate['ppid']in certificates:
                need(certificate['parent_start_ticks']==certificates[certificate['ppid']]['start_ticks'],'Certificate held parent epoch mismatch')
            if certificate.get('role')=='phase_descendant':need(certificate['ppid']in certificates,'CPU descendant parent must already be certified')
            need(certificate.get('run')==run and certificate.get('slurm_job_id')==str(job_id)and certificate['uid']==uid,'Certificate allocation identity')
            need(certificate.get('state')in('R','S','D','T','t','I'),'Certificate live state')
            need(certificate.get('allocated_uuid_set')==sorted(uuids)and type(certificate.get('gpu_leases_inherited'))is bool,'Certificate UUID/provenance')
            inherited=certificate.get('inherited_gpu_lease_uuids');need(inherited==(sorted(uuids)if certificate['gpu_leases_inherited']else []),'Certificate inherited leases')
            proof=latest_proofs.get(str(pid))
            if proof is not None:need(certificate['start_ticks']==proof['start_ticks']and certificate['uid']==proof['uid']and certificate['gpu_leases_inherited'],'GPU proof/certificate mismatch')
            need(integer(row.get('start_ticks'),True)and integer(row.get('uid'))and row['start_ticks']==certificate['start_ticks']and row['uid']==uid and isinstance(row.get('lease_uuids'),list)and len(row['lease_uuids'])==8 and set(row['lease_uuids'])==uuids,'Freeze allocation mismatch')
            command=certificate.get('command');need(isinstance(command,list)and command and all(type(t)is str for t in command)and command[0],'Certificate command')
            need(row.get('role')==certificate.get('role')and row['role']in('phase','ray_worker','retriever','trainer','phase_descendant'),'Unknown signal role')
            if certificate['role']=='ray_worker'and certificate['command'][0].startswith('ray::')and anchor is None:anchor=w.load_runtime_anchor(run,verify_current=False)
            try:w.certificate_identity(certificate,anchor)
            except (ValueError,KeyError,TypeError)as exc:raise JournalError('Certificate immutable projection: '+str(exc))from exc
            role=row['role'];basenames={Path(t).name for t in command if t}
            need((role=='phase')==(pid==phase),'Wrong phase signal role')
            if role=='ray_worker':need(command[0].startswith('ray::')or Path(command[0]).name=='raylet','Ray command proof')
            if role=='retriever':need('retriever_local.py'in basenames,'Retriever command proof')
            if role=='trainer':need(w.is_trainer_command(command,run),'Trainer command proof')
            need(pid not in held,'Duplicate STOP');held[pid]=row['start_ticks'];certificates[pid]=certificate;stable=False;counts['freeze_events']+=1
        elif event=='owned_freeze_stable':
            need(active and not prelaunch and not validated and held and phase in held,'Stable freeze requires certified phase STOP')
            for pid,certificate in certificates.items():
                if certificate['ppid']in certificates:need(certificate['parent_start_ticks']==certificates[certificate['ppid']]['start_ticks'],'Stable certificate parent epoch mismatch')
                if certificate['role']!='phase_descendant':continue
                seen={pid};parent=certificate['ppid']
                while parent!=phase:
                    need(parent in certificates and parent not in seen,'CPU descendant lacks certified phase ancestry');seen.add(parent);parent=certificates[parent]['ppid']
            snapshots=[row.get(name)for name in ('before_certificates','after_certificates')]
            need(all(isinstance(snapshot,dict)for snapshot in snapshots),'Two raw stopped certificate snapshots required')
            for snapshot in snapshots:
                need(set(snapshot)=={str(pid)for pid in row.get('pids',[])},'Stopped snapshot PID set differs')
                for key,observed in snapshot.items():
                    need(isinstance(observed,dict)and observed.get('state')in ('T','t')and str(observed.get('pid'))==key,'Stopped snapshot state/PID invalid')
                    try:need(w.certificate_identity(observed,anchor)==w.certificate_identity(certificates[int(key)],anchor),'Stopped snapshot immutable identity differs')
                    except (ValueError,KeyError,TypeError)as exc:raise JournalError('Stopped snapshot projection: '+str(exc))from exc
            pids=row.get('pids');need(isinstance(pids,list)and len(pids)==len(set(pids))and phase in pids and set(pids)<=set(held),'Stable freeze PID set');need(row.get('epochs')=={str(p):held[p]for p in pids},'Stable freeze epoch mismatch');live_frozen=set(pids);stable=True
        elif event=='recovery_sample':
            need(active and (prelaunch and not held or stable)and not validated and not title_transitions,'Recovery sample without complete certified freeze/prelaunch')
            need(finite(row.get('elapsed_seconds'))and row['elapsed_seconds']<grace,'Late recovery observation')
            ok=sample(row);counts['recovery_samples']+=1;safe=safe+1 if ok else 0
        elif event=='recovery_validated':
            need(active and (prelaunch and not held or stable)and not validated and not title_transitions and safe>=3 and row.get('consecutive_safe_samples')==safe,'Recovery lacks consecutive safe3');validated=True
        elif event=='recovery_resume':
            need(active and not prelaunch and validated and not resuming and row.get('pause')is False,'Unsafe resume');resuming=True
        elif event=='owned_resume_signal':
            need(active and not prelaunch and validated and resuming and integer(row.get('signal'),True)and row['signal']==18,'CONT before validation')
            pid=row.get('pid');need(integer(pid,True)and integer(row.get('start_ticks'),True)and pid in held and pid not in resumed and row.get('start_ticks')==held[pid],'CONT foreign/reused/duplicate epoch');resumed.add(pid)
        elif event=='owned_resume_already_exited':
            need(active and not prelaunch and validated and resuming,'Exited resume before validation')
            pid=row.get('pid');need(integer(pid,True)and integer(row.get('start_ticks'),True)and pid in held and pid not in resumed and row.get('start_ticks')==held[pid] and pid not in live_frozen,'Exited original epoch must be absent from stable freeze');resumed.add(pid)
        elif event=='recovery_finished':
            need(active and validated and not title_transitions and ((prelaunch and not held and not resumed and not resuming)or(not prelaunch and resuming and resumed==set(held))),'Unfinished/missing CONT recovery')
            elapsed=row.get('elapsed_seconds');need(finite(elapsed)and elapsed>=0 and math.isclose(row.get('reserved_gpu_hours',-1),elapsed*8/3600,rel_tol=1e-9,abs_tol=1e-12),'Recovery cost mismatch')
            cost+=elapsed;counts['safe_resume_events']+=int(not prelaunch);counts['prelaunch_recoveries']+=int(prelaunch);active=False;held={};stable=False
        else:raise JournalError('Unsupported/unsafe/abort event '+repr(event))
    need(not active and not pending and not title_transitions,'Unfinished unsafe/recovery/title journal')
    need(counts['samples']>0,'Normal sample proof absent')
    need(path.read_bytes()==raw and sha(watcher_path)==WATCH_SHA,'Artifacts changed during replay')
    return dict(status='PASS_ACTUAL_GUARD_JOURNAL_REPLAY',scope='FRESH200_R3_RESOURCE_IDENTITY_REPLAY_NOT_TRAINING_ACCEPTANCE',job_id=job_id,source_commit=source_commit,run_root=run,raw_path=str(path),raw_sha256=hashlib.sha256(raw).hexdigest(),watch_sha256=WATCH_SHA,watcher_sha256=WATCH_SHA,helper_path=str(Path(__file__).resolve()),helper_sha256=sha(__file__),watcher_path=str(watcher_path),allocated_uuids=sorted(uuids),own_uid=uid,unfinished_recovery=False,cost_wall_seconds=cost,reserved_gpu_hours=cost*8/3600,**counts)
