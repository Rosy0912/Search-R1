"""Read actual V6 active-gate results during owned bursts; no model/QA proof."""
import hashlib,importlib.util,json,os,socket,sys,time,subprocess,signal
from pathlib import Path
ROOT=Path('/dataset/shaoshitong/search-r1-rl/topd')
HERE=Path(__file__).resolve().parent

def require(ok,msg):
    if not ok:raise RuntimeError(msg)

def load(name,path,sha):
    require(hashlib.sha256(path.read_bytes()).hexdigest()==sha,'Source drift: '+str(path))
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def wait(path,child):
    deadline=time.monotonic()+45
    while not path.exists():
        require(child.poll() is None,'Owner exited before receipt');require(time.monotonic()<deadline,'Receipt timeout');time.sleep(.05)
    return json.loads(path.read_text())

def trigger_once_after_snapshot(original,folder,phase):
    triggered=[False]
    def callback():
        snapshot=original()
        if not triggered[0]:
            (folder/f'trigger-{phase}.json').write_text(json.dumps(dict(actual_snapshot=snapshot,epoch=time.time()),indent=2)+'\n')
            for endpoint in (folder/'owner-0',folder/'owner-1'):(endpoint/f'command-{phase}').touch(exist_ok=False)
            triggered[0]=True
        return snapshot
    return callback

def coverage(trace,uuids,owners):
    high=any({g['uuid'] for g in c['raw_snapshot']['gpus'] if g['uuid'] in uuids}==uuids
             and all(g['sm']>15 for g in c['raw_snapshot']['gpus'] if g['uuid'] in uuids)
             for r in trace for c in r.get('captures',[]) if 'raw_snapshot' in c)
    require(high, 'FAIL_COVERAGE no actual all8 high sample')
    pending=None;accepted=[];count=0;dual_positive=[]
    for row in trace:
        if row.get('attempt')==1:pending=None
        if row.get('error')=='Process records span different completed sampling periods' and row.get('retry_authorized'):
            require(row.get('accepted') is False and row['foreign_sm_status']=='UNKNOWN_REJECTED_MIXED_PERIOD'
                    and row['retry_kind']=='FRESH_MIXED_PERIOD_MEASUREMENT','Invalid mixed rejection scope')
            caps=row['captures'];require(len(caps)>=2 and caps[-1]['capture_scope']=='POST_FAILURE_IDENTITY_CLOSE_NOT_C7_FINAL','No real postfailure identity closure')
            ids=(caps[0]['raw_snapshot']['identities'],caps[0]['uids'])
            require(all((c['raw_snapshot']['identities'],c['uids'])==ids for c in caps),'Mixed identities changed')
            windows=row['original_windows'][-1]['windows'];proof=[]
            for uid,w in windows.items():
                rows=w['samples'];pids={r['pid'] for r in rows if r['sm']>0}
                if w['status']=='SUCCESS' and owners<=pids and len({r['timestamp_us'] for r in rows})>1:
                    proof.append(dict(uuid=uid,actual_window=w))
            require(proof, 'FAIL_COVERAGE no real dual-own-positive mixed window')
            pending=ids;dual_positive+=proof;count+=1
        if pending is not None and row.get('accepted') and row.get('fresh_measurement_uuids'):
            require(all((c['raw_snapshot']['identities'],c['uids'])==pending for c in row['captures']),'Fresh mixed accepted identity changed')
            for uid in row['fresh_measurement_uuids']:
                ev=row['fresh_foreign_sm_evidence'][uid];bound=row['fresh_foreign_sm_upper_bounds'][uid]
                require(0<=bound<=15 and ev['upper_bound']==bound and ev['method'] in
                        ('DEVICE_TOTAL_AGGREGATE_UPPER_BOUND','FRESH_SUPPORTED_NONZERO_PROCESS_WINDOW'),'No fresh original full gate bound')
            accepted.append(row);pending=None
    require(count and accepted,'FAIL_COVERAGE no actual mixed rejection to fresh acceptance')
    high_supported=[]
    for row in accepted:
        final={g['uuid']:g for g in row['captures'][-1]['raw_snapshot']['gpus']}
        for uid,ev in row['fresh_foreign_sm_evidence'].items():
            if ev['method']!='FRESH_SUPPORTED_NONZERO_PROCESS_WINDOW':continue
            w=ev['window'];samples=w['samples']
            if (w['status']=='SUCCESS' and final[uid]['sm']>15 and owners<=set(ev['positive_own_pids'])
                    and owners<={r['pid'] for r in samples if r['sm']>0}
                    and len({r['timestamp_us'] for r in samples})==1):
                high_supported.append(dict(uuid=uid,actual_window=w,actual_gpu=final[uid]))
    require(high_supported,'FAIL_COVERAGE no fresh supported same-period dual-positive high-SM acceptance')
    return dict(typed_mixed_retries_observed=count,actual_all8_high_sm_observed=True,
                dual_positive_mixed_windows=dual_positive,fresh_dual_positive_high_supported=high_supported,accepted_fresh_mixed_measurements=accepted)

def main():
    require(socket.gethostname().split('.')[0]=='h100-11' and os.environ.get('AGENT_ROOT')==str(ROOT),'Own node ROOT required')
    require(HERE==ROOT/'tools/qwen25-32b-v8-dual-burst-probe','Wrong tool path')
    files=json.loads((HERE/'files.json').read_text())
    for name,sha in files.items():require(hashlib.sha256((HERE/name).read_bytes()).hexdigest()==sha,'Probe drift '+name)
    sys.path.insert(0,str(ROOT/'tools/qwen25-32b-terminal-fix-refreeze'))
    import eval64  # CPU utilities only, no model construction.
    c=load('explicit_burst_c7',ROOT/'tools/qwen25-32b-terminal-fix-refreeze/runtime_safety.py','c7b16f1dd5d7cc133e55f034025c7dc8181ddcafaf59bdccfda02ceaf51f081d')
    v6=load('explicit_burst_v6',ROOT/'tools/qwen25-32b-startup-guard-candidate-v8/startup_guard.py','9b382b777c3a22b497639eb7bee37c8ae1c1461dc978e2ea95765f480bec7b30')
    own=load('explicit_burst_cleanup',ROOT/'tools/topd-short-run.py','d4de70880869abbec64f81ccd86f6f8dd8755b01d9db2cb93e2ec112d3031a09')
    from search_r1.baselines.preflight import validate_inherited_gpu_leases
    validate_inherited_gpu_leases(expected_count=8)
    import torch
    require(not torch.cuda.is_initialized(),'Parent must remain CUDA-free')
    leases=json.loads(os.environ['SEARCH_R1_GPU_LEASES']);uuids={l['uuid'] for l in leases}
    head=subprocess.check_output(['git','-C',str(ROOT/'repo-terminal-fixed413'),'rev-parse','HEAD'],text=True).strip()
    require(head=='413a10c50f32febaafc83db8170e2b005e0e979f','Source HEAD drift')
    scope,relative=own.cgroup_scope(os.environ['SLURM_JOB_ID']);owned=own.OwnedProcesses(scope,relative)
    folder=ROOT/'logs'/('qwen25-32b-v8-dual-burst-'+os.environ['SLURM_JOB_ID']);folder.mkdir(exist_ok=False)
    baseline=c.nvml_snapshot();require(len({p['pid'] for p in baseline['applications'] if p['uuid'] in uuids})==8,'Expected protected8 baseline PIDs');result=dict(status='FAIL_TELEMETRY_ONLY_NO_MODEL_NO_TRAINING',baseline=baseline,files=files,source_commit=head,gpu_owner_process_count=2,leased_gpu_count=8,phases=[])
    child=None;children=[];guard=None
    def interrupted(signum,frame):raise RuntimeError('Own telemetry interrupted '+str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(l['uuid'] for l in leases))
        ready=[]
        for index in range(2):
            endpoint=folder/('owner-'+str(index));endpoint.mkdir()
            process=owned.spawn([sys.executable,str(HERE/'worker.py'),str(endpoint)],folder/('worker-'+str(index)+'.log'),env,pass_fds=tuple(int(l['fd']) for l in leases))
            children.append(process)
            receipt=wait(endpoint/'ready.json',process);require(receipt['pid']==process.pid,'Ready PID wrong')
            require({r['uuid'] for r in receipt['devices']}==uuids and all(r['allocated']==r['reserved']==0 for r in receipt['devices']),'Empty all8 contexts missing')
            ready.append(receipt)
        child=children[0]
        runtime=dict(baseline=baseline,retriever_identity=c.pid_identity(child.pid),cgroup=relative)
        result['ready']=ready;result['owner_identities']=[c.pid_identity(p.pid) for p in children]
        guard=v6.StartupGuard(c,expected_path=ROOT/'tools/qwen25-32b-terminal-fix-refreeze/runtime_safety.py',trace_path=folder/'guard.jsonl')
        guard.ready_check=lambda:True  # Identity retry closed before any burst.
        result['idle_gate']=guard.active_gate(runtime)
        for phase,duration in enumerate((.25,2.,16.),1):
            records=[]
            original_snapshot=guard.original_snapshot
            # Issue own burst only after the genuine initial snapshot; preserve its object.
            guard.original_snapshot=trigger_once_after_snapshot(original_snapshot,folder,phase)
            try:records.append(guard.active_gate(runtime))
            finally:guard.original_snapshot=original_snapshot
            for attempt in range(2):records.append(guard.active_gate(runtime))
            released=[wait(folder/('owner-'+str(index))/f'done-{phase}.json',process) for index,process in enumerate(children)]
            require(all(row['pid']==process.pid and not row['errors'] and all(r['allocated']==r['reserved']==0 for r in row['devices']) for row,process in zip(released,children)),'Both owners all8 release failed')
            result['phases'].append(dict(phase=phase,requested_seconds=duration,gates=records,released=released))
            (folder/'phase-progress.json').write_text(json.dumps(result,indent=2)+'\n')
            time.sleep(2)
        trace=[json.loads(line) for line in (folder/'guard.jsonl').read_text().splitlines()]
        result.update(coverage(trace,uuids,{p.pid for p in children}))
        for index,process in enumerate(children):
            (folder/('owner-'+str(index))/'exit').touch();require(process.wait(timeout=15)==0,'Owner failed exit')
        result['status']='PASS_TELEMETRY_ONLY_NO_MODEL_NO_TRAINING_NOT_TEACHER_READY'
    except BaseException as exc:result['error']=repr(exc)
    finally:
        try:owned.cleanup()
        except BaseException as exc:result['cleanup_error']=repr(exc);result['status']='FAIL_TELEMETRY_ONLY_NO_MODEL_NO_TRAINING'
        result['cleanup_record']=getattr(owned,'cleanup_record',None)
        final=c.nvml_snapshot();result['final']=final
        try:
            require(result['cleanup_record']['actual_cgroup_empty'],'Own cgroup not empty')
            require(all(p['pid'] not in {process.pid for process in children} for p in final['applications']),'Own GPU PID remains')
            old={(p['uuid'],p['pid']) for p in baseline['applications'] if p['uuid'] in uuids}
            actual={(p['uuid'],p['pid']) for p in final['applications'] if p['uuid'] in uuids}
            require(old==actual,'Foreign baseline GPU ownership changed')
            for g in final['gpus']:
                if g['uuid'] in uuids:
                    require(0<=g['sm']<=15,'Final foreign total SM unsafe')
                    memory=sum(p['used_mib'] for p in final['applications'] if p['uuid']==g['uuid'])
                    require(0<=memory<=.15*g['total_mib'],'Final foreign memory unsafe')
            for pid,identity in baseline['identities'].items():require(final['identities'].get(pid)==identity,'Foreign baseline identity changed')
            require(not torch.cuda.is_initialized(),'Parent initialized CUDA')
            result['baseline_preserved']=True;result['parent_cuda_initialized']=False
        except BaseException as exc:result['final_error']=repr(exc);result['status']='FAIL_TELEMETRY_ONLY_NO_MODEL_NO_TRAINING'
        (folder/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    require(result['status'].startswith('PASS_'),'Actual burst telemetry failed')
if __name__=='__main__':main()
