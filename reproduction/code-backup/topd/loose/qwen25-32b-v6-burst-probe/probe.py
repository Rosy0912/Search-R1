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
            (folder/f'command-{phase}').touch(exist_ok=False);triggered[0]=True
        return snapshot
    return callback

def coverage(trace,uuids):
    high=any(all(g['sm']>15 for g in cap['raw_snapshot']['gpus'] if g['uuid'] in uuids)
             and {g['uuid'] for g in cap['raw_snapshot']['gpus'] if g['uuid'] in uuids}==uuids
             for row in trace for cap in row.get('captures',[]))
    require(high, 'FAIL_COVERAGE no actual all8 high-SM sample')
    pending=None;accepted=[];count=0
    for row in trace:
        if row.get('attempt')==1:pending=None
        if row.get('error')=='Device SM increased after upper-bound sample' and row.get('retry_authorized'):
            require(row.get('accepted') is False and row.get('foreign_sm_status')=='UNKNOWN_REJECTED_TRANSITION'
                    and row.get('retry_kind')=='FRESH_LOW_TO_HIGH_MEASUREMENT', 'Invalid rejected measurement provenance')
            caps=row['captures'];require(len(caps)>=2,'Missing rejected raw pair')
            first={g['uuid']:g['sm'] for g in caps[0]['raw_snapshot']['gpus']}
            require(any(first[g['uuid']]<=15 and g['sm']>15 for g in caps[-1]['raw_snapshot']['gpus'] if g['uuid'] in uuids), 'Rejected pair has no actual low-to-high transition')
            identity=(caps[0]['raw_snapshot']['identities'],caps[0]['uids'])
            require(all((c['raw_snapshot']['identities'],c['uids'])==identity for c in caps),'Rejected transition identity changed')
            pending=identity;count+=1
        if row.get('accepted') and row.get('fresh_measurement_uuids'):
            require(pending is not None,'Accepted measurement without exact typed rejection')
            require(all((c['raw_snapshot']['identities'],c['uids'])==pending for c in row['captures']),'Fresh accepted identities changed')
            for uid in row['fresh_measurement_uuids']:
                bound=row['fresh_foreign_sm_upper_bounds'][uid];ev=row['fresh_foreign_sm_evidence'][uid]
                require(0<=bound<=15 and ev['upper_bound']==bound and ev['method'] in
                        ('DEVICE_TOTAL_AGGREGATE_UPPER_BOUND','FRESH_SUPPORTED_NONZERO_PROCESS_WINDOW'),'Missing original fresh bound')
            accepted.append(row);pending=None
    require(count and accepted,'FAIL_COVERAGE no actual rejected-to-fresh-accepted measurement')
    return dict(typed_transition_retries_observed=count,actual_all8_high_sm_observed=True,
                accepted_fresh_transition_measurements=accepted)

def main():
    require(socket.gethostname().split('.')[0]=='h100-11' and os.environ.get('AGENT_ROOT')==str(ROOT),'Own node ROOT required')
    require(HERE==ROOT/'tools/qwen25-32b-v6-burst-probe','Wrong tool path')
    files=json.loads((HERE/'files.json').read_text())
    for name,sha in files.items():require(hashlib.sha256((HERE/name).read_bytes()).hexdigest()==sha,'Probe drift '+name)
    sys.path.insert(0,str(ROOT/'tools/qwen25-32b-terminal-fix-refreeze'))
    import eval64  # CPU utilities only, no model construction.
    c=load('explicit_burst_c7',ROOT/'tools/qwen25-32b-terminal-fix-refreeze/runtime_safety.py','c7b16f1dd5d7cc133e55f034025c7dc8181ddcafaf59bdccfda02ceaf51f081d')
    v6=load('explicit_burst_v6',ROOT/'tools/qwen25-32b-startup-guard-candidate-v7/startup_guard.py','12a2990031e29724b804907e55714eac3bb3896ed31c268a1c4ffeb1656516d0')
    own=load('explicit_burst_cleanup',ROOT/'tools/topd-short-run.py','d4de70880869abbec64f81ccd86f6f8dd8755b01d9db2cb93e2ec112d3031a09')
    from search_r1.baselines.preflight import validate_inherited_gpu_leases
    validate_inherited_gpu_leases(expected_count=8)
    import torch
    require(not torch.cuda.is_initialized(),'Parent must remain CUDA-free')
    leases=json.loads(os.environ['SEARCH_R1_GPU_LEASES']);uuids={l['uuid'] for l in leases}
    head=subprocess.check_output(['git','-C',str(ROOT/'repo-terminal-fixed413'),'rev-parse','HEAD'],text=True).strip()
    require(head=='413a10c50f32febaafc83db8170e2b005e0e979f','Source HEAD drift')
    scope,relative=own.cgroup_scope(os.environ['SLURM_JOB_ID']);owned=own.OwnedProcesses(scope,relative)
    folder=ROOT/'logs'/('qwen25-32b-v6-burst-'+os.environ['SLURM_JOB_ID']);folder.mkdir(exist_ok=False)
    baseline=c.nvml_snapshot();require(len({p['pid'] for p in baseline['applications'] if p['uuid'] in uuids})==8,'Expected protected8 baseline PIDs');result=dict(status='FAIL_TELEMETRY_ONLY_NO_MODEL_NO_TRAINING',baseline=baseline,files=files,source_commit=head,phases=[])
    child=None;guard=None
    def interrupted(signum,frame):raise RuntimeError('Own telemetry interrupted '+str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(l['uuid'] for l in leases))
        child=owned.spawn([sys.executable,str(HERE/'worker.py'),str(folder)],folder/'worker.log',env,pass_fds=tuple(int(l['fd']) for l in leases))
        ready=wait(folder/'ready.json',child);require(ready['pid']==child.pid,'Ready PID wrong')
        require({r['uuid'] for r in ready['devices']}==uuids and all(r['allocated']==r['reserved']==0 for r in ready['devices']),'Empty all8 contexts missing')
        runtime=dict(baseline=baseline,retriever_identity=c.pid_identity(child.pid),cgroup=relative)
        result['ready']=ready;result['owner_identity']=runtime['retriever_identity']
        guard=v6.StartupGuard(c,expected_path=ROOT/'tools/qwen25-32b-terminal-fix-refreeze/runtime_safety.py',trace_path=folder/'guard.jsonl')
        guard.ready_check=lambda:True  # Identity retry closed before any burst.
        result['idle_gate']=guard.active_gate(runtime)
        for phase,duration in enumerate((.25,2.,6.),1):
            records=[]
            original_snapshot=guard.original_snapshot
            # Issue own burst only after the genuine initial snapshot; preserve its object.
            guard.original_snapshot=trigger_once_after_snapshot(original_snapshot,folder,phase)
            try:records.append(guard.active_gate(runtime))
            finally:guard.original_snapshot=original_snapshot
            for attempt in range(2):records.append(guard.active_gate(runtime))
            released=wait(folder/f'done-{phase}.json',child)
            require(released['pid']==child.pid and not released['errors'] and all(r['allocated']==r['reserved']==0 for r in released['devices']),'Burst all8 release failed')
            result['phases'].append(dict(phase=phase,requested_seconds=duration,gates=records,released=released))
            (folder/'phase-progress.json').write_text(json.dumps(result,indent=2)+'\n')
            time.sleep(2)
        trace=[json.loads(line) for line in (folder/'guard.jsonl').read_text().splitlines()]
        result.update(coverage(trace,uuids))
        (folder/'exit').touch();require(child.wait(timeout=15)==0,'Owner failed exit')
        result['status']='PASS_TELEMETRY_ONLY_NO_MODEL_NO_TRAINING_NOT_TEACHER_READY'
    except BaseException as exc:result['error']=repr(exc)
    finally:
        try:owned.cleanup()
        except BaseException as exc:result['cleanup_error']=repr(exc);result['status']='FAIL_TELEMETRY_ONLY_NO_MODEL_NO_TRAINING'
        result['cleanup_record']=getattr(owned,'cleanup_record',None)
        final=c.nvml_snapshot();result['final']=final
        try:
            require(result['cleanup_record']['actual_cgroup_empty'],'Own cgroup not empty')
            require(child is None or all(p['pid']!=child.pid for p in final['applications']),'Owner GPU PID remains')
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
