"""Read actual V6 active-gate results during owned bursts; no model/QA proof."""
import hashlib,importlib.util,json,os,socket,sys,time,subprocess,signal
from pathlib import Path
ROOT=Path('/dataset/shaoshitong/search-r1-rl/topd')
HERE=Path(__file__).resolve().parent

def require(ok,msg):
    if not ok:raise RuntimeError(msg)

def load(name,path,sha):
    require(hashlib.sha256(path.read_bytes()).hexdigest()==sha,'Source drift: '+str(path))
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def verified_eval64(path,expected_sha='24d144fdac16b0983a11e3bd99d111f44d0bc312dc98b74847f3d16524076bee'):
    path=Path(path).resolve()
    require(hashlib.sha256(path.read_bytes()).hexdigest()==expected_sha,'Frozen eval64 utility source drift')
    cached=sys.modules.get('eval64')
    if cached is not None:
        require(Path(cached.__file__).resolve()==path and hashlib.sha256(Path(cached.__file__).read_bytes()).hexdigest()==expected_sha,'Cached foreign eval64 utility module')
    module=load('eval64',path,expected_sha)
    require(Path(module.__file__).resolve()==path,'Loaded foreign eval64 utility module')
    return module

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
    require(high,'FAIL_COVERAGE no actual all8 high sample')
    proofs=[];phase=0
    for row in trace:
        if row.get('event')=='PHASE_STARTED':phase=row['phase']
        if not row.get('accepted'):continue
        caps=row['captures'];first=caps[0]
        require(all(c['raw_snapshot']['identities']==first['raw_snapshot']['identities'] and c['uids']==first['uids'] and not c['uid_errors'] for c in caps),'Accepted identity/UID drift')
        final={g['uuid']:g for g in caps[-1]['raw_snapshot']['gpus']}
        evs=row.get('observed_foreign_sm_evidence') or {};bounds=row.get('observed_foreign_sm_upper_bounds') or {}
        for uid,ev in evs.items():
            if uid not in uuids or final[uid]['sm']<=15 or ev['method']!='FRESH_SUPPORTED_NONZERO_PROCESS_WINDOW':continue
            w=ev['window'];samples=w['samples'];require(w['status']=='SUCCESS' and samples,'Unsupported returned window')
            now=ev['evaluation_now_us'];query=ev['query_completed_us'];times=[r['timestamp_us'] for r in samples]
            require(1000000<=w['end_us']-w['start_us']<=3000000 and 0<=now-w['end_us']<=2000000 and w['end_us']<=query<=now and all(w['start_us']<t<=w['end_us'] and 0<=now-t<=1500000 for t in times),'Fresh actual window evidence required')
            pids={r['pid'] for r in samples if r['sm']>0}
            if not owners<=pids or len(set(times))<=1:continue
            maxima={}
            known={app['pid'] for app in first['raw_snapshot']['applications'] if app['uuid']==uid}
            for r in samples:
                require(r['pid'] in known and 0<=r['sm']<=100,'Unknown/invalid returned PID')
                if r['pid'] not in owners:maxima[r['pid']]=max(maxima.get(r['pid'],0),r['sm'])
            measured={int(k):v for k,v in ev['foreign_observed_maxima'].items()}
            require(measured==maxima and sum(maxima.values())<=15,'Observed foreign maxima breach or mismatch')
            require(ev['timestamp_min_us']==min(times) and ev['timestamp_max_us']==max(times) and ev['timestamp_span_us']==max(times)-min(times),'Timestamp provenance mismatch')
            require(ev['aggregate_scope']=='SUM_OF_PER_FOREIGN_PID_MAXIMA_IN_RETURNED_RECORDS_NOT_CONTINUOUS_BOUND' and ev['absent_interpretation']=='NONZERO_FILTER_WINDOW_INFERENCE_NOT_MEASURED','Incorrect window inference scope')
            require(ev['upper_bound']==bounds[uid] and 0<=bounds[uid]<=15 and bounds[uid]>=sum(maxima.values()),'Final aggregate mismatch')
            require(owners<=set(ev['positive_own_pids']),'Both actual owners not positive')
            proofs.append(dict(phase=phase,uuid=uid,actual_window=w,evidence=ev,actual_gpu=final[uid]))
    require(any(p['phase']==3 for p in proofs),'FAIL_COVERAGE no actual sustained-phase high dual-positive unequal-timestamp max-sum acceptance')
    return dict(actual_all8_high_sm_observed=True,actual_high_unequal_dual_positive_windows=proofs,
                scope='OBSERVED_RETURNED_PROCESS_RECORD_MAXIMA_NOT_CONTINUOUS_SM_GUARANTEE')

def main():
    require(socket.gethostname().split('.')[0]=='h100-11' and os.environ.get('AGENT_ROOT')==str(ROOT),'Own node ROOT required')
    require(HERE==ROOT/'tools/qwen25-32b-window-dual-burst-probe-v1','Wrong tool path')
    files=json.loads((HERE/'files.json').read_text())
    for name,sha in files.items():require(hashlib.sha256((HERE/name).read_bytes()).hexdigest()==sha,'Probe drift '+name)
    sys.path.insert(0,str(ROOT/'tools/qwen25-32b-terminal-fix-refreeze'))
    verified_eval64(ROOT/'tools/qwen25-32b-terminal-fix-refreeze/eval64.py')  # Explicit frozen CPU utility only.
    candidate_dir=ROOT/'tools/qwen25-32b-window-safety-candidate-v1'
    require(hashlib.sha256((candidate_dir/'files.json').read_bytes()).hexdigest()=='9ba791e4cca79db6d7ebbea92836a21f406acd0b83d0b47b8dd2ae08ec8f64b2','Window package manifest drift')
    for name,digest in json.loads((candidate_dir/'files.json').read_text()).items():require(hashlib.sha256((candidate_dir/name).read_bytes()).hexdigest()==digest,'Window package source drift '+name)
    c=load('explicit_burst_window',ROOT/'tools/qwen25-32b-window-safety-candidate-v1/runtime_safety.py','3316d920f76256c5b45c5bf5e3c16c5c89a26a95cdaf821eb12ab612dc7a7384')
    v6=load('explicit_burst_v6',ROOT/'tools/qwen25-32b-window-safety-candidate-v1/startup_guard.py','a9845f16e26e226d367541f617ba2ac633fb3c8e955fc0818ff49cc6324b6e2e')
    own=load('explicit_burst_cleanup',ROOT/'tools/topd-short-run.py','d4de70880869abbec64f81ccd86f6f8dd8755b01d9db2cb93e2ec112d3031a09')
    from search_r1.baselines.preflight import validate_inherited_gpu_leases
    validate_inherited_gpu_leases(expected_count=8)
    import torch
    require(not torch.cuda.is_initialized(),'Parent must remain CUDA-free')
    leases=json.loads(os.environ['SEARCH_R1_GPU_LEASES']);uuids={l['uuid'] for l in leases}
    head=subprocess.check_output(['git','-C',str(ROOT/'repo-terminal-fixed413'),'rev-parse','HEAD'],text=True).strip()
    require(head=='413a10c50f32febaafc83db8170e2b005e0e979f','Source HEAD drift')
    scope,relative=own.cgroup_scope(os.environ['SLURM_JOB_ID']);owned=own.OwnedProcesses(scope,relative)
    folder=ROOT/'logs'/('qwen25-32b-window-dual-burst-'+os.environ['SLURM_JOB_ID']);folder.mkdir(exist_ok=False)
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
        guard=v6.StartupGuard(c,expected_path=ROOT/'tools/qwen25-32b-window-safety-candidate-v1/runtime_safety.py',trace_path=folder/'guard.jsonl')
        guard.ready_check=lambda:True  # Identity retry closed before any burst.
        result['idle_gate']=guard.active_gate(runtime)
        for phase,duration in enumerate((.25,2.,16.),1):
            guard._write(dict(event='PHASE_STARTED',phase=phase,requested_seconds=duration,accepted=False))
            records=[]
            original_snapshot=guard.original_snapshot
            # Issue own burst only after the genuine initial snapshot; preserve its object.
            guard.original_snapshot=trigger_once_after_snapshot(original_snapshot,folder,phase)
            try:records.append(guard.active_gate(runtime))
            finally:guard.original_snapshot=original_snapshot
            for attempt in range(2):records.append(guard.active_gate(runtime))
            phase_deadline=time.monotonic()+45
            while not all((folder/('owner-'+str(index))/f'done-{phase}.json').is_file() for index in range(2)):
                require(time.monotonic()<phase_deadline,'Sustained own burst did not finish')
                require(all(process.poll() is None for process in children),'Own process failed during sustained burst')
                records.append(guard.active_gate(runtime))
            released=[wait(folder/('owner-'+str(index))/f'done-{phase}.json',process) for index,process in enumerate(children)]
            require(all(row['pid']==process.pid and row['phase']==phase and not row['errors'] and len(row['devices'])==8 and {r['uuid'] for r in row['devices']}==uuids and all(r['allocated']==r['reserved']==0 for r in row['devices']) for row,process in zip(released,children)),'Both owners all8 release failed')
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
