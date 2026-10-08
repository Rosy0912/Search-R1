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

def zero_tail_coverage(trace, uuids, owners):
    import math
    proofs=[]
    for index,row in enumerate(trace):
        if row.get('error')!='No positive known-own sample proves active process sampler':continue
        if not (row.get('accepted') is False and row.get('retry_authorized') is True
                and row.get('retry_kind')=='FRESH_SUPPORTED_ZERO_OWN_MEASUREMENT'
                and row.get('foreign_sm_status')=='UNKNOWN_REJECTED_TRANSITION'):continue
        caps=row['captures'];require(len(caps)==2 and caps[-1]['snapshot_role']=='POST_FAILURE_IDENTITY_CLOSE_NOT_ORIGINAL_FINAL','Actual postfailure closure missing')
        first=caps[0];ident=(first['raw_snapshot']['identities'],first['uids'])
        require(all((c['raw_snapshot']['identities'],c['uids'])==ident and not c['uid_errors'] for c in caps),'Rejected closure identity differs')
        rec=row['original_windows'];require(len(rec)==1,'One actual rejected collector required');rec=rec[0]
        query=rec['query_completed_us']
        require(rec['snapshot']==first['raw_snapshot'] and query<=rec['epoch']*1e6<=caps[-1]['epoch']*1e6<=row['epoch']*1e6 and caps[-1]['epoch']*1e6-query<=30000000,'Rejected actual clock provenance differs')
        foreign={tuple(v) for v in rec['foreign']};zero=[]
        for gpu in first['raw_snapshot']['gpus']:
            uid=gpu['uuid']
            if uid not in uuids or gpu['sm']<=15:continue
            w=rec['windows'][uid];require(w['status']=='SUCCESS' and w['samples'],'Rejected window unsupported')
            require(w['end_us']<=query and 0<=query-w['end_us']<=2000000 and 1000000<=w['end_us']-w['start_us']<=3000000,'Rejected window invalid')
            known={a['pid'] for a in first['raw_snapshot']['applications'] if a['uuid']==uid}
            maxima={};actualown=[]
            for sample in w['samples']:
                require(sample['pid'] in known and math.isfinite(sample['sm']) and 0<=sample['sm']<=100 and w['start_us']<sample['timestamp_us']<=w['end_us'] and 0<=query-sample['timestamp_us']<=1500000,'Rejected row invalid')
                if (uid,sample['pid']) in foreign:maxima[sample['pid']]=max(maxima.get(sample['pid'],0),sample['sm'])
                elif sample['pid'] in owners:actualown.append(sample)
            require(sum(maxima.values())<=15 and actualown,'Rejected foreign breach or absent actual own row')
            if all(v['sm']==0 for v in actualown):zero.append(uid)
        require(zero,'No actual supported zero-own high card')
        for accepted in trace[index+1:]:
            if accepted.get('event')=='PHASE_STARTED':break
            if not accepted.get('accepted'):continue
            require(accepted['attempt']>row['attempt'] and accepted['epoch']>=row['epoch'],'Fresh accepted call ordering differs')
            require(all((c['raw_snapshot']['identities'],c['uids'])==ident and not c['uid_errors'] for c in accepted['captures']),'Fresh identity changed')
            require(set(accepted['fresh_measurement_uuids'])>=set(zero),'Fresh complete measurement target missing')
            require(all(0<=v<=15 for v in accepted['observed_foreign_sm_upper_bounds'].values()),'Fresh bound invalid')
            require(accepted.get('validator')=='PINNED_MULTI_PID_WINDOW_FULL_SM_GATE' and len(accepted['captures'])==2,'Not full actual fresh gate')
            for uid in accepted['fresh_measurement_uuids']:
                ev=accepted['observed_foreign_sm_evidence'][uid]
                require(ev['upper_bound']==accepted['observed_foreign_sm_upper_bounds'][uid],'Fresh evidence mismatch')
                def branch(evidence, capture):
                    actual_sm=next(g['sm'] for g in capture['raw_snapshot']['gpus'] if g['uuid']==uid)
                    if evidence['method']=='DEVICE_TOTAL_AGGREGATE_UPPER_BOUND':
                        require(actual_sm<=15 and evidence['upper_bound']==actual_sm,'Fresh low aggregate is not actually low')
                        return actual_sm
                    require(actual_sm>15 and evidence['method']=='FRESH_SUPPORTED_NONZERO_PROCESS_WINDOW','Fresh unsupported evidence')
                    w=evidence['window'];now=evidence['evaluation_now_us'];query=evidence['query_completed_us']
                    require(w['status']=='SUCCESS' and w['samples'] and w['end_us']<=query<=now and 0<=now-w['end_us']<=2000000 and 1000000<=w['end_us']-w['start_us']<=3000000,'Fresh high window invalid')
                    known={a['pid'] for a in capture['raw_snapshot']['applications'] if a['uuid']==uid}
                    foreignmax={};positive=False
                    for sample in w['samples']:
                        require(sample['pid'] in known and math.isfinite(sample['sm']) and 0<=sample['sm']<=100 and w['start_us']<sample['timestamp_us']<=w['end_us'] and 0<=now-sample['timestamp_us']<=1500000,'Fresh sample invalid')
                        if (uid,sample['pid']) in foreign:foreignmax[sample['pid']]=max(foreignmax.get(sample['pid'],0),sample['sm'])
                        elif sample['pid'] in owners and sample['sm']>0:positive=True
                    require(positive and sum(foreignmax.values())<=15 and {int(k):v for k,v in evidence['foreign_observed_maxima'].items()}==foreignmax,'Fresh own activity or foreign maxima invalid')
                    return sum(foreignmax.values())
                initial_bound=branch(ev['initial_evidence'],accepted['captures'][0])
                final_ev=dict(ev,upper_bound=ev['final_foreign_upper_bound'])
                final_bound=branch(final_ev,accepted['captures'][-1])
                require(ev['initial_foreign_upper_bound']==initial_bound and ev['final_foreign_upper_bound']==final_bound and ev['upper_bound']==max(initial_bound,final_bound),'Fresh initial/final maximum differs')
            proofs.append(dict(rejected=row,fresh_accepted=accepted,actual_zero_own_high_uuids=zero));break
    require(proofs,'FAIL_COVERAGE no actual rejected zero-own then fresh full-gate acceptance')
    return proofs

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
    require(any(p['phase']==1 for p in proofs),'FAIL_COVERAGE no actual sustained-phase high dual-positive unequal-timestamp max-sum acceptance')
    return dict(actual_zero_own_remeasurements=zero_tail_coverage(trace,uuids,owners),actual_all8_high_sm_observed=True,actual_high_unequal_dual_positive_windows=proofs,
                scope='OBSERVED_RETURNED_PROCESS_RECORD_MAXIMA_NOT_CONTINUOUS_SM_GUARANTEE')

def main():
    require(socket.gethostname().split('.')[0]=='h100-11' and os.environ.get('AGENT_ROOT')==str(ROOT),'Own node ROOT required')
    require(HERE==ROOT/'tools/qwen25-32b-window-dual-burst-probe-v2','Wrong tool path')
    files=json.loads((HERE/'files.json').read_text())
    for name,sha in files.items():require(hashlib.sha256((HERE/name).read_bytes()).hexdigest()==sha,'Probe drift '+name)
    sys.path.insert(0,str(ROOT/'tools/qwen25-32b-terminal-fix-refreeze'))
    verified_eval64(ROOT/'tools/qwen25-32b-terminal-fix-refreeze/eval64.py')  # Explicit frozen CPU utility only.
    candidate_dir=ROOT/'tools/qwen25-32b-window-safety-candidate-v2'
    require(hashlib.sha256((candidate_dir/'files.json').read_bytes()).hexdigest()=='5c921921dafbb95144243f8a90310d17ea719d3557daba0d16abe83e45e6d589','Window package manifest drift')
    for name,digest in json.loads((candidate_dir/'files.json').read_text()).items():require(hashlib.sha256((candidate_dir/name).read_bytes()).hexdigest()==digest,'Window package source drift '+name)
    c=load('explicit_burst_window',ROOT/'tools/qwen25-32b-window-safety-candidate-v2/runtime_safety.py','3316d920f76256c5b45c5bf5e3c16c5c89a26a95cdaf821eb12ab612dc7a7384')
    v6=load('explicit_burst_v6',ROOT/'tools/qwen25-32b-window-safety-candidate-v2/startup_guard.py','6afa6825dbf22f6dcc7a0ed47aeea341d5f043be7a2dfd26de5fcc4273bc70ef')
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
        guard=v6.StartupGuard(c,expected_path=ROOT/'tools/qwen25-32b-window-safety-candidate-v2/runtime_safety.py',trace_path=folder/'guard.jsonl')
        guard.ready_check=lambda:True  # Identity retry closed before any burst.
        result['idle_gate']=guard.active_gate(runtime)
        for phase,duration in enumerate((16.,.25,2.),1):
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
            require(all(row['pid']==process.pid and row['phase']==phase and row['duration']==duration and not row['errors'] and len(row['devices'])==8 and {r['uuid'] for r in row['devices']}==uuids and all(r['allocated']==r['reserved']==0 for r in row['devices']) for row,process in zip(released,children)),'Both owners all8 release failed')
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
