"""Actual8 leased GPU telemetry-only probe. No model/algorithm readiness signature."""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import socket
import sys
import subprocess
import time
ROOT=Path('/dataset/shaoshitong/search-r1-rl/topd')
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tools/qwen25-32b-eval64'))
import runtime_safety_candidate as c

def require(x,message):
    if not x:raise RuntimeError(message)

def validate_snapshot(current,baseline,own,uuids,scope):
    require({g['uuid'] for g in current['gpus'] if g['uuid'] in uuids}==uuids,'Allocated GPU measurements missing')
    require(all(math.isfinite(g[k]) for g in current['gpus'] if g['uuid'] in uuids for k in ('used_mib','total_mib','sm')), 'Nonfinite GPU measurements')
    require(all(g['total_mib']>0 and 0<=g['sm']<=100 and 0<=g['used_mib']<=g['total_mib'] for g in current['gpus'] if g['uuid'] in uuids), 'Invalid GPU measurements')
    require(all(math.isfinite(r['used_mib']) and r['used_mib']>=0 for r in current['applications']), 'Unknown process memory')
    old={(r['uuid'],r['pid']) for r in baseline['applications'] if r['uuid'] in uuids}
    rows={(r['uuid'],r['pid']) for r in current['applications'] if r['uuid'] in uuids}
    require(old<=rows,'Baseline PID/UUID disappeared')
    for uid,pid in rows:
        ident=current['identities'][str(pid)]
        if (uid,pid) in old:
            require(ident==baseline['identities'][str(pid)],'Baseline identity changed')
        else:
            require((uid,pid) in own,'Unknown GPU owner')
            require(any(v.split(':',2)[-1]==scope or v.split(':',2)[-1].startswith(scope+'/') for v in ident['cgroup']), 'Own PID left cgroup')
    for g in current['gpus']:
        if g['uuid'] in uuids:
            mem=sum(r['used_mib'] for r in current['applications'] if (r['uuid'],r['pid']) in old and r['uuid']==g['uuid'])
            require(mem/g['total_mib']<=.15,'Foreign memory exceeds15%')
    return old

def wait_files(folder,prefix,children,timeout=45):
    began=time.monotonic()
    while True:
        if all((folder/f'{prefix}-{i}.json').is_file() for i in range(8)):
            return [json.loads((folder/f'{prefix}-{i}.json').read_text()) for i in range(8)]
        require(all(p.poll() is None for p in children),'Worker exited before receipt')
        require(time.monotonic()-began<timeout,'Worker receipt timeout')
        time.sleep(.1)

def wait_active_snapshot(baseline,own,uuids,scope,children,attempts,verify_leases,
                         warmup_seconds=2.,timeout_seconds=12.):
    """Wait for actual rolling SM measurements, saving every attempt before checks.

    Never infer high SM from receipt existence, pmon, or a different sample.
    identity/foreign memory/8 own GPU owners/real lease descriptors stay checked.
    """
    began=time.monotonic()
    while True:
        require(time.monotonic()-began<timeout_seconds,'Bounded pulse warmup never reached all8 high-SM')
        require(all(p.poll() is None for p in children),'Pulse worker exited during warmup')
        verify_leases()
        snapshot=c.nvml_snapshot()
        elapsed=time.monotonic()-began
        attempts.append(dict(elapsed_seconds=elapsed,captured_epoch=time.time(),snapshot=snapshot))
        require(elapsed<=timeout_seconds,'Bounded pulse warmup snapshot exceeded deadline')
        foreign=validate_snapshot(snapshot,baseline,own,uuids,scope)
        actual={(r['uuid'],r['pid']) for r in snapshot['applications']}
        require(own<=actual,'Notall8 own GPU processes resident during warmup')
        ready=elapsed>=warmup_seconds and all(g['sm']>15 for g in snapshot['gpus'] if g['uuid'] in uuids)
        attempts[-1]['all8_high_sm_after_warmup']=ready
        if ready:return snapshot,foreign
        time.sleep(.5)


def main():
    require(socket.gethostname().split('.')[0]=='h100-11' and os.environ.get('AGENT_ROOT')==str(ROOT),'Wrong node/root')
    require(HERE==ROOT/'tools/gpu-sm-pulse-probe','Wrong tool directory')
    require(hashlib.sha256(Path(c.__file__).read_bytes()).hexdigest()=='c7b16f1dd5d7cc133e55f034025c7dc8181ddcafaf59bdccfda02ceaf51f081d', 'Candidate drift')
    require(hashlib.sha256((ROOT/'tools/topd-short-run.py').read_bytes()).hexdigest()=='d4de70880869abbec64f81ccd86f6f8dd8755b01d9db2cb93e2ec112d3031a09', 'Cleanup tool drift')
    import torch
    require(not torch.cuda.is_initialized(),'Parent must not initialize CUDA')
    from search_r1.baselines.preflight import validate_inherited_gpu_leases
    validate_inherited_gpu_leases(expected_count=8)  # actual original double15% before any CUDA
    leases=json.loads(os.environ['SEARCH_R1_GPU_LEASES']);uuids={r['uuid'] for r in leases}
    spec=importlib.util.spec_from_file_location('pulse_owned',ROOT/'tools/topd-short-run.py')
    launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)
    job=os.environ['SLURM_JOB_ID'];scope,relative=launcher.cgroup_scope(job)
    owned=launcher.OwnedProcesses(scope,relative)
    folder=ROOT/'logs'/f'gpu-sm-pulse-{job}';folder.mkdir(exist_ok=False)
    baseline=c.nvml_snapshot()
    require(len({r['pid'] for r in baseline['applications'] if r['uuid'] in uuids})==8,'Expected existing8 baseline PIDs')
    head=subprocess.check_output(['git','-C',str(ROOT/'repo'),'rev-parse','HEAD'],text=True).strip()
    require(head=='438af3529b40663aae7b8293080e09fda57f6a87','Source drift')
    result=dict(status='FAILED_TELEMETRY_ONLY_NO_TRAINING',scope='TINY_CUDA_SAMPLER_PROBE_NO_MODEL_NO_TRAINING',baseline=baseline,
                source_commit=head,file_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'worker.py',HERE/'probe.py',Path(c.__file__),ROOT/'tools/topd-short-run.py']})
    children=[];own=set()
    def interrupted(signum,frame):raise RuntimeError('Own probe interrupted '+str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        for rank,lease in enumerate(leases):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=lease['uuid'],SEARCH_R1_GPU_LEASES=json.dumps([lease]),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
            children.append(owned.spawn([sys.executable,str(HERE/'worker.py'),str(folder),str(rank)],folder/f'worker-{rank}.log',env,pass_fds=(int(lease['fd']),)))
            own.add((lease['uuid'],children[-1].pid))
        ready=wait_files(folder,'ready',children)
        require(all(not r['cuda_initialized'] for r in ready),'Workers must initially be CPUready')
        require(all(r['identity']['pid']==p.pid and r['uuid']==leases[i]['uuid'] for i,(r,p) in enumerate(zip(ready,children))), 'CPUready identity mismatch')
        result['ready']=ready
        (folder/'start').touch(exist_ok=False)
        result['busy']=wait_files(folder,'busy',children)
        require(all(r['identity']==ready[i]['identity'] and r['uuid']==leases[i]['uuid'] for i,r in enumerate(result['busy'])), 'Busy identity changed')
        result['warmup_attempts']=[]
        from search_r1.baselines.topd_local_gpu import lease_ownership
        def verify_leases():
            for record in leases:lease_ownership(record)
        active,foreign=wait_active_snapshot(baseline,own,uuids,relative,children,
                                            result['warmup_attempts'],verify_leases)
        result['active']=dict(snapshot=active)
        result['raw_active_pmon']=subprocess.check_output(['nvidia-smi','pmon','-s','u','-c','1'],text=True,timeout=15)
        windows,now=c.collect_process_windows(active,foreign)
        result['active']=dict(snapshot=active,windows=windows,query_completed_us=now)
        post_window=c.nvml_snapshot()
        validate_snapshot(post_window,baseline,own,uuids,relative)
        require(post_window['identities']==active['identities'], 'Process identities changed during NVML window')
        result['active']['post_window_snapshot']=post_window
        bounds,evidence=c.sm_aggregate_bounds(post_window,foreign,own,windows,int(time.time()*1000000))
        require(all(v<=15 for v in bounds.values()),'Foreign active SM exceeds15%')
        require(len(evidence)==8 and all(v['method']=='FRESH_SUPPORTED_NONZERO_PROCESS_WINDOW' for v in evidence.values()),'Need actual supported active evidence all8')
        result['active'].update(bounds=bounds,evidence=evidence)
        (folder/'stop').touch(exist_ok=False)
        released=wait_files(folder,'released',children)
        require(all(r['identity']==ready[i]['identity'] and r['uuid']==leases[i]['uuid'] for i,r in enumerate(released)), 'Release identity changed')
        require(all(r['status']=='PULSE_FINISHED_NO_TRAINING' and r['allocated_bytes']==r['reserved_bytes']==0 for r in released),'All8 allocator releases required')
        result['released']=released
        time.sleep(2)
        idle=c.nvml_snapshot();validate_snapshot(idle,baseline,own,uuids,relative)
        result['idle']=dict(snapshot=idle)
        idle_bounds,idle_evidence=c.sm_aggregate_bounds(idle,foreign,own,{},int(time.time()*1000000))
        require(all(v<=15 for v in idle_bounds.values()),'Idle foreign upper bound exceeds15%')
        result['idle']=dict(snapshot=idle,bounds=idle_bounds,evidence=idle_evidence)
        result['transition_foreign_upper_bounds']={u:max(bounds[u],idle_bounds[u]) for u in bounds}
        (folder/'exit').touch(exist_ok=False)
        for child in children:require(child.wait(timeout=15)==0,'Worker failed exit')
        result['status']='TELEMETRY_PROBE_PASS_NO_MODEL_NO_TRAINING_NOT_EVAL_READY'
    except BaseException as exc:result['error']=repr(exc)
    finally:
        try:owned.cleanup()
        except BaseException as exc:result.update(status='FAILED_TELEMETRY_ONLY_NO_TRAINING',cleanup_error=repr(exc))
        result['cleanup_record']=getattr(owned,'cleanup_record',None)
        final=c.nvml_snapshot();result['final']=final
        try:
            validate_snapshot(final,baseline,set(),uuids,relative)
            require(not own&{(r['uuid'],r['pid']) for r in final['applications']},'Own GPU PID remains')
            require(result['cleanup_record']['actual_cgroup_empty'],'Own descendants remain')
            final_foreign={(r['uuid'],r['pid']) for r in baseline['applications'] if r['uuid'] in uuids}
            result['final_foreign_bounds'],result['final_foreign_evidence']=c.sm_aggregate_bounds(final,final_foreign,set(),{},int(time.time()*1000000))
            require(all(v<=15 for v in result['final_foreign_bounds'].values()),'Final foreign SM exceeds15%')
            require(not torch.cuda.is_initialized(),'Parent touched CUDA')
            result['parent_cuda_initialized']=False
            result['baseline_preserved']=True
        except BaseException as exc:result.update(status='FAILED_TELEMETRY_ONLY_NO_TRAINING',final_error=repr(exc))
        (folder/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
    require(result['status']=='TELEMETRY_PROBE_PASS_NO_MODEL_NO_TRAINING_NOT_EVAL_READY','Actual telemetry proof failed')

if __name__=='__main__':main()
