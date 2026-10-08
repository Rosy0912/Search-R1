"""Candidate single-own-child scoring supervisor. Never submits jobs."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
from score_gate import ROOT, require, digest, check_identity, write_json
from caller_contracts import same_cpu_identity, context_receipt, verify_bundle

LAUNCHER_SHA='d4de70880869abbec64f81ccd86f6f8dd8755b01d9db2cb93e2ec112d3031a09'
SAFETY_SHA='c7b16f1dd5d7cc133e55f034025c7dc8181ddcafaf59bdccfda02ceaf51f081d'


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def slurm():
    require(socket.gethostname()=='h100-11' and os.environ.get('SLURM_JOB_NODELIST')=='h100-11', 'Ownh11 Slurm only')
    require(os.environ.get('AGENT_ROOT')==str(ROOT), 'OwnROOT only')
    require(int(os.environ.get('SLURM_GPUS_ON_NODE','0'))==8 and len(os.environ.get('SLURM_JOB_GPUS','').split(','))==8, 'Actual8GPU allocation required')
    require(int(os.environ.get('SLURM_CPUS_PER_TASK','0'))>=128 and int(os.environ.get('SLURM_MEM_PER_NODE','0'))>=384*1024, '128CPU384GiB required')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--provenance',required=True)
    parser.add_argument('--candidate',required=True)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    bundle=verify_bundle(Path(__file__).parent)
    provenance=json.loads(Path(args.provenance).read_text())
    check_identity(provenance,args.candidate)
    if args.check_only:
        print('CPU_SELECTEDTEACHER_SOURCE_CHECK_ONLY_NOT_GPU_ACCEPTANCE');return
    slurm()
    from search_r1.baselines.preflight import validate_inherited_gpu_leases
    validate_inherited_gpu_leases(8)
    require(digest(ROOT/'tools/topd-short-run.py')==LAUNCHER_SHA, 'Owned-process launcher drift')
    safety_dir=ROOT/'tools/qwen25-32b-terminal-fix-refreeze'
    require(digest(safety_dir/'runtime_safety.py')==SAFETY_SHA, 'Active foreign accounting drift')
    sys.path.insert(0,str(safety_dir))
    from runtime_safety import active_gate,nvml_snapshot,pid_identity
    launcher=load(ROOT/'tools/topd-short-run.py','score_owned_launcher')
    scope,relative=launcher.cgroup_scope(os.environ['SLURM_JOB_ID'])
    owned=launcher.OwnedProcesses(scope,relative)
    out=ROOT/'logs'/('topd32b-score-gate-'+os.environ['SLURM_JOB_ID']);out.mkdir(exist_ok=False)
    leases=json.loads(os.environ['SEARCH_R1_GPU_LEASES']);fds=tuple(x['fd'] for x in leases)
    baseline=nvml_snapshot();write_json(out/'baseline-nvml.json',baseline)
    result=dict(status='RUNNING_NO_TRAINING',scope='32B_NATIVE_SCORE_ONLY_UNREVIEWED',baseline=baseline,
                pending=['LOCAL_GPU_RETRIEVER_COEXISTENCE','ACTOR_FSDP_COEXISTENCE','ROLLOUT_VLLM_COEXISTENCE'],
                provenance_sha256=digest(args.provenance),job=os.environ['SLURM_JOB_ID'],cgroup=relative)
    result['caller_bundle']=bundle
    env=dict(os.environ)
    for k in list(env):
        if k.lower() in ('http_proxy','https_proxy','all_proxy','no_proxy'):env.pop(k)
    env.update(NO_PROXY='127.0.0.1,localhost,::1',no_proxy='127.0.0.1,localhost,::1',
               PYTHONPATH=str(ROOT/'repo-terminal-fixed413'),CUDA_VISIBLE_DEVICES=','.join(x['uuid'] for x in leases),
               HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1',TORCH_ALLOW_TF32_CUBLAS_OVERRIDE='0',NVIDIA_TF32_OVERRIDE='0',
               TOPD32B_SCORER_PARENT=str(os.getpid()),TOPD32B_SCORER_CGROUP=relative)
    stop=threading.Event();monitor_thread=None;errors=[];runtime=None;child=None
    def sample():
        raw=launcher.gpu_sample();raw['epoch']=time.time();raw['own_processes']=[launcher.process_sample(p) for p in sorted(launcher.cgroup_pids(scope)-owned.baseline)]
        with (out/'gpu-memory-timeseries.jsonl').open('a') as f:f.write(json.dumps(raw)+'\n')
    def monitor():
        while not stop.wait(1):
            try:
                sample()
                if runtime is not None:
                    evidence=active_gate(runtime)
                    with (out/'foreign-accounting.jsonl').open('a') as f:f.write(json.dumps(dict(epoch=time.time(),**evidence))+'\n')
            except BaseException as exc:errors.append(repr(exc))
    def wait_file(name,timeout=1800):
        deadline=time.monotonic()+timeout
        while not (out/name).exists():
            require(child.poll() is None,'Scorer exited before '+name)
            require(not errors,'Accounting error: '+repr(errors))
            require(time.monotonic()<deadline,'Timeout waiting '+name);time.sleep(.2)
        return json.loads((out/name).read_text())
    def interrupted(signum,frame):raise RuntimeError('Own scorer supervisor interrupted '+str(signum))
    def permit(name, phase):
        require(pid_identity(child.pid)==result['scorer_identity'], 'Authorization recipient identity changed')
        write_json(out/name, dict(phase=phase,pid=child.pid,
            starttime_ticks=result['scorer_identity']['starttime_ticks'],
            job=os.environ['SLURM_JOB_ID'],provenance_sha256=result['provenance_sha256']))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        child=owned.spawn([sys.executable,str(Path(__file__).with_name('worker.py')),'--provenance',args.provenance,
                          '--candidate',args.candidate,'--directory',str(out)],out/'worker.log',env,pass_fds=fds)
        result['scorer_identity']=pid_identity(child.pid)
        cpu=wait_file('cpu-ready.json');require(cpu['pid']==child.pid and cpu['ppid']==os.getpid(),'CPUready own identity differs')
        require(cpu['starttime_ticks']==result['scorer_identity']['starttime_ticks']
                and cpu['provenance_sha256']==result['provenance_sha256'] and cpu['bundle']==bundle,
                'CPU ready process/source bundle differs')
        same_cpu_identity(cpu['snapshot'],provenance)
        validate_inherited_gpu_leases(8)
        permit('allow-context.json','CONTEXT_ONLY')
        context=wait_file('context-ready.json');require(context['pid']==child.pid,'Context own identity differs')
        context_receipt(context,result['scorer_identity'],[x['uuid'] for x in leases])
        require(pid_identity(child.pid)==result['scorer_identity'],'Scorer identity changed')
        # The generic c7 accounting service field is the scorer, NOT a retriever.
        runtime=dict(baseline=baseline,cgroup=relative,retriever_identity=result['scorer_identity'],accounted_role='SCORER_ONLY_NO_RETRIEVER')
        validate_inherited_gpu_leases(8)
        first=active_gate(runtime,pre_model=True);write_json(out/'prestage-accounting.json',first)
        monitor_thread=threading.Thread(target=monitor,daemon=True);monitor_thread.start()
        permit('allow-scoring.json','SCORING_ONLY')
        deadline=time.monotonic()+8000
        while not (out/'scoring-done.json').exists():
            require(child.poll() is None,'Scorer exited before completion handshake')
            require(not errors,'Foreign accounting/sampling failed '+repr(errors))
            require(time.monotonic()<deadline,'Scorer overall deadline');time.sleep(.2)
        require(json.loads((out/'scoring-done.json').read_text())['pid']==child.pid,'Done own PID differs')
        result['monitor_stop']=launcher.stop_monitor(stop,monitor_thread)
        permit('allow-exit.json','EXIT_ONLY')
        child.wait(timeout=30)
        require(child.returncode==0,'Own scorer failed; retain all evidence')
        scored=json.loads((out/'scoring/result.json').read_text())
        require(scored['status']=='SCORER_NUMERICAL_RELEASE_COMPLETE_UNREVIEWED_NOT_TRAINING_READY','Incomplete scorer gate')
        result.update(status='NUMERICAL_RELEASE_COMPLETE_UNREVIEWED_NO_TRAINING',scoring_result_sha256=digest(out/'scoring/result.json'))
    except BaseException as exc:
        result.update(status='FAILED_NO_ACCEPTANCE',error=repr(exc));raise
    finally:
        try:
            result['monitor_stop']=launcher.stop_monitor(stop,monitor_thread)
            owned.cleanup();result['cleanup']=owned.cleanup_record
            final=nvml_snapshot();write_json(out/'final-nvml.json',final)
            old={(x['uuid'],x['pid']) for x in baseline['applications']};now={(x['uuid'],x['pid']) for x in final['applications']}
            require(old<=now,'Foreign baseline PIDUUID disappeared')
            for pid,identity in baseline['identities'].items():require(final['identities'].get(pid)==identity,'Foreign baseline identity changed')
            require(child is None or not any(x['pid']==child.pid for x in final['applications']),'Scorer remains on GPU after cleanup')
            require(not errors,'Sampling/accounting failure retained')
            result['foreign_baseline_preserved']=True
        except BaseException as exc:
            result.update(status='FAILED_NO_ACCEPTANCE',cleanup_error=repr(exc));raise
        finally:
            result['sampler_errors']=errors
            result['parent_cuda_initialized']='torch' in sys.modules and sys.modules['torch'].cuda.is_initialized()
            if result['parent_cuda_initialized']:result['status']='FAILED_NO_ACCEPTANCE'
            write_json(out/'supervisor-result.json',result)

if __name__=='__main__':main()
