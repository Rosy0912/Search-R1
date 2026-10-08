"""Isolated candidate for one owned8rank original TP1 generation window; no deployment here."""
import argparse
import copy
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
from probe_package import (ROOT,PROTECTED,allocation,atomic_json,bundle_payload,decode,digest,file_sha,
    guard_record,rank_records,read_bound,require,runtime_output,same_node_path,validate_package_files,validate_release,
    validate_owned_lease_fds,RUNNER_MODULE,WINDOW_PROFILE,WINDOW_CONTRACT)


def process(pid):
    fields=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
    require(fields[0]!='Z','Process exited: '+str(pid))
    return dict(pid=pid,start_ticks=int(fields[19]))


def protected():
    rows={pid:process(pid)['start_ticks'] for pid in PROTECTED}
    require(rows==PROTECTED,'Original background PID/start identity changed')
    return rows


def healthy_guard(path,job,uuids):
    lines=Path(path).read_text().splitlines()
    require(lines,'Resource guard has no samples')
    return guard_record(decode(lines[-1].encode()),job,uuids,time.time())


def owned_adapter(entry):
    path=same_node_path(entry['path']); require(file_sha(path)==entry['sha256'],'Adapter source changed')
    spec=importlib.util.spec_from_file_location('tree_probe_parent_resource_adapter',str(path))
    value=importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    require(all(callable(getattr(value,key,None)) for key in ('prepare_and_validate_bundle','check')),
            'Explicit resource adapter API missing')
    return value


def wait_probe(probe,guard,leases,uuids,adapter,owned,scope,process_sample,cgroup_pids,run,mono):
    """Actual parent polling body, also exercised with CPU-only controlled peers."""
    with (Path(run)/'process-resources.jsonl').open('a') as stream:
        while probe.poll() is None:
            require(guard.poll() is None and time.monotonic()-mono<1620,
                    'Guard exited or own diagnostic time limit')
            validate_owned_lease_fds(leases,uuids);check=adapter.check('supervisor_live')
            stream.write(json.dumps(dict(epoch=time.time(),adapter=check,processes=[process_sample(pid) for pid in
                sorted(cgroup_pids(scope)-owned.baseline)]),allow_nan=False)+'\n')
            stream.flush();time.sleep(1)
    require(probe.returncode==0,'Actual8rank diagnostic failed')


def compiler_header_environment():
    """Restore the original same-node Python3.10 compiler include environment.

    Existing extracted headers are read only. Do not install/download headers,
    fall back to another interpreter or bypass native compilation on failure.
    """
    header_root=ROOT/'python-dev'
    include=header_root/'usr/include'
    primary=include/'python3.10'
    require(header_root.is_dir() and header_root.resolve()==header_root,
            'Existing same-node Python header root missing or aliased')
    for directory in (primary,include):
        require(directory.is_dir() and header_root in directory.resolve().parents,
                'Existing same-node Python include directory missing or escaped')
    for header in (primary/'Python.h',primary/'pyconfig.h',include/'x86_64-linux-gnu/python3.10/pyconfig.h'):
        require(header.is_file() and header_root in header.resolve().parents,
                'Existing same-node Python header missing or escaped: '+str(header))
        with header.open('rb') as stream:
            require(bool(stream.read(1)),'Existing Python header unreadable or empty: '+str(header))
    return {'C_INCLUDE_PATH':str(primary)+':'+str(include)}


def main():
    started=time.time(); mono=time.monotonic()
    parser=argparse.ArgumentParser();parser.add_argument('--package',required=True)
    parser.add_argument('--manifest-sha256','--manifest-sha',required=True);parser.add_argument('--release',required=True)
    parser.add_argument('--release-sha256','--release-sha',required=True);args=parser.parse_args()
    package=same_node_path(args.package)
    require(package==Path(__file__).resolve().parent,'Supervisor/package differs')
    manifest_sha=args.manifest_sha256
    require(os.environ.get('TREE_PROBE_PACKAGE_MANIFEST_SHA256')==manifest_sha and
            same_node_path(os.environ['TREE_PROBE_PACKAGE_MANIFEST_PATH'])==package/'package-manifest.json',
            'Inherited package pin differs from explicit argument')
    manifest=read_bound(os.environ['TREE_PROBE_PACKAGE_MANIFEST_PATH'],manifest_sha)
    validate_package_files(package,manifest)
    template=read_bound(package/'bundle-template.json',manifest['files']['bundle-template.json'])
    release_path=same_node_path(args.release);release_sha=args.release_sha256
    require(os.environ.get('TREE_PROBE_RELEASE_SHA256')==release_sha and
            same_node_path(os.environ['TREE_PROBE_RELEASE_PATH'])==release_path,'Inherited release pin differs')
    release=read_bound(release_path,release_sha);validate_release(manifest,manifest_sha,template,release)
    job=os.environ.get('SLURM_JOB_ID','')
    require(re.fullmatch('[1-9][0-9]*',job) and socket.gethostname().split('.')[0]=='h100-9' and
            os.environ.get('SLURM_JOB_NODELIST')=='h100-9' and os.environ.get('SLURM_JOB_GPUS'),'Wrong actual allocation')
    require(len(os.environ.get('CUDA_VISIBLE_DEVICES','').split(','))==8,'Eight actual selected GPUs required')
    # Imported exact accepted package helper: this performs same-OFD idempotent flock.
    from search_r1.baselines.preflight import validate_inherited_gpu_leases,gpu_snapshot,check_selected_gpus
    import search_r1.baselines.preflight as preflight
    require(Path(preflight.__file__).resolve()==package/'repo/search_r1/baselines/preflight.py',
            'Imported preflight differs from frozen package')
    validate_inherited_gpu_leases(8)
    background_before=protected()
    lock=(ROOT/'shared-gpu-job.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    from tree_shared_owned_runtime_fixed import cgroup_scope,OwnedProcesses,gpu_sample,process_sample,cgroup_pids,available
    import tree_shared_owned_runtime_fixed as owned_source
    require(Path(owned_source.__file__).resolve()==ROOT/'tools/tree_shared_owned_runtime_fixed.py',
            'Owned cleanup helper import path differs')
    inventory,processes=gpu_snapshot();selected=check_selected_gpus(os.environ['CUDA_VISIBLE_DEVICES'].split(','),inventory,processes)
    uuids=sorted(row['uuid'] for row in selected)
    require(len(uuids)==8 and set(decode((ROOT/'tools/expected-gpu-uuids.json').read_bytes()))==set(uuids),
            'Original eight GPU UUIDs changed')
    leases=decode(os.environ['SEARCH_R1_GPU_LEASES'].encode())
    require(len(leases)==8 and {row['uuid'] for row in leases}==set(uuids),'Inherited lease UUID set changed')
    allocated=subprocess.check_output(['scontrol','show','job',job],text=True)
    allocated_cpus=allocation(allocated,job)
    class CandidateOwnedProcesses(OwnedProcesses):
        def spawn(self,command,log,env=None,pass_fds=()):
            return super().spawn(['bash','-c','cd "$TREE_PROBE_REPO" && exec "$@"','probe-cwd',*command],
                                 log,env,pass_fds)
    scope,relative=cgroup_scope(job);owned=CandidateOwnedProcesses(scope,relative)
    require(not(set(PROTECTED)&cgroup_pids(scope)),'Protected original processes appear in own cgroup; refuse cleanup')
    port=manifest['torchrun_port'];require(type(port) is int and 1024<=port<65536,'Invalid reviewed loopback port');available(port)
    run=Path(runtime_output(job)).parent;run.mkdir()
    output=run/'gpu-records';output.mkdir()
    atomic_json(run/'initial-allocation.json',dict(job_id=job,allocation_text=allocated,allocated_cpus=allocated_cpus,
        allocated_gpus=8,requested_cpus=32,initial_gpu=gpu_sample(),background_pid_start_ticks=background_before,
        package_manifest_sha256=manifest_sha,release_sha256=release_sha))
    for key in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy','WANDB_API_KEY','RAY_ADDRESS'):
        os.environ.pop(key,None)
    os.environ.update(compiler_header_environment())
    os.environ.update(TREE_PROBE_REPO=str(package/'repo'),PYTHONPATH=str(package/'repo')+':'+str(package)+':'+str(ROOT/'tools'),
        PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        NCCL_NVLS_ENABLE='0',NCCL_SOCKET_IFNAME='lo',GLOO_SOCKET_IFNAME='lo',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        HF_DATASETS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',NO_PROXY='127.0.0.1,localhost,::1',no_proxy='127.0.0.1,localhost,::1')
    result=dict(status='FAILED_OR_UNKNOWN_ZERO_UPDATE_GPU_PROBE',job_id=job,actual_training_accepted=False,
        probability_consistency_accepted=False,gpu_training_capacity_proven=False,planned_optimizer_writes=0,actual_optimizer_writes=None,
        requested_cpus=32,allocated_cpus=allocated_cpus,allocated_gpus=8,package_manifest_sha256=manifest_sha,
        release_sha256=release_sha,selection_sha256=manifest['selection_sha256'],runner_commit=manifest['runner_commit'],
        diagnostic_profile=WINDOW_PROFILE,window_contract=WINDOW_CONTRACT,window_inputs_sha256=manifest['window_inputs_sha256'],
        actual_generation_invocations=None,actual_actor_scoring_native_forwards=None,actual_new_emitted_tokens=None)
    guard=None;probe=None;adapter=None
    def interrupted(sig,frame): raise RuntimeError('Own diagnostic interrupted '+str(sig))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        guard_path=run/'resources.jsonl'
        guard=owned.spawn([sys.executable,str(ROOT/'tools/resource_guard_identity.py'),'--output',str(guard_path)],run/'guard.log')
        deadline=time.monotonic()+30
        while not (guard_path.is_file() and guard_path.with_suffix('.ready').is_file()):
            require(guard.poll() is None and time.monotonic()<deadline,'Own guard startup failed');time.sleep(.2)
        healthy_guard(guard_path,job,uuids)
        bundle=copy.deepcopy(template);bundle['output']=str(output);bundle['release']=dict(path=str(release_path),sha256=release_sha)
        require(digest(bundle_payload(bundle,job))==manifest['bundle_payload_sha256'],'Runtime bundle payload changed')
        bundle_entry=atomic_json(run/'bundle.json',bundle)
        launch=dict(schema_version=1,job_id=job,bundle_sha256=bundle_entry['sha256'],
            bundle_payload_sha256=manifest['bundle_payload_sha256'],package_manifest_sha256=manifest_sha,
            package_manifest_path=str(package/'package-manifest.json'),release_sha256=release_sha,
            gpu_uuids=uuids,leases=leases,supervisor=process(os.getpid()),guard=process(guard.pid),
            guard_path=str(guard_path),allocation_text=allocated,cgroup=str(scope),cgroup_relative=relative,
            background_pid_start_ticks=background_before,shared_lock_fd=lock.fileno(),shared_lock_path=str(ROOT/'shared-gpu-job.lock'),
            source_manifest_files=manifest['files'],start_epoch=started,torchrun_loopback_port=port,
            compiler_environment=dict(C_INCLUDE_PATH=os.environ['C_INCLUDE_PATH']),
            diagnostic_profile=WINDOW_PROFILE,window_contract=WINDOW_CONTRACT,window_inputs_sha256=manifest['window_inputs_sha256'],
            installed_kernel_binaries=manifest['installed_kernel_binaries'],actual_loaded_binary_identity_proven=False,
            runtime_scope='One original TP1 call5 generation/rank on complete320 prompts,80micro4 actor scores/rank; no initialized Ray/retrieval/Tree outer/evaluation/optimizer/scheduler/backward; capacity unproven')
        launch_entry=atomic_json(run/'launch.json',launch)
        os.environ.update(TREE_PROBE_BUNDLE_PATH=bundle_entry['path'],TREE_PROBE_BUNDLE_SHA256=bundle_entry['sha256'],
            TREE_PROBE_LAUNCH_PATH=launch_entry['path'],TREE_PROBE_LAUNCH_SHA256=launch_entry['sha256'])
        adapter=owned_adapter(bundle['reviewed_resource_adapter'])
        require(adapter.prepare_and_validate_bundle(bundle) is True,'Parent complete resource/identity gate failed')
        # Parent validation may import Torch and passive Ray, but must initialize neither.
        torch=sys.modules.get('torch');ray=sys.modules.get('ray')
        require(torch is None or not torch.cuda.is_initialized(),'Parent gate initialized CUDA')
        require(ray is None or not ray.is_initialized(),'Parent gate initialized Ray')
        require(not any(name=='vllm' or name.startswith('vllm.') for name in sys.modules),'Parent gate imported vLLM')
        validate_owned_lease_fds(leases,uuids);adapter.check('supervisor_before_torchrun')
        # Parent retains lease FDs; child/ranks deliberately do not inherit them.
        command=[sys.executable,'-m','torch.distributed.run','--nnodes=1','--nproc_per_node=8','--master_addr=127.0.0.1',
            '--master_port='+str(port),'-m',RUNNER_MODULE,'--bundle',bundle_entry['path'],'--bundle-sha',bundle_entry['sha256']]
        atomic_json(run/'torchrun-command.json',dict(command=command,command_sha256=digest(command),lease_fds_inherited_by_children=False))
        probe=owned.spawn(command,run/'probe.log')
        wait_probe(probe,guard,leases,uuids,adapter,owned,scope,process_sample,cgroup_pids,run,mono)
        require(guard.poll() is None,'Guard exited before final rank checks')
        validate_owned_lease_fds(leases,uuids);adapter.check('supervisor_completed')
        records=rank_records(output,bundle,bundle_entry['sha256'])
        samples=[decode(line.encode()) for line in guard_path.read_text().splitlines()]
        require(samples,'No actual guard samples')
        for row in samples: guard_record(row,job,uuids,row['epoch'])
        healthy_guard(guard_path,job,uuids)
        protected();result.update(diagnostic_body_complete=True,probe_exit_code=probe.returncode,
            actual_optimizer_writes=0,
            actual_generation_invocations=sum(x['record']['generation_invocations'] for x in records),
            actual_actor_scoring_native_forwards=sum(x['record']['actor_scoring_native_forwards'] for x in records),
            actual_new_emitted_tokens=sum(x['record']['actual_new_emitted_tokens'] for x in records),
            rank_records=records,guard_samples=len(samples),final_guard_healthy=True,bundle_sha256=bundle_entry['sha256'],
            launch_sha256=launch_entry['sha256'])
    except BaseException as error:
        result.update(error_type=type(error).__name__,error=str(error))
        raise
    finally:
        signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
        try:
            atomic_json(run/'termination-cost.json',dict(job_id=job,start_epoch=started,termination_epoch=time.time(),
                allocated_gpus=8,requested_cpus=32,allocated_cpus=allocated_cpus,scope='Own supervisor before cleanup; launcher/preflight and scheduler separate'))
        except BaseException as error:
            # Evidence write failure must never bypass own process cleanup.
            result['termination_cost_error']=repr(error)
        try:
            owned.cleanup();result['cleanup_residual']=owned.cleanup_residual
            result['final_gpu']=gpu_sample();protected()
            remaining={int(line.split(',')[1].strip()) for line in result['final_gpu']['processes']}
            require(remaining==set(PROTECTED),'Original8 GPU processes must be preserved, own GPU processes absent')
            result['original_background_pid_start_ticks_preserved']=True
            validate_inherited_gpu_leases(8)
            if result.get('diagnostic_body_complete') and not result.get('termination_cost_error'):
                result['status']='OWNED_SINGLE_ORIGINAL_TP1_WINDOW_OBSERVED_PENDING_INDEPENDENT_REVIEW'
        except BaseException as error:
            result.update(status='FAILED_OR_UNKNOWN_ZERO_UPDATE_GPU_PROBE',cleanup_error=repr(error));raise
        finally:
            result.update(start_epoch=started,end_epoch=time.time(),wall_seconds=time.time()-started,
                gpu_hours=8*(time.time()-started)/3600,allocated_cpu_hours=allocated_cpus*(time.time()-started)/3600,
                consumed_cpu_scope='Per-process samples and per-rank ready; allocation hours are not consumed CPU time')
            # During this body job remains RUNNING; actual terminal is a subsequent read-only accounting step.
            try:result['slurm_at_supervisor_end']=subprocess.check_output(['scontrol','show','job',job],text=True)
            except BaseException as error:result['slurm_at_supervisor_end_unknown']=repr(error)
            atomic_json(run/'result.json',result)
            lock.close()


if __name__=='__main__':main()
