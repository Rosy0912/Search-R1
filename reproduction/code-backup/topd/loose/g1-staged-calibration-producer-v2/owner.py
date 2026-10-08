#!/usr/bin/env python3
"""Two owned calibration children. Parent never imports Torch or initializes CUDA."""
import argparse, fcntl, hashlib, importlib.util, json, os, pathlib, socket, subprocess, sys, time
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd')
from original_blocks import DUMP_SHA,block_hashes

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def atomic(path,value):
    p=pathlib.Path(path);tmp=p.with_suffix(p.suffix+'.tmp')
    with open(tmp,'x') as f:os.chmod(tmp,0o600);json.dump(value,f,sort_keys=True);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p);fd=os.open(p.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
def support(path,expected):
    if sha(path)!=expected:raise ValueError('owned lifecycle source bytes mismatch')
    spec=importlib.util.spec_from_file_location('owned_g1_support',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def leases(records):
    if len(records)!=8 or len({x['uuid'] for x in records})!=8:raise ValueError('need8unique actual leases')
    for record in records:
        fd=int(record['fd']);path=pathlib.Path('/tmp')/f'search-r1-gpu-leases-{os.getuid()}'/(record['uuid']+'.lock')
        if os.fstat(fd).st_ino!=path.stat().st_ino:raise ValueError('lease inode mismatch')
        other=os.open(path,os.O_RDWR)
        try:
            try:fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:pass
            else:raise ValueError('lease not locked')
            fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        finally:os.close(other)
    return tuple(int(x['fd']) for x in records)

def gpu_rows():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True,timeout=30)
    return {uuid.strip():int(index.strip()) for index,uuid in (line.split(',') for line in raw.splitlines())}

def owned_gpu_remaining(snapshot,pids):
    return [line for line in snapshot['processes'] if int(line.split(',')[1].strip()) in pids]

def gpu_baseline(tools,snapshot):
    records=[]
    for line in snapshot['processes']:
        pieces=[x.strip() for x in line.split(',')];pid=int(pieces[1]);identity=tools.process_identity(pid)
        if 'starttime_ticks' not in identity:raise ValueError('initial GPU PID identity unavailable')
        records.append({'uuid':pieces[0],'pid':pid,'identity':identity})
    return records

def baseline_gpu_missing(tools,snapshot,records):
    present={(line.split(',')[0].strip(),int(line.split(',')[1].strip())) for line in snapshot['processes']}
    return [r for r in records if (r['uuid'],r['pid']) not in present or tools.process_identity(r['pid']).get('starttime_ticks')!=r['identity']['starttime_ticks']]

def run_stage(phase,directory,config_path,config,tools,baseline,global_gpu_baseline):
    if baseline_gpu_missing(tools,tools.gpu_sample(),global_gpu_baseline):raise ValueError('original global GPU baseline identity changed before stage')
    scope,relative=tools.cgroup_scope(str(config['job']));owned=tools.OwnedProcesses(scope,relative)
    environment=dict(os.environ)
    for key in list(environment):
        if key.lower() in ('http_proxy','https_proxy','all_proxy','no_proxy'):environment.pop(key)
    environment.update(VLLM_WORKER_MULTIPROC_METHOD='spawn',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',NO_PROXY='127.0.0.1,localhost',TORCH_ALLOW_TF32_CUBLAS_OVERRIDE='0',NVIDIA_TF32_OVERRIDE='0',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')
    environment['CUDA_VISIBLE_DEVICES']=','.join(str(gpu_rows()[u]) for u in config['uuids']) if phase=='A' else ','.join(config['uuids'])
    identities={};monitor=[];child=None;error=None
    try:
        child=owned.spawn([sys.executable,str(pathlib.Path(__file__).with_name('stage_worker.py')),'--config',str(config_path),'--phase',phase,'--directory',str(directory)],directory/f'phase-{phase}.log',env=environment,pass_fds=leases(config['leases']))
        identities[child.pid]=tools.process_identity(child.pid)
        began=time.monotonic()
        while child.poll() is None:
            current=tools.cgroup_pids(scope)-owned.baseline
            for pid in current:identities[pid]=tools.process_identity(pid)
            monitor.append({'epoch':time.time(),'gpu':tools.gpu_sample(),'own_process_samples':[tools.process_sample(pid) for pid in sorted(current)]})
            if time.monotonic()-began>config['stage_timeout_seconds']:raise TimeoutError('owned calibration stage timeout')
            time.sleep(1)
        if child.returncode!=0:raise RuntimeError(f'owned phase{phase} exited {child.returncode}')
        receipt=json.loads((directory/f'phase-{phase}-receipt.json').read_text())
        if receipt['pid']!=child.pid or receipt['config_sha256']!=sha(config_path) or receipt['source_sha256']!=DUMP_SHA or receipt['uuids']!=config['uuids']:raise ValueError('stage receipt identity mismatch')
    except BaseException as exc:error=exc
    finally:
        # All actual own-step new processes (including vLLM workers) must exit.
        current=tools.cgroup_pids(scope)-owned.baseline
        for pid in current:identities[pid]=tools.process_identity(pid)
        try:owned.cleanup()
        except BaseException as exc:
            if error is None:error=exc
        remaining=sorted(tools.cgroup_pids(scope)-owned.baseline);after=tools.gpu_sample()
        global_missing=baseline_gpu_missing(tools,after,global_gpu_baseline)
        missing=[pid for pid,identity in baseline.items() if tools.process_identity(pid).get('starttime_ticks')!=identity.get('starttime_ticks')]
        seen=set()
        for sample in monitor:
            for line in sample['gpu']['processes']:
                pieces=[x.strip() for x in line.split(',')]
                if int(pieces[1]) in identities:seen.add(pieces[0])
        barrier={'original_global_gpu_baseline':global_gpu_baseline,'original_global_gpu_baseline_missing':global_missing,'original_global_gpu_baseline_preserved':not global_missing,'seen_actual_gpu_uuids':sorted(seen),'scope':'ACTUAL_OWNED_STAGE_EXIT_NO_TRAINING','phase':phase,'child_pid':child.pid if child else None,'returncode':child.returncode if child else None,'identities':list(identities.values()),'cleanup':getattr(owned,'cleanup_record',None),'own_processes_remaining':remaining,'own_processes_empty':not remaining,'own_gpu_records_remaining':owned_gpu_remaining(after,set(identities)),'own_gpu_pids_empty':not owned_gpu_remaining(after,set(identities)),'baseline_identity_missing':missing,'baseline_preserved':not missing,'gpu_after':after,'monitor_samples':monitor,'config_sha256':sha(config_path),'parent_cuda_initialized':False}
        if phase=='A' and (directory/'generation-trace.json').exists():
            barrier.update(trace_sha256=sha(directory/'generation-trace.json'),token_trace_sha256=sha(directory/'hidden-input-token-trace.json'))
        atomic(directory/f'phase-{phase}-owner-barrier.json',barrier)
        if remaining or barrier['own_gpu_records_remaining'] or missing or global_missing or seen!=set(config['uuids']):
            if error is None:error=RuntimeError('actual stage release/baseline protection failed')
    if error:raise error
    return barrier

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--directory',required=True);ap.add_argument('--support-sha256',required=True);ap.add_argument('--manifest');ap.add_argument('--dump-only',action='store_true');a=ap.parse_args();os.umask(0o077)
    if socket.gethostname()!='h100-11' or os.environ.get('SLURM_JOB_NODELIST')!='h100-11':raise ValueError('own h11 Slurm only')
    config=json.loads(pathlib.Path(a.config).read_text());directory=pathlib.Path(a.directory)
    if str(config['job'])!=os.environ.get('SLURM_JOB_ID') or not directory.resolve().is_relative_to((ROOT/'logs').resolve()):raise ValueError('own job/log directory only')
    if config['dump_args']['layers']!='21,24,27' or config['dump_args']['G']!=5 or config['dump_args']['max_turns']!=4:raise ValueError('original G1 calibration protocol changed')
    if not pathlib.Path(config['dump_args']['model']).resolve().is_relative_to(ROOT.resolve()) or not pathlib.Path(config['dump_args']['data']).resolve().is_relative_to(ROOT.resolve()):raise ValueError('non-own local inputs')
    if not os.environ.get('RETRIEVER_URL','').startswith('http://127.0.0.1:'):raise ValueError('own loopback retrieval only')
    from gates import conditional_base64
    for key in ('result_path','qa_path'):
        if not pathlib.Path(config['base64_binding'][key]).resolve().is_relative_to(ROOT.resolve()):raise ValueError('non-own base64 evidence')
    config['base64_route']=conditional_base64(config['base64_binding'])
    tools=support(pathlib.Path(__file__).with_name('owner_support.py'),a.support_sha256);scope,relative=tools.cgroup_scope(str(config['job']))
    if config['cgroup']!=relative:raise ValueError('own step cgroup mismatch')
    leases(config['leases']);config['uuids']=[x['uuid'] for x in config['leases']]
    config['model_files']={str(p.relative_to(pathlib.Path(config['dump_args']['model']))):sha(p) for p in pathlib.Path(config['dump_args']['model']).rglob('*') if p.is_file() and p.suffix in ('.json','.safetensors','.txt')}
    config['model_files_inventory_sha256']=hashlib.sha256(json.dumps(config['model_files'],sort_keys=True).encode()).hexdigest()
    config['data_sha256']=sha(config['dump_args']['data'])
    config['canonical_data']=str(pathlib.Path(config['dump_args']['data']).resolve());config['canonical_model']=str(pathlib.Path(config['dump_args']['model']).resolve())
    directory.mkdir(parents=True,exist_ok=False);config_path=directory/'frozen-input-config.json';atomic(config_path,config)
    baseline={pid:tools.process_identity(pid) for pid in tools.cgroup_pids(scope)}
    gpu_initial=tools.gpu_sample();global_baseline=gpu_baseline(tools,gpu_initial);result={'scope':'STAGED_G1_CALIBRATION_PROTOTYPE_NO_TRAINING','status':'RUNNING','complete':False,'source_sha256':DUMP_SHA,'blocks':block_hashes(pathlib.Path(__file__).parent/'original/ts_probe_dump.py'),'config_sha256':sha(config_path),'support_sha256':a.support_sha256,'gpu_initial':gpu_initial}
    try:
        result['phaseA']=run_stage('A',directory,config_path,config,tools,baseline,global_baseline)
        result['phaseB']=run_stage('B',directory,config_path,config,tools,baseline,global_baseline)
        dump=directory/'dump'/f"{config['dump_args']['tag']}_{config['dump_args']['offset']}.npz"
        result.update(status='DUMP_COMPLETE_HARDWARE_NOT_INDEPENDENTLY_ACCEPTED',complete=True,npz_sha256=sha(dump),jsonl_sha256=sha(dump.with_suffix('.jsonl')))
    except BaseException as exc:result.update(status='FAILED_NO_PUBLICATION',error_type=type(exc).__name__,error=str(exc));raise
    finally:atomic(directory/'result.json',result)
    if a.dump_only:
        print(json.dumps({'scope':result['scope'],'status':result['status'],'dump':str(directory/'dump')}));return
    if not a.manifest:raise ValueError('complete producer requires own live manifest target')
    heads=directory/'heads';heads.mkdir()
    scope,relative=tools.cgroup_scope(str(config['job']));owned=tools.OwnedProcesses(scope,relative);child=None;head_identity=None;head_error=None
    try:
        train=pathlib.Path(__file__).parent/'original/ts_probe_train.py'
        from publication import TRAIN_SHA,publish
        if sha(train)!=TRAIN_SHA:raise ValueError('original train source mismatch')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')
        child=owned.spawn([sys.executable,str(train),'--dumps',str(directory/'dump'),'--layers','21,24,27','--commit-position','Xa','--commit-label','flip','--save',str(heads)],directory/'heads.log',env=env)
        head_identity=tools.process_identity(child.pid)
        if child.wait(timeout=1800)!=0:raise RuntimeError('original dualhead fitting failed')
    except BaseException as exc:head_error=exc
    finally:
        try:owned.cleanup()
        except BaseException as exc:
            if head_error is None:head_error=exc
        gpu_after=tools.gpu_sample();remaining=sorted(tools.cgroup_pids(scope)-owned.baseline)
        missing=[pid for pid,identity in baseline.items() if tools.process_identity(pid).get('starttime_ticks')!=identity.get('starttime_ticks')]
        head_exit={'returncode':child.returncode if child else None,'identity':head_identity,'own_processes_remaining':remaining,'own_gpu_records_remaining':owned_gpu_remaining(gpu_after,{child.pid} if child else set()),'baseline_preserved':not missing,'baseline_identity_missing':missing,'cleanup':getattr(owned,'cleanup_record',None)}
        atomic(directory/'heads-owner-exit.json',head_exit)
    if head_error:
        result.update(status='FAILED_HEADS_NO_PUBLICATION',complete=False,error_type=type(head_error).__name__,error=str(head_error));atomic(directory/'result.json',result);raise head_error
    if not pathlib.Path(a.manifest).resolve().is_relative_to(ROOT.resolve()):raise ValueError('manifest non-ownROOT')
    receipt=publish(directory,a.manifest,heads,head_exit)
    # No fallible artifact writes after live publication. Prepared raw evidence plus
    # actual live bytes/heads/exit is the independent verifier's source of truth.
    print(json.dumps({'scope':result['scope'],'status':'CALIBRATION_PUBLISHED_UNREVIEWED_NO_TRAINING','manifest_sha256':receipt['manifest_file_sha256']}))
if __name__=='__main__':main()
