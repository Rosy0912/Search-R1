"""Own-node teacher IPC. No rewards, optimizer, collectives, or remote endpoint."""
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import struct
import time

import torch
from safetensors.torch import save as tensor_save, load as tensor_load

IDENTITY=('run_id','code_commit','config_sha256','data_identity','teacher_receipt_sha256','source_sha256')
TENSORS={'prompts','responses','attention','row_indices'}
MAX_FRAME=128*1024*1024


def require(condition,message):
    if not condition:raise RuntimeError(message)


def digest(raw):return hashlib.sha256(raw).hexdigest()


def receive_exact(connection,size):
    require(0<=size<=MAX_FRAME,'IPC frame exceeds bound')
    chunks=[]
    while size:
        part=connection.recv(min(size,1024*1024))
        require(part,'Incomplete own teacher IPC frame')
        chunks.append(part);size-=len(part)
    return b''.join(chunks)


def receive(connection):
    lengths=struct.unpack('!QQ',receive_exact(connection,16))
    require(lengths[0]<=65536,'Oversized teacher metadata')
    metadata=json.loads(receive_exact(connection,lengths[0]))
    raw=receive_exact(connection,lengths[1])
    return metadata,raw


def send(connection,metadata,raw):
    encoded=json.dumps(metadata,allow_nan=False,sort_keys=True).encode()
    require(len(encoded)<=65536 and len(raw)<=MAX_FRAME,'IPC output exceeds bound')
    connection.sendall(struct.pack('!QQ',len(encoded),len(raw))+encoded+raw)


def lease_ownership(record):
    """Only descriptor ownership, never occupancy during active scoring models."""
    import fcntl
    fd=int(record['fd']);path=Path('/tmp')/f'search-r1-gpu-leases-{os.getuid()}'/(record['uuid']+'.lock')
    require(os.fstat(fd).st_ino==path.stat().st_ino,'Scorer lease file mismatch')
    probe=os.open(path,os.O_RDWR)
    try:
        try:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:pass
        else:raise RuntimeError('Scorer lease is not held')
        try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:raise RuntimeError('Scorer descriptor is foreign') from error
    finally:os.close(probe)


def own_peer(connection,scope):
    require(hasattr(socket,'SO_PEERCRED'),'Need Linux peer credentials')
    pid,uid,_=struct.unpack('3i',connection.getsockopt(socket.SOL_SOCKET,socket.SO_PEERCRED,12))
    require(uid==os.getuid(),'Teacher IPC peer UID differs')
    rows=Path(f'/proc/{pid}/cgroup').read_text().splitlines()
    require(any(row.split(':',2)[-1]==scope or row.split(':',2)[-1].startswith(scope+'/') for row in rows),'Teacher peer outside own Slurm cgroup')
    return pid


def snapshot(teacher):
    from .topd_runtime_audit import frozen_teacher_snapshot
    record=frozen_teacher_snapshot(teacher)
    record['buffer_devices']=sorted({str(buffer.device) for buffer in teacher.model.buffers()})
    return record



def cuda_process_snapshot(uuid):
    import subprocess
    text=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=10)
    matches=[]
    for line in text.splitlines():
        actual,pid,memory=[field.strip() for field in line.split(',')]
        if int(pid)==os.getpid():
            require(actual==uuid,'Actual scorer CUDA process is on a foreign UUID')
            matches.append(dict(uuid=actual,pid=int(pid),used_mib=float(memory)))
    require(len(matches)==1,'Scorer actual CUDA process UUID/memory unavailable')
    return matches[0]


def initialize_cuda_context(device):
    """Create an own visible context without scoring/model precision changes."""
    token=torch.empty((),device=device,dtype=torch.float32)
    torch.cuda.synchronize()
    del token
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def release_cuda(teacher):
    """Private own-process clearing is fail-closed pending native proof."""
    teacher.model.to('cpu');teacher.device=torch.device('cpu')
    torch.cuda.synchronize()
    require(torch.__version__.split('+')[0]=='2.4.0','Private workspace release requires reviewed pinned torch2.4.0')
    clear=getattr(torch._C,'_cuda_clearCublasWorkspaces',None)
    require(callable(clear),'Pinned torch lacks verified own cuBLAS workspace release')
    clear();torch.cuda.empty_cache();torch.cuda.synchronize()
    return dict(allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved())


class ScorerSession:
    def __init__(self,teacher,manifest,rank,uuid,ownership,device='cuda:0',release=release_cuda):
        self.teacher,self.manifest,self.rank,self.uuid=teacher,manifest,rank,uuid
        self.ownership,self.device,self.release=ownership,torch.device(device),release
        self.initial=snapshot(teacher)
        self.next_step=1
        require(self.initial['devices']==['cpu'] and not self.initial['training']
                and all(device=='cpu' for device in self.initial['buffer_devices'])
                and self.initial['dtypes']==['torch.float32'] and self.initial['training_modules_count']==self.initial['requires_grad_parameters']==self.initial['grad_parameters']==0,'Teacher must start frozen/eval/CPU')

    def score(self,metadata,raw):
        expected=set(IDENTITY)|{'capability','outer_step','temperature','chunk','input_sha256'}
        require(set(metadata)==expected,'Teacher request metadata has missing/forbidden fields')
        require(all(metadata[key]==self.manifest[key] for key in IDENTITY),'Teacher request identity mismatch')
        require(metadata['capability']==self.manifest['services'][self.rank]['capability'],'Teacher request authentication failed')
        step=metadata['outer_step']
        require(type(step) is int and step==self.next_step and step<=self.manifest['outer_updates'],'Teacher outer sequence differs')
        require(metadata['input_sha256']==digest(raw),'Teacher input hash mismatch')
        tensors=tensor_load(raw)
        require(set(tensors)==TENSORS,'Teacher input has missing/forbidden tensors')
        prompts,responses,attention,indices=(tensors[key] for key in ('prompts','responses','attention','row_indices'))
        rows=int(self.manifest['rows_per_service'])
        require(prompts.dtype==responses.dtype==indices.dtype==torch.int64 and attention.dtype in (torch.int64,torch.bool),'Teacher token/mask dtype differs')
        require(prompts.ndim==responses.ndim==attention.ndim==2 and attention.shape==responses.shape and len(prompts)==len(responses)==rows,'Teacher input shape differs')
        require(torch.equal(indices,torch.arange(self.rank*rows,(self.rank+1)*rows)),'Teacher row order differs')
        require(bool(((attention==0)|(attention==1)).all()),'Teacher attention is not binary')
        temperature=metadata['temperature'];chunk=metadata['chunk']
        require(math.isfinite(temperature) and temperature>0 and temperature==self.manifest['temperature'] and type(chunk) is int and chunk==self.manifest['chunk'],'Teacher temperature/chunk differs')
        self.ownership()
        wall=time.perf_counter();cpu_before=time.process_time();tokens_before=self.teacher.scored_tokens
        before_memory=None;gpu_pid_before=gpu_pid_peak=gpu_pid_released=None
        try:
            if self.device.type=='cuda':
                torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
                require(not torch.is_autocast_enabled(),'Scorer must never enable CUDA autocast')
                initialize_cuda_context(self.device)
                gpu_pid_before=cuda_process_snapshot(self.uuid)
                before_memory=dict(allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved())
            move_start=time.perf_counter()
            self.teacher.model.to(self.device);self.teacher.device=self.device
            if self.device.type=='cuda':torch.cuda.synchronize()
            h2d=time.perf_counter()-move_start
            gpu_before=snapshot(self.teacher) if step==1 else None
            outputs=[];start=time.perf_counter()
            events=None
            if self.device.type=='cuda':
                events=(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True));events[0].record()
            for row in range(rows):
                outputs.append(self.teacher.log_probs(prompts[row:row+1],responses[row:row+1],attention[row:row+1],temperature,None,chunk))
            result=torch.cat(outputs)
            require(bool(torch.isfinite(result).all()),'Nonfinite teacher result')
            scoring=time.perf_counter()-start
            event_elapsed_ms=None
            if events:
                events[1].record();torch.cuda.synchronize();event_elapsed_ms=events[0].elapsed_time(events[1])
            gpu_after=snapshot(self.teacher) if step==self.manifest['outer_updates'] else None
            if self.device.type=='cuda':gpu_pid_peak=cuda_process_snapshot(self.uuid)
            peak=dict(allocated_bytes=torch.cuda.max_memory_allocated(),reserved_bytes=torch.cuda.max_memory_reserved()) if self.device.type=='cuda' else None
        finally:
            release_start=time.perf_counter()
            released=self.release(self.teacher)
            offload=time.perf_counter()-release_start
        cpu_after=snapshot(self.teacher)
        if self.device.type=='cuda':gpu_pid_released=cuda_process_snapshot(self.uuid)
        require(cpu_after['devices']==['cpu'] and cpu_after['parameter_sha256']==self.initial['parameter_sha256']
                and all(device=='cpu' for device in cpu_after['buffer_devices']) and not cpu_after['training'] and cpu_after['training_modules_count']==cpu_after['requires_grad_parameters']==cpu_after['grad_parameters']==0,'Teacher release frozen hash/eval/device failed')
        require(released['allocated_bytes']==released['reserved_bytes']==0,'Teacher CUDA workspace/allocator not released')
        for boundary in (gpu_before,gpu_after):
            if boundary:require(boundary['parameter_sha256']==self.initial['parameter_sha256'],'Teacher GPU frozen hash differs')
        self.ownership()
        payload=tensor_save(dict(teacher_log_probs=result.cpu().contiguous(),row_indices=indices))
        receipt={key:self.manifest[key] for key in IDENTITY}
        receipt.update(kind='teacher_scorer_outer',rank=self.rank,uuid=self.uuid,pid=os.getpid(),cgroup=self.manifest['cgroup'],slurm_job=self.manifest.get('slurm_job'),outer_step=step,
                       input_sha256=digest(raw),output_sha256=digest(payload),rows=rows,row_indices_sha256=digest(tensor_save(dict(row_indices=indices))),
                       scored_tokens_delta=self.teacher.scored_tokens-tokens_before,targets_none=True,dtype='float32',attention='eager',tf32=False,
                       loaded_cpu=self.initial,gpu_before=gpu_before,gpu_after=gpu_after,released_cpu=cpu_after,status='RELEASED',
                       allocator_before=before_memory,allocator_peak=peak,allocator_released=released,
                       gpu_pid_before=gpu_pid_before,gpu_pid_phase_sample=gpu_pid_peak,gpu_pid_released=gpu_pid_released,
                       gpu_pid_scope='nvidia-smi own PID snapshots; cumulative peaks require parent sampler; residual context can remain with allocator0',h2d_wall_seconds=h2d,scoring_wall_seconds=scoring,
                       offload_wall_seconds=offload,scoring_cuda_event_phase_ms=event_elapsed_ms,
                       cuda_event_scope='elapsed stream phase including host gaps; not pure active kernel time',wall_seconds=time.perf_counter()-wall,process_cpu_seconds=time.process_time()-cpu_before)
        self.next_step+=1
        return receipt,payload


class LocalTeacherClient:
    """No CUDA in Ray driver; pure own-socket parallel dispatch and reconstruction."""
    def __init__(self,path,teacher_path,required_kind='TRAINED_TEACHER_FROZEN',student_tokenizer=None):
        self.path=Path(path);self.manifest=json.loads(self.path.read_text())
        require(self.manifest['kind']==required_kind and self.manifest['backend']=='local_gpu_fp32_parallel','Wrong teacher backend/kind')
        require(Path(self.manifest['teacher_path']).resolve()==Path(teacher_path).resolve(),'IPC teacher provenance differs')
        if required_kind=='TRAINED_TEACHER_FROZEN':
            require(student_tokenizer is not None and tokenizer_identity(student_tokenizer)==self.manifest['student_tokenizer'],'Driver/scorer tokenizer padding/vocabulary differs')
            require(self.manifest['execution_device']=='cuda:0' and self.manifest['rows_per_service']==320
                    and self.manifest['source_sha256']==implementation_fingerprint(), 'Production GPU/source/320row contract differs')
            require(all(os.environ.get('SEARCH_R1_TOPD_'+name) for name in ('AUDIT_DIR','RUN_ID','COMMIT','CONFIG_SHA256','DATA_IDENTITY')),'Production IPC needs complete audit identity')
        if required_kind=='TRAINED_TEACHER_FROZEN':
            require(len({service['pid'] for service in self.manifest['services']})==8,'Need eight distinct actual scorer PIDs')
        services=self.manifest['services'];require(len(services)==8 and len({s['uuid'] for s in services})==8,'Need eight distinct scorer UUIDs')
        root=self.path.parent.resolve()
        require(all(Path(s['socket']).resolve().is_relative_to(root) for s in services),'Nonlocal teacher socket path')
        for key,env in [('run_id','RUN_ID'),('code_commit','COMMIT'),('config_sha256','CONFIG_SHA256'),('data_identity','DATA_IDENTITY')]:
            value=os.environ.get('SEARCH_R1_TOPD_'+env)
            if value is not None:require(value==self.manifest[key],'IPC manifest differs from driver identity')
        self.scored_tokens=0;self.parameter_count=None;self.last_receipts=[]

    def _one(self,rank,step,prompts,responses,attention,temperature,chunk):
        service=self.manifest['services'][rank];rows=self.manifest['rows_per_service']
        indices=torch.arange(rank*rows,(rank+1)*rows)
        raw=tensor_save(dict(prompts=prompts.contiguous().cpu(),responses=responses.contiguous().cpu(),attention=attention.contiguous().cpu(),row_indices=indices))
        metadata={key:self.manifest[key] for key in IDENTITY}
        metadata.update(capability=service['capability'],outer_step=step,temperature=temperature,chunk=chunk,input_sha256=digest(raw))
        with socket.socket(socket.AF_UNIX) as connection:
            connection.settimeout(self.manifest.get('timeout_seconds',1800));connection.connect(service['socket'])
            if self.manifest['kind']!='CPU_FIXTURE_NO_ALGORITHM':
                require(own_peer(connection,self.manifest['cgroup'])==service['pid'],'Actual Unix peer PID differs from scorer manifest')
            send(connection,metadata,raw);receipt,payload=receive(connection)
        require(receipt.get('status')=='RELEASED','Scorer failed/no release; actor update forbidden')
        require(all(receipt.get(key)==self.manifest[key] for key in IDENTITY),'Scorer output identity differs')
        require(receipt['pid']==service['pid'] and receipt['cgroup']==self.manifest['cgroup'],'Scorer actual PID/cgroup differs')
        require(receipt['rank']==rank and receipt['uuid']==service['uuid'] and receipt['outer_step']==step and receipt['rows']==rows
                and receipt['input_sha256']==digest(raw) and receipt['output_sha256']==digest(payload),'Scorer output binding differs')
        require(receipt['allocator_released']==dict(allocated_bytes=0,reserved_bytes=0),'Scorer retained CUDA workspace')
        initial=receipt['loaded_cpu'];released=receipt['released_cpu']
        for state in (initial,released,receipt['gpu_before'],receipt['gpu_after']):
            if state is not None:
                require(state['parameter_sha256']==initial['parameter_sha256'] and state['dtypes']==['torch.float32']
                        and not state['training'] and state['training_modules_count']==state['requires_grad_parameters']==state['grad_parameters']==0,'Scorer actual frozen/eval/FP32 state differs')
        require(all(device=='cpu' for device in released['buffer_devices']),'Scorer buffers retained CUDA')
        require(initial['devices']==released['devices']==['cpu'],'Scorer model not on CPU after release')
        if step==1:require(receipt['gpu_before'] is not None and receipt['gpu_before']['devices']==[self.manifest['execution_device']],'Missing first scoring device hash')
        if step==self.manifest['outer_updates']:require(receipt['gpu_after'] is not None and receipt['gpu_after']['devices']==[self.manifest['execution_device']],'Missing last scoring device hash')
        require(receipt['targets_none'] and receipt['dtype']=='float32' and receipt['attention']=='eager' and receipt['tf32'] is False,'Scorer math/privacy differs')
        result=tensor_load(payload)
        require(set(result)=={'teacher_log_probs','row_indices'} and torch.equal(result['row_indices'],indices)
                and result['teacher_log_probs'].shape==responses.shape and result['teacher_log_probs'].dtype==torch.float32
                and bool(torch.isfinite(result['teacher_log_probs']).all()),'Scorer output order/shape/finite differs')
        require(receipt['row_indices_sha256']==digest(tensor_save(dict(row_indices=indices))),'Scorer row-index receipt differs')
        require(bool((result['teacher_log_probs'][attention==0]==0).all()),'Scorer padded response logp differs')
        pad=self.manifest['student_tokenizer']['pad_token_id']
        expected_tokens=int((prompts!=pad).sum()+attention.sum())
        require(receipt['scored_tokens_delta']==expected_tokens,'Scorer actual scored-token count differs')
        return receipt,result['teacher_log_probs']

    def score(self,prompts,responses,attention,temperature,chunk,step):
        require(prompts.device.type==responses.device.type==attention.device.type=='cpu','Driver IPC inputs must remain CPU')
        rows=self.manifest['rows_per_service'];require(len(prompts)==len(responses)==len(attention)==8*rows,'Teacher rollout cardinality differs')
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures=[pool.submit(self._one,rank,step,prompts[rank*rows:(rank+1)*rows],responses[rank*rows:(rank+1)*rows],attention[rank*rows:(rank+1)*rows],temperature,chunk) for rank in range(8)]
            values=[future.result() for future in futures]  # All eight release before return.
        self.last_receipts=[value[0] for value in values]
        hashes={r['released_cpu']['parameter_sha256'] for r in self.last_receipts}
        require(len(hashes)==1,'Teacher replicas differ')
        self.parameter_count=self.last_receipts[0]['released_cpu']['parameters']
        self.scored_tokens+=sum(r['scored_tokens_delta'] for r in self.last_receipts)
        return torch.cat([value[1] for value in values])


def serve(session,path,peer_check,receipt_path,persistent=True):
    path=Path(path);require(not path.exists(),'Refuse existing own teacher socket')
    with socket.socket(socket.AF_UNIX) as listener:
        listener.bind(str(path));os.chmod(path,0o600);listener.listen(8)
        try:
            requests=0
            while persistent or requests<session.manifest['outer_updates']:
                requests+=1
                with listener.accept()[0] as connection:
                    try:
                        peer_check(connection);metadata,raw=receive(connection)
                        receipt,payload=session.score(metadata,raw)
                        with Path(receipt_path).open('a') as stream:stream.write(json.dumps(receipt,allow_nan=False)+'\n')
                        send(connection,receipt,payload)
                    except Exception as error:
                        send(connection,dict(status='FAILED',error_type=type(error).__name__,error=str(error)),b'')
                        raise
        finally:path.unlink(missing_ok=True)



def tokenizer_identity(tokenizer):
    return dict(vocabulary_sha256=digest(json.dumps(tokenizer.get_vocab(),sort_keys=True).encode()),
                bos_token_id=tokenizer.bos_token_id,eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id)


def implementation_fingerprint():
    root=Path(__file__).parent
    files={name:digest((root/name).read_bytes()) for name in ('topd_local_gpu.py','distillation.py','topd_runtime_audit.py','controller.py')}
    return digest(json.dumps(files,sort_keys=True).encode())



def file_digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):value.update(chunk)
    return value.hexdigest()


def verify_teacher_files(source,manifest):
    if manifest['kind']=='TRAINED_TEACHER_FROZEN':
        require(file_digest(source/'teacher-manifest.json')==manifest['teacher_receipt_sha256'],'Scorer teacher receipt changed')
        report=json.loads((source/'teacher-manifest.json').read_text());files=report['teacher']['files']
        extra={'teacher-manifest.json'}
    else:
        files=manifest['hardware_model_files'];extra=set()
    actual={str(path.relative_to(source)) for path in source.rglob('*') if path.is_file()}
    require(actual==set(files)|extra and files,'Scorer HF file inventory differs')
    for relative,expected in files.items():
        path=(source/relative).resolve()
        require(path.is_relative_to(source.resolve()) and file_digest(path)==expected,'Scorer actual HF file SHA differs')
    return digest(json.dumps(files,sort_keys=True).encode())


def service_main():
    import argparse
    from transformers import AutoTokenizer
    from .distillation import FrozenTeacher
    from .preflight import validate_inherited_gpu_leases
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True);parser.add_argument('--rank',required=True,type=int)
    args=parser.parse_args();path=Path(args.manifest);manifest=json.loads(path.read_text())
    require(manifest['kind'] in ('TRAINED_TEACHER_FROZEN','HARDWARE_SURROGATE_NO_TRAINING'),'Service cannot run fixture as production')
    require(manifest['source_sha256']==implementation_fingerprint(),'Actual scorer source differs')
    require(os.environ.get('HF_HUB_OFFLINE')=='1' and os.environ.get('TRANSFORMERS_OFFLINE')=='1','Scorer must be offline, without remote model fetching')
    require(0<=args.rank<8 and len(manifest['services'])==8,'Scorer rank/count differs')
    own=manifest['services'][args.rank]
    require(os.environ.get('CUDA_VISIBLE_DEVICES')==own['uuid'],'Scorer must use its exact leased UUID')
    require(os.environ.get('SLURM_JOB_ID')==manifest['slurm_job'],'Scorer Slurm job differs')
    rows=Path('/proc/self/cgroup').read_text().splitlines()
    require(any(row.split(':',2)[-1]==manifest['cgroup'] or row.split(':',2)[-1].startswith(manifest['cgroup']+'/') for row in rows),'Scorer outside own cgroup')
    validate_inherited_gpu_leases(1)  # Initial idle15%; never called during teacher residency.
    records=json.loads(os.environ['SEARCH_R1_GPU_LEASES']);require(len(records)==1 and records[0]['uuid']==own['uuid'],'Scorer inherited lease differs')
    source=Path(manifest['teacher_path']);root=Path(os.environ['AGENT_ROOT']).resolve()
    require(source.resolve().is_relative_to(root) and path.resolve().is_relative_to(root) and Path(manifest['student_path']).resolve().is_relative_to(root),'Scorer assets/socket manifest must be own local')
    actual_hf_sha=verify_teacher_files(source,manifest)
    torch.set_num_threads(1)
    from verl.utils import hf_tokenizer
    tokenizer=hf_tokenizer(manifest['student_path'],local_files_only=True,trust_remote_code=False)
    require(tokenizer_identity(tokenizer)==manifest['student_tokenizer'],'Actual service tokenizer differs from driver contract')
    teacher=FrozenTeacher(str(source),tokenizer,'cpu')
    def active_ownership():
        lease_ownership(records[0])
        require(os.environ.get('CUDA_VISIBLE_DEVICES')==own['uuid'],'Active scorer visible UUID differs')
        rows=Path('/proc/self/cgroup').read_text().splitlines()
        require(any(row.split(':',2)[-1]==manifest['cgroup'] or row.split(':',2)[-1].startswith(manifest['cgroup']+'/') for row in rows),'Active scorer left own cgroup')
    session=ScorerSession(teacher,manifest,args.rank,own['uuid'],active_ownership)
    ready={key:manifest[key] for key in IDENTITY}
    ready.update(rank=args.rank,uuid=own['uuid'],pid=os.getpid(),cgroup=manifest['cgroup'],loaded_cpu=session.initial,actual_hf_files_sha256=actual_hf_sha)
    (path.parent/f'scorer-ready-{args.rank}.json').write_text(json.dumps(ready,allow_nan=False)+'\n')
    serve(session,own['socket'],lambda connection:own_peer(connection,manifest['cgroup']),path.parent/f'scorer-{args.rank}.jsonl')


if __name__=='__main__':service_main()
