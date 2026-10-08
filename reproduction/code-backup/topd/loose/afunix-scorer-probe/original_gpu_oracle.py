"""Prospective original FrozenTeacher GPU oracle; hardware surrogate, never RL.

A distinct single-UUID leased process. No ScorerSession or LocalTeacherClient.
CPU-ready before other own GPU models; score only after eight scorers RELEASED0.
"""
import argparse
import hashlib
import json
import inspect
import os
from pathlib import Path
import socket
import subprocess
import time

IDENTITY=('run_id','code_commit','config_sha256','data_identity','teacher_receipt_sha256','source_sha256')
SCOPE='ORIGINAL_FROZENTEACHER_GPU_FUNCTION_ORACLE_NO_TRAINING'


def require(condition,message):
    if not condition:raise RuntimeError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def atomic_json(path,value):
    path=Path(path);require(not path.exists(),'Refuse existing oracle result')
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('x') as f:json.dump(value,f,sort_keys=True,allow_nan=False);f.write('\n')
    os.replace(temporary,path)


def validate_input(torch,tensors,step,pad):
    require(set(tensors)=={'prompts','responses','attention','row_indices'},'Oracle accepts token inputs only')
    p,r,a,i=(tensors[k] for k in ('prompts','responses','attention','row_indices'))
    width=(32,500,4096)[step-1];prompt_width=2048 if step==3 else 256
    require(p.shape==(16,prompt_width) and r.shape==a.shape==(16,width),'Original16-row shapes differ')
    require(all(t.dtype==torch.long and t.device.type=='cpu' for t in tensors.values()),'Oracle input dtype/device differs')
    require(torch.equal(i,torch.arange(16)),'Oracle global row order differs')
    require(bool(((a==0)|(a==1)).all()),'Oracle attention is not binary')
    if step<3:require(bool((a==1).all()) and bool((p!=pad).all()),'Short original input mask differs')
    else:
        require(int((p[15]!=pad).sum())==2048 and int(a[15].sum())==4096,'Longest original row was masked/truncated')
        for row in range(15):
            require(int((p[row]!=pad).sum())==256 and int(a[row].sum())==(32 if row%2==0 else 500),'Long-round control rows differ')
    return int((p!=pad).sum()+a.sum())


def original_score(torch,teacher,tensors,row_tokens=None):
    # Exactly the original one-row function, same chunk/temp/targets; no backend wrapper.
    p,r,a=(tensors[k] for k in ('prompts','responses','attention'))
    outputs=[]
    for i in range(16):
        before=teacher.scored_tokens
        outputs.append(teacher.log_probs(p[i:i+1],r[i:i+1],a[i:i+1],temperature=1.,targets=None,chunk=16))
        if row_tokens is not None:row_tokens.append(teacher.scored_tokens-before)
    output=torch.cat(outputs)
    require(output.shape==r.shape and output.dtype==torch.float32,'Original GPU output shape/dtype differs')
    require(bool(torch.isfinite(output).all()) and bool((output[a==0]==0).all()),'Original GPU output not finite/maskedzero')
    return output


def actual_flags(torch):
    return dict(matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                matmul_precision=torch.get_float32_matmul_precision(),
                autocast_enabled=torch.is_autocast_enabled(),cpu_threads=torch.get_num_threads(),
                environment={k:os.environ.get(k) for k in ('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE',
                    'NVIDIA_TF32_OVERRIDE','CUBLAS_WORKSPACE_CONFIG','CUDA_VISIBLE_DEVICES')})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--directory',type=Path,required=True);args=parser.parse_args()
    root=Path(os.environ['AGENT_ROOT']).resolve();directory=args.directory.resolve();manifest_path=args.manifest.resolve()
    require(root==Path('/dataset/shaoshitong/search-r1-rl/topd') and directory.is_relative_to(root/'logs')
            and manifest_path.is_relative_to(root/'logs'),'Oracle assets/control must be own local')
    require(socket.gethostname().split('.')[0]=='h100-11' and os.environ.get('SLURM_JOB_NODELIST')=='h100-11','Oracle must stay on own node')
    manifest=json.loads(manifest_path.read_text());own=manifest['services'][4]
    require(manifest['kind']=='HARDWARE_SURROGATE_NO_TRAINING' and manifest['outer_updates']==3,'Oracle hardware scope differs')
    require(os.environ.get('SLURM_JOB_ID')==manifest['slurm_job'],'Oracle job differs')
    def own_scope():
        paths=[line.split(':',2)[-1] for line in Path('/proc/self/cgroup').read_text().splitlines()]
        require(any(path==manifest['cgroup'] or path.startswith(manifest['cgroup']+'/') for path in paths),'Oracle outside own cgroup')
    own_scope()
    require(os.environ.get('CUDA_VISIBLE_DEVICES')==own['uuid'],'Oracle requires original rank4 UUID')
    records=json.loads(os.environ['SEARCH_R1_GPU_LEASES'])
    require(len(records)==1 and records[0]['uuid']==own['uuid'],'Oracle requires only own single inherited lease')
    require(os.environ.get('HF_HUB_OFFLINE')=='1' and os.environ.get('TRANSFORMERS_OFFLINE')=='1','Oracle must be offline')
    from search_r1.baselines.preflight import validate_inherited_gpu_leases
    validate_inherited_gpu_leases(1)  # Once at CPU-ready startup, before own model residency.
    import torch
    from safetensors.torch import load_file,save_file,save
    from search_r1.baselines.distillation import FrozenTeacher
    from search_r1.baselines.topd_local_gpu import (implementation_fingerprint,verify_teacher_files,
        tokenizer_identity,lease_ownership,snapshot,initialize_cuda_context,cuda_process_snapshot,release_cuda)
    from verl.utils import hf_tokenizer
    require(not torch.cuda.is_initialized(),'Oracle CUDA context existed before CPU-ready')
    require(torch.__version__.split('+')[0]=='2.4.0','Oracle requires pinned torch2.4.0 release')
    require(implementation_fingerprint()==manifest['source_sha256'],'Oracle actual source differs')
    commit=subprocess.check_output(['git','-C',str(root/'repo'),'rev-parse','HEAD'],text=True).strip()
    require(commit==manifest['code_commit'],'Oracle actual code commit differs')
    source=Path(manifest['teacher_path']).resolve();require(source.is_relative_to(root),'Oracle HF must be copied locally')
    hf_identity=verify_teacher_files(source,manifest)
    torch.set_num_threads(1)
    tokenizer=hf_tokenizer(str(source),local_files_only=True,trust_remote_code=False)
    require(tokenizer_identity(tokenizer)==manifest['student_tokenizer'],'Oracle tokenizer identity differs')
    teacher=FrozenTeacher(str(source),tokenizer,'cpu');initial=snapshot(teacher)
    rank4_ready=json.loads((manifest_path.parent/'scorer-ready-4.json').read_text())
    require(initial==rank4_ready['loaded_cpu'],'Original oracle teacher differs from scorer frozen model')
    identity={key:manifest[key] for key in IDENTITY}
    common=dict(identity,scope=SCOPE,pid=os.getpid(),uuid=own['uuid'],rank=4,cgroup=manifest['cgroup'],
        slurm_job=manifest['slurm_job'],host=socket.gethostname(),source_kind='HARDWARE_SURROGATE_NO_TRAINING',
        manifest_sha256=sha(manifest_path),oracle_script_sha256=sha(__file__),torch_version=torch.__version__,
        actual_hf_files_sha256=hf_identity,offline=True,loaded_cpu=initial,
        original_function='search_r1.baselines.distillation.FrozenTeacher.log_probs',
        original_function_source_sha256=hashlib.sha256(inspect.getsource(FrozenTeacher.log_probs).encode()).hexdigest(),
        original_function_file_sha256=sha(inspect.getfile(FrozenTeacher)))
    atomic_json(directory/'ready.json',dict(common,status='CPU_READY_NO_CUDA_CONTEXT',cuda_initialized=torch.cuda.is_initialized(),ready_epoch=time.time()))
    for step in range(1,4):
        request_path=directory/f'request-round{step}.json';deadline=time.monotonic()+1800
        while not request_path.exists():
            require(time.monotonic()<deadline,'Oracle request timeout');time.sleep(.2)
        request=json.loads(request_path.read_text())
        require(set(request)==set(IDENTITY)|{'outer_step','input_sha256','all8_released_zero','candidate_receipts_sha256','scorers_released_epoch'},'Oracle request fields differ')
        require(all(request[key]==manifest[key] for key in IDENTITY) and request['outer_step']==step,'Oracle request identity/order differs')
        require(request['all8_released_zero'] is True,'Oracle cannot overlap eight service scoring')
        candidate_path=directory.parent/f'receipts-round{step}.json'
        require(sha(candidate_path)==request['candidate_receipts_sha256'],'Candidate receipt artifact changed')
        candidates=json.loads(candidate_path.read_text());require(len(candidates)==8,'Need eight candidate receipts')
        # Inspect actual worker receipts independently, rather than trusting request boolean.
        for rank in range(8):
            receipt=json.loads((manifest_path.parent/f'scorer-{rank}.jsonl').read_text().splitlines()[-1])
            service=manifest['services'][rank]
            require(receipt==candidates[rank] and (receipt['rank'],receipt['pid'],receipt['uuid'],receipt['cgroup'])
                    ==(rank,service['pid'],service['uuid'],manifest['cgroup']),'Actual candidate rank/PID/UUID/cgroup differs')
            require(receipt['rows']==2 and receipt['row_indices_sha256']==hashlib.sha256(save({'row_indices':torch.arange(rank*2,(rank+1)*2)})).hexdigest(),'Candidate actual row count/order differs')
            require(receipt['targets_none'] and receipt['dtype']=='float32' and receipt['attention']=='eager' and receipt['tf32'] is False,'Candidate math/privacy differs')
            require(receipt['outer_step']==step and receipt['status']=='RELEASED'
                    and receipt['allocator_released']=={'allocated_bytes':0,'reserved_bytes':0}
                    and all(receipt[key]==manifest[key] for key in IDENTITY),'Scorer not actually released before oracle')
        input_path=directory/f'input-round{step}.safetensors'
        require(sha(input_path)==request['input_sha256'],'Oracle exact input bytes differ')
        tensors=load_file(str(input_path),device='cpu');expected_tokens=validate_input(torch,tensors,step,tokenizer.pad_token_id)
        record=dict(common,outer_step=step,input_sha256=sha(input_path),rows=16,row_indices=list(range(16)),
                    temperature=1.,chunk=16,targets_none=True,dtype='float32',attention='eager',status='FAILED',
                    expected_scored_tokens=expected_tokens,request_sha256=sha(request_path),
                    row_indices_sha256=hashlib.sha256(save({'row_indices':tensors['row_indices']})).hexdigest(),
                    attention_sha256=hashlib.sha256(save({'attention':tensors['attention']})).hexdigest(),
                    candidate_receipts_sha256=sha(candidate_path),scorers_released_epoch=request['scorers_released_epoch'])
        started=time.monotonic();cpu_start=time.process_time();before_tokens=teacher.scored_tokens
        try:
            own_scope();lease_ownership(records[0])
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            torch.set_float32_matmul_precision('highest');record['actual_flags']=actual_flags(torch)
            require(not record['actual_flags']['autocast_enabled'] and not record['actual_flags']['matmul_allow_tf32']
                    and record['actual_flags']['matmul_precision']=='highest'
                    and os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')=='0'
                    and os.environ.get('NVIDIA_TF32_OVERRIDE') in (None,'0'),'Oracle precision override differs')
            record['cuda_started_epoch']=time.time()
            require(record['scorers_released_epoch']<=record['cuda_started_epoch'],'Oracle GPU phase preceded candidate release')
            initialize_cuda_context('cuda:0');record['gpu_pid_before']=cuda_process_snapshot(own['uuid'])
            teacher.model.to('cuda:0');teacher.device=torch.device('cuda:0')
            record['gpu_before']=snapshot(teacher)
            score_start=time.monotonic();record['scoring_started_epoch']=time.time();row_tokens=[]
            output=original_score(torch,teacher,tensors,row_tokens);torch.cuda.synchronize()
            record['scoring_completed_epoch']=time.time();record['scored_tokens_by_row']=row_tokens
            expected_by_row=[int((tensors['prompts'][i]!=tokenizer.pad_token_id).sum()+tensors['attention'][i].sum()) for i in range(16)]
            require(row_tokens==expected_by_row,'Original per-row scored-token count differs')
            record['scoring_wall_seconds']=time.monotonic()-score_start
            record['gpu_after']=snapshot(teacher);record['gpu_pid_phase_sample']=cuda_process_snapshot(own['uuid'])
            record['allocator_peak']=dict(allocated_bytes=torch.cuda.max_memory_allocated(),reserved_bytes=torch.cuda.max_memory_reserved())
            record['scored_tokens_delta']=teacher.scored_tokens-before_tokens
            require(record['scored_tokens_delta']==expected_tokens,'Oracle original scored-token count differs')
            require(record['gpu_before']['parameter_sha256']==record['gpu_after']['parameter_sha256']==initial['parameter_sha256'],'Oracle GPU frozen parameter hash changed')
            output_path=directory/f'reference-round{step}.safetensors';require(not output_path.exists(),'Refuse original oracle output overwrite')
            temporary=output_path.with_name(output_path.name+'.tmp');require(not temporary.exists(),'Oracle output temp exists')
            save_file({'original_gpu_log_probs':output.contiguous()},str(temporary));os.replace(temporary,output_path)
            record['output_sha256']=sha(output_path);record['finite']=True;record['masked_zero_verified']=True
        except BaseException as error:
            record['error']=repr(error);raise
        finally:
            try:
                record['allocator_released']=release_cuda(teacher);record['released_cpu']=snapshot(teacher)
                require(record['allocator_released']=={'allocated_bytes':0,'reserved_bytes':0},'Original oracle did not release completely')
                require(record['released_cpu']==initial,'Original oracle teacher did not return to unchanged CPU state')
                record['gpu_pid_released']=cuda_process_snapshot(own['uuid']);record['completed_epoch']=time.time()
                if 'error' not in record:record['status']='RELEASED'
            except BaseException as error:
                record['release_error']=repr(error)
            record['wall_seconds']=time.monotonic()-started;record['process_cpu_seconds']=time.process_time()-cpu_start
            atomic_json(directory/f'receipt-round{step}.json',record)
        require(record['status']=='RELEASED' and 'release_error' not in record,'Original oracle release failed')
    atomic_json(directory/'done.json',dict(common,status='THREE_ORIGINAL_GPU_PHASES_RELEASED_NO_TRAINING',rounds=3,done_epoch=time.time(),last_receipt_sha256=sha(directory/'receipt-round3.json')))


if __name__=='__main__':main()
