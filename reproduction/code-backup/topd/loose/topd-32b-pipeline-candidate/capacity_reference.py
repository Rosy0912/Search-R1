"""External prototype: one leased eight-GPU FP32 teacher, original scoring method.

No production integration, teacher generation, optimizer, or training entry point.
Native deployment and quality acceptance remain required; importing creates no CUDA.
"""
import hashlib
import inspect
import math
import os
from pathlib import Path
import socket
import threading
import time

import torch

ORIGINAL_METHOD_SHA='e23b1cdb73736723ff422a9c041f2678b5e8f04cb9a496bd219ad87827b5ef7d'
PARAMETERS_32B=32763876352
LAYER_PARAMETERS=487605248
VOCAB_32B=152064
SCOPE='SHARED_8GPU_TEACHER_PROTOTYPE_NOT_PRODUCTION_READY'


def require(condition,message):
    if not condition:raise RuntimeError(message)


def method_sha(method):return hashlib.sha256(inspect.getsource(method).encode()).hexdigest()


def device_map_32b():
    mapping={name:'cuda:0' for name in ('model.embed_tokens','model.rotary_emb','model.norm','lm_head')}
    offset=0
    for device,count in enumerate((6,9,9,8,8,8,8,8)):
        for layer in range(offset,offset+count):mapping[f'model.layers.{layer}']=f'cuda:{device}'
        offset+=count
    return mapping


def parameter_map(model,mapping):
    """Complete deterministic name coverage; never move an unaccounted tensor."""
    actual={}
    for kind,tensors in [('parameter',model.named_parameters()),('buffer',model.named_buffers())]:
        for name,value in tensors:
            matches=[key for key in mapping if name==key or name.startswith(key+'.')]
            require(matches,'Unmapped '+kind+': '+name)
            target=mapping[max(matches,key=len)]
            require(target in {f'cuda:{i}' for i in range(8)},'Foreign map device')
            actual[name]=dict(kind=kind,device=target,bytes=value.numel()*value.element_size())
    return actual


def snapshot(model):
    parameters=hashlib.sha256();buffers=hashlib.sha256()
    for destination,tensors in [(parameters,model.named_parameters()),(buffers,model.named_buffers())]:
        for name,tensor in tensors:
            require(tensor.dtype==torch.float32,'Non-FP32 model tensor: '+name)
            destination.update(name.encode()+b'\0');destination.update(str(tuple(tensor.shape)).encode()+b'\0')
            destination.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return dict(parameter_sha256=parameters.hexdigest(),buffer_sha256=buffers.hexdigest(),
                parameter_count=sum(p.numel() for p in model.parameters()),
                parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
                parameter_devices=sorted({str(p.device) for p in model.parameters()}),
                buffer_devices=sorted({str(b.device) for b in model.buffers()}),
                training_modules=sum(int(m.training) for m in model.modules()),
                requires_grad_parameters=sum(int(p.requires_grad) for p in model.parameters()),
                grad_parameters=sum(int(p.grad is not None) for p in model.parameters()))


def frozen_equal(left,right):
    return all(left[key]==right[key] for key in ('parameter_sha256','buffer_sha256','parameter_count','parameter_bytes'))


def quality_gate(receipt):
    require(receipt['status']=='PASS' and receipt['teacher_kind']=='QWEN2_5_32B_BASE_QUALITY_TEST'
            and receipt['questions']==64 and receipt['correct']>=26 and receipt['correct']<=64
            and receipt['greedy'] is True and receipt['pass_at_k']==1 and receipt['metric']=='EM'
            and receipt['teacher_model']=='Qwen/Qwen2.5-32B','32B same64 quality gate incomplete/failed')
    for key in ('question_sha256','teacher_weights_inventory_sha256','student_comparison_sha256'):
        require(len(receipt[key])==64 and all(c in '0123456789abcdef' for c in receipt[key]),'Missing quality identity')
    return receipt['teacher_weights_inventory_sha256']


class NativeCudaRuntime:
    device='cuda:0'
    kind='NATIVE_UNVALIDATED_PROTOTYPE'
    """All CUDA operations and lease validation are explicit, own-process only."""
    def __init__(self,records,scope,job,root):
        self.records=records;self.scope=scope;self.job=str(job);self.root=Path(root).resolve()
        self.uuids=[record['uuid'] for record in records]
        require(len(records)==len(set(self.uuids))==8,'Need eight distinct inherited UUID leases')
        require(os.environ.get('CUDA_VISIBLE_DEVICES')==','.join(self.uuids),'Visible UUID order differs')
        self.verify()

    def verify(self):
        from search_r1.baselines.topd_local_gpu import lease_ownership
        import transformers
        require(torch.__version__=='2.4.0+cu121' and transformers.__version__=='4.46.3','Unreviewed native Torch/Transformers')
        require(self.root==Path('/dataset/shaoshitong/search-r1-rl/topd'),'Foreign ROOT')
        require(socket.gethostname().split('.')[0]=='h100-11' and os.environ.get('SLURM_JOB_NODELIST')=='h100-11','Foreign node')
        require(os.environ.get('SLURM_JOB_ID')==self.job,'Foreign job')
        require(self.scope.startswith('/') and any(line.split(':',2)[-1]==self.scope or line.split(':',2)[-1].startswith(self.scope+'/') for line in Path('/proc/self/cgroup').read_text().splitlines()),'Foreign cgroup')
        for record in self.records:lease_ownership(record)

    def flags(self):
        require(os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')=='0'
                and os.environ.get('NVIDIA_TF32_OVERRIDE') in (None,'0'),'Environment forces TF32')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest')
        require(not torch.is_autocast_enabled(),'Autocast context forbidden')
        return dict(matmul_allow_tf32=False,cudnn_allow_tf32=False,matmul_precision='highest',autocast_enabled=False)

    def synchronize(self):
        for device in range(8):torch.cuda.synchronize(device)

    def reset_peaks(self):
        for device in range(8):torch.cuda.reset_peak_memory_stats(device)

    def peaks(self):
        return [dict(device=i,uuid=uuid,allocated_bytes=torch.cuda.max_memory_allocated(i),reserved_bytes=torch.cuda.max_memory_reserved(i)) for i,uuid in enumerate(self.uuids)]

    def dispatch(self,model,mapping):
        import accelerate
        from accelerate import dispatch_model
        require(accelerate.__version__=='1.15.0','Unreviewed Accelerate version')
        return dispatch_model(model,mapping,main_device='cuda:0',force_hooks=True,offload_buffers=False)

    def verify_partition(self,model,inventory):
        for name,tensor in list(model.named_parameters())+list(model.named_buffers()):
            require(str(tensor.device)==inventory[name]['device'],'Actual partition device differs: '+name)

    def unhook(self,model):
        from accelerate.hooks import remove_hook_from_module
        remove_hook_from_module(model,recurse=True)
        require(not any(hasattr(module,'_hf_hook') or hasattr(module,'_old_forward') for module in model.modules()),'Dispatch hook remains')
        model.__dict__.pop('hf_device_map',None)

    def release_allocators(self):
        require(hasattr(torch._C,'_cuda_clearCublasWorkspaces'),'Pinned own-process workspace API unavailable')
        torch._C._cuda_clearCublasWorkspaces()
        for device in range(8):
            with torch.cuda.device(device):torch.cuda.empty_cache();torch.cuda.synchronize(device)
        return [dict(device=device,uuid=uuid,allocated_bytes=torch.cuda.memory_allocated(device),reserved_bytes=torch.cuda.memory_reserved(device)) for device,uuid in enumerate(self.uuids)]

    def gpu_identity(self):
        import subprocess
        raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=15)
        values=[]
        for line in raw.splitlines():
            uuid,pid,memory=[part.strip() for part in line.split(',')]
            if int(pid)==os.getpid():
                require(uuid in self.uuids,'Own PID on unleased GPU');value=float(memory)
                require(math.isfinite(value) and value>=0,'Invalid NVML memory')
                values.append(dict(pid=int(pid),uuid=uuid,used_mib=value))
        require({row['uuid'] for row in values}==set(self.uuids),'Missing own PID on one of eight leased GPUs')
        return values


class SharedScorer:
    def __init__(self,teacher,runtime,mapping,allowed_ids,expected_parameters=PARAMETERS_32B):
        from search_r1.baselines.distillation import FrozenTeacher
        require(type(teacher) is FrozenTeacher and method_sha(FrozenTeacher.log_probs)==ORIGINAL_METHOD_SHA,'Original FrozenTeacher method differs')
        self.original=FrozenTeacher.log_probs;self.teacher=teacher;self.runtime=runtime;self.mapping=mapping
        self.allowed_ids=set(allowed_ids);self.lock=threading.Lock();self.state='CPU_READY';self.rounds=0
        self.initial=snapshot(teacher.model);self.inventory=parameter_map(teacher.model,mapping)
        require(self.initial['parameter_count']==expected_parameters and self.initial['parameter_devices']==['cpu']
                and all(d=='cpu' for d in self.initial['buffer_devices']),'Wrong full teacher/CPU state')
        require(self.initial['training_modules']==self.initial['requires_grad_parameters']==self.initial['grad_parameters']==0,'Teacher not frozen eval')
        require(teacher.model.config._attn_implementation=='eager','Eager attention required')

    def validate(self,tensors):
        require(set(tensors)=={'prompts','responses','attention','row_indices'},'Accept only token tensors; no gold')
        p,r,a,i=(tensors[key] for key in ('prompts','responses','attention','row_indices'))
        require(all(t.dtype==torch.long and t.device.type=='cpu' for t in tensors.values()),'CPU int64 inputs required')
        require(p.ndim==r.ndim==2 and r.shape==a.shape and p.shape[0]==r.shape[0]>0
                and p.shape[1]<=2048 and r.shape[1]<=4096,'Unchanged scoring context/row shapes required')
        require(i.shape==(r.shape[0],) and torch.equal(i,torch.arange(r.shape[0])),'Global row order differs')
        require(bool(((a==0)|(a==1)).all()),'Nonbinary mask')
        require(all(int(value) in self.allowed_ids for value in p.flatten())
                and all(int(value) in self.allowed_ids for value in r.flatten()),'Unmapped student token ID')
        require(bool(((p!=self.teacher.tokenizer.pad_token_id).sum(-1)>0).all()),'Empty prompt')
        return p,r,a

    def score_round(self,tensors,batch_size=4):
        require(batch_size in (1,4,8),'Explicit scorer batch must be1/4/8')
        require(self.lock.acquire(blocking=False),'Shared teacher concurrent request forbidden')
        output=None;record=dict(scope=SCOPE,runtime_kind=self.runtime.kind,round=self.rounds+1,original_method_sha256=ORIGINAL_METHOD_SHA,original_file_sha256=hashlib.sha256(Path(inspect.getfile(self.original)).read_bytes()).hexdigest(),initial_cpu=self.initial,status='FAILED')
        try:
            require(self.state in ('CPU_READY','RELEASED'),'Teacher not released')
            from safetensors.torch import save
            p,r,a=self.validate(tensors);record['input_sha256']=hashlib.sha256(save(tensors)).hexdigest();self.runtime.verify();record['actual_flags']=self.runtime.flags()
            self.state='STAGING';record['started_epoch']=time.time();self.runtime.reset_peaks()
            self.teacher.model=self.runtime.dispatch(self.teacher.model,self.mapping);self.teacher.device=torch.device(self.runtime.device)
            self.runtime.synchronize();record['staged_epoch']=time.time()
            staged=snapshot(self.teacher.model);record['staged_frozen']=staged;require(staged['training_modules']==staged['requires_grad_parameters']==staged['grad_parameters']==0 and frozen_equal(staged,self.initial),'Staging changed frozen FP32 tensors')
            self.runtime.verify_partition(self.teacher.model,self.inventory)
            record['gpu_pid_staged']=self.runtime.gpu_identity();self.state='SCORING';before=self.teacher.scored_tokens;rows=[]
            for start in range(0,r.shape[0],batch_size):
                stop=min(start+batch_size,r.shape[0])
                rows.append(self.original(self.teacher,p[start:stop],r[start:stop],a[start:stop],temperature=1.,targets=None,chunk=16))
            output=torch.cat(rows);record['scored_epoch']=time.time()
            expected=int((p!=self.teacher.tokenizer.pad_token_id).sum()+a.sum())
            require(self.teacher.scored_tokens-before==expected,'Scored token count differs')
            require(output.shape==r.shape and output.dtype==torch.float32 and output.device.type=='cpu'
                    and bool(torch.isfinite(output).all()) and bool((output[a==0]==0).all()),'Output finite/mask/row shape differs')
            record.update(scored_tokens=expected,rows=r.shape[0],batch_size=batch_size,row_order_verified=True,masked_zero_verified=True)
            record['gpu_pid_scored']=self.runtime.gpu_identity();record['allocator_peaks']=self.runtime.peaks()
            record['output_sha256']=hashlib.sha256(save(dict(teacher_log_probs=output,row_indices=tensors['row_indices']))).hexdigest()
        except Exception as error:
            record['error']=repr(error)
        finally:
            if self.state in ('STAGING','SCORING'):
                try:
                    self.runtime.synchronize();self.runtime.unhook(self.teacher.model)
                    self.teacher.model.to('cpu');self.teacher.device=torch.device('cpu');self.runtime.synchronize()
                    released=snapshot(self.teacher.model)
                    require(released['training_modules']==released['requires_grad_parameters']==released['grad_parameters']==0 and frozen_equal(released,self.initial) and released['parameter_devices']==['cpu']
                            and all(d=='cpu' for d in released['buffer_devices']),'CPU roundtrip changed tensors or retained GPU')
                    record['released_cpu']=released;record['allocators_released']=self.runtime.release_allocators()
                    require(len(record['allocators_released'])==8 and all(row['allocated_bytes']==row['reserved_bytes']==0 for row in record['allocators_released']),'One of eight GPU allocators not zero')
                    record['gpu_pid_released']=self.runtime.gpu_identity();record['released_epoch']=time.time();self.state='RELEASED'
                except Exception as error:record['release_error']=repr(error);self.state='FAILED'
            self.lock.release()
        if 'error' not in record and 'release_error' not in record and self.state=='RELEASED':
            record['status']='RELEASED';self.rounds+=1;return output,record
        raise RuntimeError(record)


def load_cpu_teacher(path,student_tokenizer,quality_receipt,expected_inventory_sha256):
    """Explicit construction only after reviewed64QA identity, never at import."""
    import json
    from search_r1.baselines.distillation import FrozenTeacher
    require(quality_gate(quality_receipt)==expected_inventory_sha256,'Teacher quality/weight identity differs')
    path=Path(path).resolve();root=Path('/dataset/shaoshitong/search-r1-rl/topd')
    require(path.is_relative_to(root),'Teacher weights must be own local')
    files={}
    for item in sorted(path.rglob('*')):
        if item.is_file():
            h=hashlib.sha256()
            with item.open('rb') as stream:
                for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
            files[str(item.relative_to(path))]=h.hexdigest()
    require(hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()==expected_inventory_sha256,'Actual HF files differ from64QA weights')
    require(not torch.cuda.is_initialized(),'CPU teacher startup must precede CUDA initialization')
    teacher=FrozenTeacher(str(path),student_tokenizer,device='cpu')
    require(teacher.parameter_count==PARAMETERS_32B and teacher.model.config.vocab_size==VOCAB_32B,'Wrong32B teacher dimensions')
    require(not torch.cuda.is_initialized(),'CPU-ready teacher initialized CUDA')
    return teacher
