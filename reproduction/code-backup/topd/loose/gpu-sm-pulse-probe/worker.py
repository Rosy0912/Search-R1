"""Bounded leased single-UUID CUDA pulse; no model, retrieval or training."""
import json
import os
from pathlib import Path
import sys
import time

def write(path, value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');os.replace(temp,path)

def main():
    import torch
    from search_r1.baselines.topd_local_gpu import lease_ownership, cuda_process_snapshot
    folder=Path(sys.argv[1]);rank=int(sys.argv[2]);lease=json.loads(os.environ['SEARCH_R1_GPU_LEASES'])
    assert len(lease)==1 and os.environ['CUDA_VISIBLE_DEVICES']==lease[0]['uuid']
    lease_ownership(lease[0])
    from runtime_safety_candidate import pid_identity
    identity=pid_identity(os.getpid());uid=lease[0]['uuid']
    write(folder/f'ready-{rank}.json',dict(identity=identity,uuid=uid,cuda_initialized=torch.cuda.is_initialized(),scope='NO_MODEL_NO_TRAINING'))
    deadline=time.monotonic()+90
    while not (folder/'start').exists():
        if time.monotonic()>deadline:raise RuntimeError('No own start command')
        time.sleep(.05)
    result=dict(identity=identity,uuid=uid,status='FAILED_NO_TRAINING')
    try:
        torch.backends.cuda.matmul.allow_tf32=False
        a=torch.ones((4096,4096),device='cuda',dtype=torch.float32)
        b=torch.ones_like(a);out=torch.empty_like(a)
        torch.mm(a,b,out=out);torch.cuda.synchronize()
        result['pulse_spec']=dict(matrix_shape=[4096,4096],dtype=str(a.dtype),tensor_bytes=(a.numel()+b.numel()+out.numel())*a.element_size(),max_seconds=45)
        result['actual_gpu']=cuda_process_snapshot(uid)
        write(folder/f'busy-{rank}.json',dict(identity=identity,uuid=uid,actual_gpu=result['actual_gpu'],pulse_spec=result['pulse_spec'],started_epoch=time.time()))
        began=time.monotonic()
        while not (folder/'stop').exists():
            if time.monotonic()-began>45:raise RuntimeError('Own pulse exceeded45s')
            torch.mm(a,b,out=out);torch.cuda.synchronize()
        del a,b,out
        result['status']='PULSE_FINISHED_NO_TRAINING'
    except BaseException as exc:
        result['error']=repr(exc)
    finally:
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
            torch._C._cuda_clearCublasWorkspaces()
            torch.cuda.empty_cache()
            result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved())
        else:result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),allocated_bytes=0,reserved_bytes=0)
        result['released_epoch']=time.time()
        write(folder/f'released-{rank}.json',result)
    if result['status']!='PULSE_FINISHED_NO_TRAINING' or result['allocated_bytes'] or result['reserved_bytes']:
        raise RuntimeError('Pulse or own allocator release failed')
    # Keep identity alive while supervisor measures high->idle transition.
    while not (folder/'exit').exists():
        if time.monotonic()>deadline:raise RuntimeError('Missing own exit command')
        time.sleep(.05)

if __name__=='__main__':main()
