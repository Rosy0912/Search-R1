"""One actual owner PID on eight leased GPUs; no model or inference."""
import json,os,sys,time,threading,subprocess
from pathlib import Path

def write(path,row):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(row,indent=2)+'\n');os.replace(temp,path)

def main():
    import torch
    from search_r1.baselines.topd_local_gpu import lease_ownership
    folder=Path(sys.argv[1]);leases=json.loads(os.environ['SEARCH_R1_GPU_LEASES'])
    assert len(leases)==8 and os.environ['CUDA_VISIBLE_DEVICES']==','.join(x['uuid'] for x in leases)
    for lease in leases:lease_ownership(lease)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest');torch.set_num_threads(1)
    rows=[]
    for i,lease in enumerate(leases):
        with torch.cuda.device(i):
            tiny=torch.empty((),device='cuda');torch.cuda.synchronize();del tiny
            torch._C._cuda_clearCublasWorkspaces();torch.cuda.empty_cache()
            raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=15)
            actual=[dict(uuid=parts[0].strip(),pid=int(parts[1]),used_mib=float(parts[2])) for line in raw.splitlines() if len(parts:=line.split(','))==3 and parts[1].strip()==str(os.getpid())]
            assert {p['uuid'] for p in actual}=={x['uuid'] for x in leases[:i+1]}, 'Actual CUDA ordinal/UUID mapping differs'
            rows.append(dict(device=i,actual_gpu_pid_uuids=actual,uuid=lease['uuid'],allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved()))
    assert all(r['allocated']==r['reserved']==0 for r in rows)
    write(folder/'ready.json',dict(pid=os.getpid(),torch_version=torch.__version__,cuda_version=torch.version.cuda,devices=rows,scope='TOY_TELEMETRY_ONLY_NO_MODEL'))
    began=time.monotonic()
    for phase,duration in enumerate((16.,.25,2.),1):
        while not (folder/f'command-{phase}').exists():
            if time.monotonic()-began>420:raise RuntimeError('Own command timeout')
            time.sleep(.03)
        errors=[]
        barrier=threading.Barrier(8)
        def pulse(i):
            a=b=out=None
            try:
                with torch.cuda.device(i):
                    a=torch.ones((4096,4096),device=f'cuda:{i}',dtype=torch.float32)
                    b=torch.ones_like(a);out=torch.empty_like(a);barrier.wait(timeout=30)
                    start=time.monotonic()
                    while time.monotonic()-start<duration:torch.mm(a,b,out=out);torch.cuda.synchronize(i)
            except BaseException as exc:errors.append(repr(exc))
            finally:
                a=b=out=None
                with torch.cuda.device(i):
                    torch.cuda.synchronize(i)
        threads=[threading.Thread(target=pulse,args=(i,)) for i in range(8)]
        for t in threads:t.start()
        for t in threads:t.join(timeout=45)
        assert not any(t.is_alive() for t in threads), 'Live CUDA pulse threads'
        torch._C._cuda_clearCublasWorkspaces()
        for i in range(8):
            with torch.cuda.device(i):torch.cuda.synchronize(i);torch.cuda.empty_cache()
        rows=[dict(uuid=l['uuid'],allocated=torch.cuda.memory_allocated(i),reserved=torch.cuda.memory_reserved(i)) for i,l in enumerate(leases)]
        write(folder/f'done-{phase}.json',dict(pid=os.getpid(),phase=phase,duration=duration,errors=errors,devices=rows,epoch=time.time()))
        assert not errors and all(r['allocated']==r['reserved']==0 for r in rows)
    while not (folder/'exit').exists():
        if time.monotonic()-began>420:raise RuntimeError('Own exit timeout')
        time.sleep(.05)
if __name__=='__main__':main()
