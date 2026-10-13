"""User-authorized GPU sharing: observe memory headroom, never signal foreigners."""
import argparse,json,os
from pathlib import Path
import time

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--unit',required=True)
    a=p.parse_args();out=Path(a.output)
    import pynvml as nv
    nv.nvmlInit()
    selected=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    handles=[nv.nvmlDeviceGetHandleByUUID(x.encode()) for x in selected]
    def own(pid):
        try:return any(a.unit in line.split(':',2)[-1].split('/') for line in Path(f'/proc/{pid}/cgroup').read_text().splitlines())
        except FileNotFoundError:return None
    if not own(os.getpid()):raise RuntimeError('Shared guard outside own service cgroup')
    while True:
        rows=[]
        for uid,h in zip(selected,handles):
            mem=nv.nvmlDeviceGetMemoryInfo(h);util=nv.nvmlDeviceGetUtilizationRates(h)
            processes=nv.nvmlDeviceGetComputeRunningProcesses(h)+nv.nvmlDeviceGetGraphicsRunningProcesses(h)
            ours={};foreign={}
            for v in processes:
                owner=own(v.pid)
                if owner is None:continue
                if not isinstance(v.usedGpuMemory,int) or not 0<=v.usedGpuMemory<=mem.total:
                    raise RuntimeError('Unknown GPU process memory')
                target=ours if owner else foreign
                target[v.pid]=max(target.get(v.pid,0),v.usedGpuMemory)
            row=dict(uuid=uid,total_bytes=mem.total,free_bytes=mem.free,used_bytes=mem.used,
                     total_gpu_util=util.gpu,own_memory_bytes=sum(ours.values()),foreign=foreign)
            rows.append(row)
        event=dict(epoch=time.time(),unit=a.unit,allocation_kind='user_authorized_shared_systemd',
                   physical_gpus=8,slurm_reserved_gpus=0,resource_protocol='memory_headroom_shared_v1',gpus=rows)
        failed=any(v['free_bytes']<12*1024**3 or v['own_memory_bytes']>36*1024**3 for v in rows)
        if failed:event.update(event='STOP_OWN_SHARED_EVALUATOR',reason='Physical free memory below12GiB or own memory above36GiB; stop only own task')
        with out.open('a') as f:f.write(json.dumps(event,allow_nan=False)+'\n');f.flush()
        out.with_suffix('.ready').touch()
        if failed:raise RuntimeError(event['reason'])
        time.sleep(1)

if __name__=='__main__':main()
