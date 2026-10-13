"""Read-only continuous physical memory guard for own shared training unit."""
import argparse
import json
import os
from pathlib import Path
import time
from production_contract import require
from shared_identity import verify_cgroup,selected_hardware,PROTOCOL,job_id,unit_name

def pid_identity(pid):
    stat=(Path('/proc')/str(pid)/'stat').read_text();tail=stat[stat.rfind(')')+2:].split()
    return int(tail[19]),(Path('/proc')/str(pid)/'cgroup').read_text()

def own_resource_counters(identity):
    """Read only this service's small cgroup counters; absent peak stays unknown."""
    scope=Path(identity['scope']);result={}
    for name in ('pids.current','pids.max','pids.peak','pids.events','memory.current','memory.peak','memory.events'):
        path=scope/name
        if not path.is_file():result[name]=None;continue
        text=path.read_text().strip()
        result[name]=dict(line.split() for line in text.splitlines()) if name.endswith('.events') else text
    return result

def sample():
    identity=verify_cgroup();nv,mapping=selected_hardware();rows=[]
    for _,uid,handle in mapping:
        memory=nv.nvmlDeviceGetMemoryInfo(handle);own=0;foreign={};seen=set()
        for proc in nv.nvmlDeviceGetComputeRunningProcesses(handle):
            if proc.pid in seen:continue
            seen.add(proc.pid)
            require(isinstance(proc.usedGpuMemory,int) and 0<=proc.usedGpuMemory<=memory.total,'Unknown live per-process GPU memory')
            try:start,cgroup=pid_identity(proc.pid)
            except FileNotFoundError:raise RuntimeError('GPU process identity changed during observation; retry only own guard termination')
            paths=[line.split(':',2)[-1] for line in cgroup.splitlines()]
            if any(path==identity['relative'] or path.startswith(identity['relative']+'/') for path in paths):
                require(pid_identity(proc.pid)==(start,cgroup),'Own GPU PID identity changed while reading');own+=proc.usedGpuMemory
            else:foreign[str(proc.pid)]=dict(start_time=start,memory_bytes=proc.usedGpuMemory)
        row=dict(uuid=uid,total_memory_bytes=memory.total,total_used_memory_bytes=memory.used,free_memory_bytes=memory.free,
                 total_gpu_util=nv.nvmlDeviceGetUtilizationRates(handle).gpu,own_memory_bytes=own,foreign=foreign)
        rows.append(row)
        require(memory.free>=8*1024**3 and own<=72*1024**3,'Own shared training physical memory headroom/cap breached: '+json.dumps(row))
    return dict(epoch=time.time(),job_id=job_id(),unit=unit_name(),resource_protocol=PROTOCOL,
                physical_gpus=8,slurm_reserved_gpus=0,gpus=rows,own_cgroup_resources=own_resource_counters(identity),
                ray_num_server_call_thread=os.environ.get('RAY_num_server_call_thread'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    with a.output.open('x') as stream:
        while True:
            try:value=sample()
            except BaseException as error:
                stream.write(json.dumps(dict(epoch=time.time(),job_id=job_id(),event='STOP',error=repr(error)))+'\n');stream.flush();os.fsync(stream.fileno());raise
            stream.write(json.dumps(value)+'\n');stream.flush()
            ready=a.output.with_suffix('.ready')
            if not ready.exists():ready.write_text('ACTUAL_SHARED_TRAINING_MEMORY_GUARD_READY\n')
            time.sleep(1)
if __name__=='__main__':main()
