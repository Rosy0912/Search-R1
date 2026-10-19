"""Bounded two-phase node coordinator: source8 exits before cold16 starts."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from .contract import node_coordinates,read_plan,sha,validate_gate


def write(path,value):
    temp=path.with_suffix('.pending')
    with temp.open('x') as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
    os.link(temp,path);temp.unlink()


def launch(cmd,env,log,seconds):
    with log.open('xb') as out:
        p=subprocess.Popen(cmd,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        try:return p.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=10)
            return 124


def idle_hardware():
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.total','--format=csv,noheader,nounits'],text=True,timeout=20)
    devices=[r.split(',') for r in rows.splitlines() if r.strip()]
    assert len(devices)==8 and len({d[0].strip() for d in devices})==8
    assert all('A100' in d[1] and int(d[2].strip())>=79000 for d in devices)
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True,timeout=20).strip()
    assert not processes,'Allocated GPU already used; do not interfere'
    return devices


def main():
    pp=Path(sys.argv[1]);plan=read_plan(pp);root=Path(plan['root'])
    node,master,port=node_coordinates(os.environ);directory=root/f'node-{node}';directory.mkdir(exist_ok=False)
    write(directory/'entry-once.json',dict(at=time.time(),node=node,hostname=socket.gethostname(),master=master,port=port,plan_sha256=sha(pp),environment={k:os.environ.get(k) for k in ['RANK','WORLD_SIZE','NODE_RANK','NODE_COUNT','MASTER_ADDR','MASTER_PORT']}))
    deadline=time.monotonic()+300
    while not (root/'scheduler-release-once.json').exists():
        assert time.monotonic()<deadline,'Registered unique gate missing';time.sleep(2)
    submission=json.loads((root/'submission.json').read_text());validate_gate(json.loads((root/'scheduler-release-once.json').read_text()),submission,sha(pp))
    write(directory/'hardware.json',dict(at=time.time(),devices=idle_hardware(),scheduler_uid=submission['uid'],no_prior_gpu_processes=True))
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NCCL_DEBUG='WARN',NCCL_NVLS_ENABLE='0',TORCH_NCCL_ASYNC_ERROR_HANDLING='1')
    for n in ['RAY_ADDRESS','ROCR_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES']:env.pop(n,None)
    common=[plan['runtime'],'-m','torch.distributed.run']
    worker=['-m','reproduction.scaleup_native_gpu.worker','--plan',str(pp)]
    if node==0:
        rc=launch(common+['--standalone','--nnodes=1','--nproc-per-node=8','--max-restarts=0']+worker+['--phase','save8'],env,directory/'save8.log',900)
        (directory/'save8.exit.code').open('x').write(str(rc)+'\n');assert rc==0,'Source tiny save failed; do not retry'
        names=[f'source-rank-{r}.pt' for r in range(8)]+[f'source-layout-{r}.pt' for r in range(8)]+['driver.pt']
        assert json.loads((root/'state/save8-complete.json').read_text())['all_ranks_passed']
        # Source torchrun has exited; verify its CUDA contexts are gone before publishing.
        idle_hardware()
        write(root/'source-ready.json',dict(at=time.time(),job=plan['job'],uid=submission['uid'],plan_sha256=sha(pp),source8_processes_exited=True,files={n:sha(root/'state'/n) for n in names}))
    else:
        deadline=time.monotonic()+1000
        while not (root/'source-ready.json').exists():
            assert time.monotonic()<deadline,'Tiny source8 did not finish; no automatic replay';time.sleep(2)
    ready=json.loads((root/'source-ready.json').read_text());assert ready['uid']==submission['uid'] and ready['plan_sha256']==sha(pp) and ready['source8_processes_exited']
    for n,h in ready['files'].items():assert sha(root/'state'/n)==h
    cmd=common+['--nnodes=2','--nproc-per-node=8','--node-rank='+str(node),'--master-addr='+master,'--master-port='+str(port),'--max-restarts=0']+worker+['--phase','load16']
    rc=launch(cmd,env,directory/'load16.log',600)
    (directory/'load16.exit.code').open('x').write(str(rc)+'\n');assert rc==0,'Tiny CUDA cold16 failed; preserve all evidence, never retry'
    (directory/'exit.code').open('x').write('0\n')


if __name__=='__main__':main()
