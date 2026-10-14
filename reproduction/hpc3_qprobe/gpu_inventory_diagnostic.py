"""Actual single-GPU index/XML/CUDA diagnostic; no model or training state."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import time

from reproduction.common8.native_allocation import fields,allocated_indices,add_gpu_minor_numbers,cuda_uuids
from reproduction.hpc3_qprobe.control import publish


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    root=args.output.resolve()
    if not str(root).startswith('/data/user/jxu957/experiments/qprobe-teacher-warmup200-20261014/preparation/gpu-inventory717607/'):
        raise ValueError('Diagnostic output must belong to this specific own failed-job diagnostic')
    root.mkdir(parents=True,exist_ok=False)
    job_id=os.environ['SLURM_JOB_ID'];node=socket.gethostname().split('.')[0]
    detail=subprocess.check_output(['scontrol','show','job','-d',job_id],text=True,timeout=30)
    job=fields(detail)
    assert os.getuid()==14090 and job['JobState']=='RUNNING' and job['UserId']=='jxu957(14090)' and job['NumNodes']=='1'
    commands=[['nvidia-smi','--help-query-gpu'],['nvidia-smi','--query-gpu=index,uuid,memory.total','--format=csv,noheader,nounits'],['nvidia-smi','-q','-x']]
    observations=[]
    for command in commands:
        result=subprocess.run(command,capture_output=True,text=True,timeout=30)
        value=dict(command=command,exit=result.returncode,stdout=result.stdout,stderr=result.stderr,epoch=time.time())
        publish(root/('command-'+str(len(observations))+'.json'),value);observations.append(value)
        assert result.returncode==0,'Actual NVIDIA query failed; raw result preserved'
    raw=observations[1]['stdout'];rows=[[x.strip() for x in row] for row in csv.reader(io.StringIO(raw))]
    rows=add_gpu_minor_numbers(rows,observations[2]['stdout']);assigned=allocated_indices(detail,node,1)
    visible=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    import torch
    assert torch.cuda.device_count()==1
    actual=cuda_uuids(1)
    index_map={int(row[0]):row[1] for row in rows};minor_map={int(row[3]):row[1] for row in rows}
    expected=[index_map[int(token)] if token.isdecimal() else token for token in visible]
    assert len(visible)==1 and expected==actual and set(actual)=={minor_map[i] for i in assigned}
    value=torch.arange(16,device='cuda',dtype=torch.float32).square().sum()
    assert value.item()==1240;torch.cuda.synchronize()
    proof=dict(accepted=True,job_id=job_id,uid=os.getuid(),node=node,actual_job_detail=detail,
        visible_devices=visible,actual_slurm_indices=assigned,actual_gpu_inventory=rows,
        actual_cuda_runtime_uuids=actual,kernel_sum=value.item(),completed_training_outers=0,
        optimizer_writes=0,loaded_model=False,checked_unix=time.time(),
        scope='actual GPU query/index/CUDA kernel only; not native4 model/training acceptance')
    publish(root/'accepted.json',proof);print(json.dumps(proof),flush=True)


if __name__=='__main__':main()
