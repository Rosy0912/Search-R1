"""Exercise lazy Ray GPU binding plus FSDP synchronization, without a trained model."""
import argparse
import json
import os
from pathlib import Path
import socket
import time


class BindingActor:
    def __init__(self, rank, port):
        from datetime import timedelta
        import torch
        import torch.distributed as dist
        # Exercise the same import and hook used by frozen model workers, after
        # Ray's actor GPU mask is installed rather than in process setup.
        from verl.workers import fsdp_workers
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        if torch.cuda.device_count()!=1:
            raise ValueError('Ray actor saw more than its single assigned GPU')
        torch.cuda.set_device(0)
        dist.init_process_group('nccl',init_method='tcp://127.0.0.1:'+str(port),
                                rank=rank,world_size=8,timeout=timedelta(seconds=90))
        torch.manual_seed(100+rank)
        self.model=FSDP(torch.nn.Linear(128,128),device_id=0,sync_module_states=True)
        self.rank=rank

    def evidence(self):
        import subprocess
        import torch
        import torch.distributed as dist
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        with FSDP.summon_full_params(self.model,writeback=False):
            checksum=float(sum(p.double().sum().item() for p in self.model.parameters()))
        values=[None]*8;dist.all_gather_object(values,checksum)
        if len(set(values))!=1:
            raise ValueError('FSDP parameter synchronization differs across ranks')
        rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid',
                                      '--format=csv,noheader,nounits'],text=True)
        uuids={r.split(',')[1].strip() for r in rows.splitlines()
               if r.split(',')[0].strip()==str(os.getpid())}
        if len(uuids)!=1:
            raise ValueError('Ray worker has CUDA contexts on multiple physical GPUs')
        return dict(rank=self.rank,pid=os.getpid(),gpu_uuid=next(iter(uuids)),
                    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    visible_device_count=torch.cuda.device_count(),checksum=checksum,
                    fsdp_sync_verified=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);a=parser.parse_args()
    if (os.getuid()!=1066 or os.environ.get('SLURM_JOB_NODELIST')!='node6' or
            os.environ.get('SLURM_JOB_NAME')!='rt16-seven-ray-binding-probe-n6-v1' or
            not os.environ.get('SLURM_JOB_ID','').isdigit()):
        raise ValueError('Requires the exact task-owned eight-GPU diagnostic allocation')
    from release.recursive_bran_a.h100_migration.ray_cli_compat import prepare_click
    prepare_click()
    import ray
    from .runner import worker_setup
    os.environ['TREE_SEVEN_EVAL_MODE']='frozen_milestone_v1'
    os.environ['TREE_SEVEN_CHECKPOINT_IDENTITY']='communication-binding-diagnostic-only'
    sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1];sock.close()
    env={k:v for k,v in os.environ.items() if k.startswith(('TREE_','NCCL_')) or k=='PYTHONPATH'}
    try:
        ray.init(address='local',num_gpus=8,num_cpus=16,include_dashboard=False,
                 object_store_memory=1024**3,_node_ip_address='127.0.0.1',
                 _temp_dir='/tmp/seven-ray-binding-'+os.environ['SLURM_JOB_ID'],
                 runtime_env={'worker_process_setup_hook':worker_setup,'env_vars':env})
        actor=ray.remote(num_gpus=1,num_cpus=1)(BindingActor)
        workers=[actor.remote(rank,port) for rank in range(8)]
        rows=ray.get([w.evidence.remote() for w in workers],timeout=150)
        if ({r['rank'] for r in rows}!=set(range(8)) or
                len({r['gpu_uuid'] for r in rows})!=8 or
                not all(r['visible_device_count']==1 and r['fsdp_sync_verified'] for r in rows)):
            raise ValueError('Eight distinct physical single-GPU Ray/FSDP workers not verified')
        result=dict(status='EIGHT_RANK_LAZY_RAY_BINDING_FSDP_VERIFIED',at=time.time(),
                    job=os.environ['SLURM_JOB_ID'],source=os.environ['RECURSIVE_SOURCE_COMMIT'],ranks=rows,
                    scope='Tiny random Linear FSDP/communication diagnostic; no trained model, optimizer update, rollout or evaluation')
        with Path(a.output).open('x') as f:json.dump(result,f,indent=2);f.flush();os.fsync(f.fileno())
        print(json.dumps(result),flush=True)
    finally:
        ray.shutdown()


if __name__=='__main__':main()
