"""Real tiny GPU FSDP8 native save, then new16-process original cold restore.

Controlled logical gradients test exact model/Adam state across shard layouts.
This is not a7B checkpoint, rollout equivalence, production guard migration,
or representative training throughput benchmark.
"""
import argparse
import copy
import datetime
import importlib.util
import json
import os
from pathlib import Path
import random
import socket
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from .contract import read_plan, sha
from .transition import POLICY, transform


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__();self.first=torch.nn.Linear(17,31);self.second=torch.nn.Linear(31,13)
        self.register_buffer('fixed_buffer',torch.tensor([17,31,13],dtype=torch.int64))
        with torch.no_grad():
            for p in self.parameters():p.copy_(torch.arange(p.numel()).reshape(p.shape)/1024)
    def forward(self,x):return self.second(torch.tanh(self.first(x)))


def training(local):
    m=Tiny().cuda(local)
    m.first=FSDP(m.first,device_id=local,use_orig_params=False)
    m.second=FSDP(m.second,device_id=local,use_orig_params=False)
    m=FSDP(m,device_id=local,use_orig_params=False)
    o=torch.optim.AdamW(m.parameters(),lr=1e-6,betas=(.9,.999),weight_decay=.01,foreach=False)
    s=torch.optim.lr_scheduler.LambdaLR(o,lambda i:min(i/57,1.0))
    m(torch.ones(2,17,device='cuda')).sum().backward();o.zero_grad(set_to_none=True)
    assert not o.state
    return m,o,s


def gradient(n,call):return torch.sin(torch.arange(n,dtype=torch.float32)+call)*.02


def update(m,o,lay,rank,world,call,core):
    # Execute real FSDP forward/backward, then use controlled logical gradients
    # to isolate native-state/Adam reconstruction from topology-dependent sums.
    m(torch.ones(2,17,device='cuda')).sum().backward()
    for n,p in m.named_parameters():
        p.grad=core.split(gradient(lay['parameters'][n]['logical_numel'],call),rank,world).to(p.device)
    o.step();o.zero_grad(set_to_none=True)


class Stage:
    def state_dict(self):return dict(counter=137,selected=[4,9,2])
    def load_state_dict(self,s):assert s==self.state_dict()


def driver(native):
    return SimpleNamespace(global_steps=50,actual_outer_updates=50,
        checkpoint_sampler=native.CheckpointBatchSampler(169028,512,19,'tiny-index-fixture169028'),
        baseline_controller=SimpleNamespace(method='tree_grpo',stage=Stage(),tree_rng=random.Random(39)))


def reconstruct(states,lay,core):
    ps={n:core.canonical_vector([s['parameters'][n] for s in states],v['logical_numel'],n)
        for n,v in lay['parameters'].items()}
    ms={}
    for pid,n in lay['optimizer_pid_to_name'].items():
        ms[pid]=dict(step=states[0]['optimizer']['state'][pid]['step'].clone())
        for k in ['exp_avg','exp_avg_sq']:
            ms[pid][k]=core.canonical_vector([s['optimizer']['state'][pid][k] for s in states],lay['parameters'][n]['logical_numel'],k)
    return dict(parameters=ps,moments=ms)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['save8','load16'],required=True);ap.add_argument('--plan',required=True);args=ap.parse_args()
    pp=Path(args.plan);plan=read_plan(pp);root=Path(plan['root']);data=root/'state'
    local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local);torch.set_num_threads(1)
    assert torch.__version__=='2.4.0+cu121' and torch.cuda.device_count()==8
    dist.init_process_group('nccl',timeout=datetime.timedelta(seconds=180))
    rank,world=dist.get_rank(),dist.get_world_size();assert world==(8 if args.phase=='save8' else 16)
    native=module('frozen_training_state',root/'assets/training-state-frozen.py')
    core=module('frozen_flat_core',root/'assets/flat-core-frozen.py')
    m,o,s=training(local);lay=core.layout(m,o)
    identity=dict(rank=rank,world_size=world,model='tiny-native-cuda-v1',data='tiny-index-fixture169028',global_batch=512,global_minibatch=128,configured_steps=200)
    hardware=json.loads((root/f'node-{rank//8}/hardware.json').read_text())
    gpu=hardware['devices'][local][0].strip()
    me=dict(rank=rank,local_rank=local,world_size=world,hostname=socket.gethostname(),uuid=gpu,plan_sha256=sha(pp))
    ids=[None]*world;dist.all_gather_object(ids,me)
    assert len({x['uuid'] for x in ids})==world and len({x['hostname'] for x in ids})==(1 if world==8 else 2)
    if args.phase=='save8':
        random.seed(7000+rank);np.random.seed(9000+rank);torch.random.default_generator.manual_seed(11000+rank);torch.cuda.manual_seed(13000+rank)
        for outer in range(1,51):
            for call in range((outer-1)*20+1,outer*20+1):update(m,o,lay,rank,world,call,core)
            s.step()
        saved=native.worker_snapshot(m,o,s,identity,cuda=True)
        assert {int(v['step'].item()) for v in saved['optimizer']['state'].values()}=={1000}
        native.atomic_save(saved,data/f'source-rank-{rank}.pt');native.atomic_save(lay,data/f'source-layout-{rank}.pt')
        if rank==0:
            d=driver(native);it=iter(d.checkpoint_sampler)
            for _ in range(50):next(it)
            native.atomic_save(native.driver_snapshot(d,dict(model=identity['model'],data=identity['data'])),data/'driver.pt')
        report=dict(**me,source_actual_adam_steps=1000,scheduler_last_epoch=s.last_epoch,native_sha256=sha(data/f'source-rank-{rank}.pt'),cuda_rng_state_sha256=native.digest(saved['rng']['cuda']),actual_gpu_fsdp=True)
    else:
        ready=json.loads((root/'source-ready.json').read_text());assert ready['plan_sha256']==sha(pp)
        for n,h in ready['files'].items():assert sha(data/n)==h,n
        source=[torch.load(data/f'source-rank-{r}.pt',map_location='cpu',weights_only=False) for r in range(8)]
        layouts=[torch.load(data/f'source-layout-{r}.pt',map_location='cpu',weights_only=False) for r in range(8)]
        full=reconstruct(source,layouts[0],core)
        rng_before=native.digest(native.rng_state(cuda=True))
        def private_cuda_state(seed):return torch.Generator(device=torch.device('cuda',local)).manual_seed(seed).get_state()
        converted=transform(source,layouts,lay,identity,core,native.digest,local,private_cuda_state)
        assert native.digest(native.rng_state(cuda=True))==rng_before,'Converter mutated caller RNG'
        native.atomic_save(converted,data/f'converted-rank-{rank}.pt')
        loaded=torch.load(data/f'converted-rank-{rank}.pt',map_location='cpu',weights_only=False)
        assert not o.state
        native.restore_worker(m,o,s,loaded,identity,cold=True)
        restored=native.worker_snapshot(m,o,s,identity,cuda=True)
        assert native.digest(restored)==native.digest(converted),'Original CUDA cold restore changed a native field'
        restored_states=[None]*world;dist.all_gather_object(restored_states,restored)
        core.same(full,reconstruct(restored_states,lay,core),'all full logical model/Adam before next update')
        # Verify the actual next CUDA RNG samples using an independent private generator.
        gen=torch.Generator(device=torch.device('cuda',local));gen.set_state(converted['rng']['cuda']['state'])
        expected_draw=torch.rand(8,generator=gen,device='cuda');actual_draw=torch.rand(8,device='cuda')
        assert torch.equal(expected_draw,actual_draw)
        d=driver(native);ds=torch.load(data/'driver.pt',map_location='cpu',weights_only=False)
        did=dict(model=identity['model'],data=identity['data']);native.restore_driver(d,ds,did)
        sample=native.CheckpointBatchSampler(169028,512,19,identity['data']);sample.load_state_dict(ds['sampler'])
        assert next(iter(d.checkpoint_sampler))==next(iter(sample)) and ds['sampler']['cursor']==25600
        ordered=list(layouts[0]['parameters']);params=[torch.nn.Parameter(full['parameters'][n].to('cuda')) for n in ordered]
        ref=torch.optim.AdamW(params,lr=1e-6,betas=(.9,.999),weight_decay=.01,foreach=False)
        ref.load_state_dict(dict(state=copy.deepcopy(full['moments']),param_groups=copy.deepcopy(source[0]['optimizer']['param_groups'])))
        rs=torch.optim.lr_scheduler.LambdaLR(ref,lambda i:min(i/57,1.0));rs.load_state_dict(source[0]['scheduler'])
        # Constructing LambdaLR sets lr=0; restore the actual saved optimizer group LR.
        for rg,sg in zip(ref.param_groups,source[0]['optimizer']['param_groups']):rg['lr']=sg['lr']
        first_next=None
        for call in range(1001,1021):
            for n,param in zip(ordered,params):param.grad=gradient(layouts[0]['parameters'][n]['logical_numel'],call).to('cuda')
            ref.step();ref.zero_grad(set_to_none=True);update(m,o,lay,rank,world,call,core)
            if call==1001:first_next=[int(st['step'].item()) for st in o.state.values()]
        s.step();rs.step();after=native.worker_snapshot(m,o,s,identity,cuda=True)
        core.same(s.state_dict(),rs.state_dict(),'scheduler after next completed tiny outer')
        all_after=[None]*world;dist.all_gather_object(all_after,after)
        expected=dict(parameters={n:param.detach().cpu() for n,param in zip(ordered,params)},moments=native.cpu_copy(ref.state_dict()['state']))
        core.same(expected,reconstruct(all_after,lay,core),'next actual CUDA controlled-gradient Adam1001to1020')
        assert first_next and set(first_next)=={1001} and s.last_epoch==51
        assert {int(st['step'].item()) for st in after['optimizer']['state'].values()}=={1020}
        d.global_steps=d.actual_outer_updates=51;driver_after=native.driver_snapshot(d,did)
        assert driver_after['sampler']['cursor']==26112 and driver_after['next_step']==52
        report=dict(**me,actual_original_native_cuda_cold_restore=True,source_adam1000_target_first1001_final1020=True,full_native_fields_after_cold_load_exact=True,all_full_logical_model_and_adam_exact=True,next_controlled_gradient_gpu_adam20_exact=True,scheduler_after_next_outer=51,original_driver_sampler_next_batch_exact=True,driver_check_in_worker_fixture_not_production_Ray_driver=True,cuda_rng_policy=POLICY,source_ranks0to7_rng_retained_new8to15_derived=True,actual_next_cuda_samples_exact=True,converter_did_not_consume_process_rng=True,converted_native_sha256=sha(data/f'converted-rank-{rank}.pt'),tiny_only_not7b_or_rollout_trajectory_equivalence=True,production_guard_accounting_not_adapted=True,production_migration=False,production_throughput_benchmark=False)
    report.update(torch_version=torch.__version__,no_llm_generation_or_retrieval=True)
    native.atomic_save(report,data/f'{args.phase}-rank-{rank}.json',json_format=True)
    dist.barrier()
    if rank==0:native.atomic_save(dict(phase=args.phase,world_size=world,all_gpu_identities=ids,all_ranks_passed=True),data/f'{args.phase}-complete.json',json_format=True)
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
