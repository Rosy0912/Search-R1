"""CPU structural/fault tests only; fake CUDA bytes never claim real GPU state."""
import copy
import importlib.util
import os
from pathlib import Path
import random
import unittest

import numpy as np
import torch

from reproduction.scaleup_native_gpu.contract import CORE_SHA,NATIVE_SHA,sha
from reproduction.scaleup_native_gpu.transition import transform,rng_transition

ROOT=Path(os.environ['TINY_NATIVE_PROBE_ROOT'])
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
assert sha(ROOT/'assets/training-state-frozen.py')==NATIVE_SHA
assert sha(ROOT/'assets/flat-core-frozen.py')==CORE_SHA
native=load('tiny_native_frozen',ROOT/'assets/training-state-frozen.py')
core=load('tiny_flat_core',ROOT/'assets/flat-core-frozen.py')


def fake_cuda_state(seed):return torch.tensor(list(seed.to_bytes(8,'little')+b'\0'*8),dtype=torch.uint8)


def fixture():
    layout=dict(parameters={'flat':dict(logical_numel=14,local_numel=2,dtype='torch.float32',fqns=['weight'],shapes=[[14]],numels=[14])},optimizer_pid_to_name={0:'flat'})
    target=copy.deepcopy(layout);target['parameters']['flat']['local_numel']=1
    values=[]
    for rank in range(8):
        p=core.split(torch.arange(14,dtype=torch.float32)/1024,rank,8)
        rng=dict(python=random.Random(rank).getstate(),numpy=np.random.RandomState(rank).get_state(),torch=torch.Generator().manual_seed(rank).get_state(),cuda=dict(device=rank,state=fake_cuda_state(rank)))
        state=dict(version=2,identity=dict(rank=rank,world_size=8,model='tiny',data='fixed',global_batch=512,configured_steps=200),parameters={'flat':p},local_shards={'flat':p.clone()},gradients={'flat':None},buffers={'fixed':torch.tensor([1,2])},training=True,scheduler=dict(last_epoch=50),optimizer=dict(state={0:dict(step=torch.tensor(1000.),exp_avg=core.split(torch.ones(14)*.1,rank,8),exp_avg_sq=core.split(torch.ones(14)*.01,rank,8))},param_groups=[dict(params=[0],lr=1e-6)]),rng=rng)
        values.append(state)
    return values,[copy.deepcopy(layout) for _ in range(8)],target


class Transition(unittest.TestCase):
    def setUp(self):self.states,self.layouts,self.target=fixture()
    def convert(self,rank=8):
        identity=dict(self.states[0]['identity'],rank=rank,world_size=16)
        return transform(self.states,self.layouts,self.target,identity,core,native.digest,rank%8,fake_cuda_state)

    def test_all_logical_parameters_and_adam_preserved(self):
        outputs=[self.convert(r) for r in range(16)]
        for key in ['parameters','local_shards']:
            core.same(core.canonical_vector([s[key]['flat'] for s in self.states],14,key),core.canonical_vector([s[key]['flat'] for s in outputs],14,key),key)
        for key in ['exp_avg','exp_avg_sq']:
            core.same(core.canonical_vector([s['optimizer']['state'][0][key] for s in self.states],14,key),core.canonical_vector([s['optimizer']['state'][0][key] for s in outputs],14,key),key)

    def test_first8_rng_exact_and_new8_deterministic_without_global_rng(self):
        before=native.digest(native.rng_state())
        for rank in range(16):
            a=self.convert(rank);b=self.convert(rank)
            self.assertEqual(native.digest(a),native.digest(b))
            if rank<8:self.assertEqual(native.digest(a['rng']),native.digest(self.states[rank]['rng']))
        self.assertEqual(before,native.digest(native.rng_state()))
        self.assertFalse(torch.cuda.is_initialized())

    def test_wrong_source_cuda_device(self):
        self.states[0]['rng']['cuda']['device']=1
        with self.assertRaises(ValueError):self.convert()

    def test_missing_or_malformed_cuda_bytes(self):
        for mode in ['missing','dtype','shape']:
            states=copy.deepcopy(self.states)
            if mode=='missing':states[0]['rng'].pop('cuda')
            if mode=='dtype':states[0]['rng']['cuda']['state']=torch.ones(16)
            if mode=='shape':states[0]['rng']['cuda']['state']=torch.ones(17,dtype=torch.uint8)
            with self.assertRaises(ValueError):rng_transition([s['rng'] for s in states],8,0,native.digest,fake_cuda_state)

    def test_negative_second_moment(self):
        self.states[0]['optimizer']['state'][0]['exp_avg_sq'][0]=-1
        with self.assertRaises(ValueError):self.convert()

    def test_fractional_adam_counter(self):
        self.states[0]['optimizer']['state'][0]['step']=torch.tensor(1000.5)
        with self.assertRaises(ValueError):self.convert()

    def test_missing_source_rank(self):
        self.states.pop()
        with self.assertRaises(ValueError):self.convert()

    def test_target_local_device_or_state_format(self):
        parents=[s['rng'] for s in self.states]
        with self.assertRaises(ValueError):rng_transition(parents,8,1,native.digest,fake_cuda_state)
        with self.assertRaises(ValueError):rng_transition(parents,8,0,native.digest,lambda seed:torch.zeros(1,dtype=torch.uint8))


if __name__=='__main__':unittest.main()
