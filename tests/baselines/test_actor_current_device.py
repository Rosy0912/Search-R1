"""Explicit logical CUDA routing CPU proofs; not actual GPU transfer evidence."""
import ast
import contextlib
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from tensordict import TensorDictBase
from verl import DataProto
from verl.workers.actor.dp_actor import DataParallelPPOActor


def actor_fixture():
    path=Path(__file__).with_name('test_sequence_mean_actor.py')
    spec=importlib.util.spec_from_file_location('device_actor_fixture',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('logical_rank,visible',[(0,'7'),(3,'0,1,2,3,4,5,6,7'),(7,'0,1,2,3,4,5,6,7')])
def test_actual_cpu_input_actor_uses_explicit_logical_rank(monkeypatch,logical_rank,visible):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES',visible)
    module=actor_fixture();data=module.fixture();before=data.batch.clone()
    calls=[]
    monkeypatch.setattr(torch.cuda,'current_device',lambda:logical_rank)
    def route(self,device=None,**kwargs):
        # Would be the old unspecified device path; refuse it in this proof.
        assert device==logical_rank
        calls.append(device);return self
    monkeypatch.setattr(TensorDictBase,'cuda',route)
    gradients,metrics=module.actor(data,16,2)
    assert calls==[logical_rank]*8 and metrics['actor/optimizer_steps']==1
    assert all(t.device.type=='cpu' for t in data.batch.values())
    for key in before.keys():torch.testing.assert_close(data.batch[key],before[key],rtol=0,atol=0)
    torch.testing.assert_close(gradients[0].reshape(()),module.oracle(data),rtol=2e-7,atol=2e-7)


@pytest.mark.parametrize('logical_rank',[0,5])
def test_actual_logprob_reference_micro_routes_current_device(monkeypatch,logical_rank):
    data=actor_fixture().fixture()
    data.meta_info.update(micro_batch_size=4,use_dynamic_bsz=False)
    calls=[];monkeypatch.setattr(torch.cuda,'current_device',lambda:logical_rank)
    def route(self,device=None,**kwargs):
        assert device==logical_rank;calls.append(device);return self
    monkeypatch.setattr(TensorDictBase,'cuda',route)
    def forward(micro_batch,temperature):
        assert all(t.device.type=='cpu' for t in micro_batch.values())
        value=torch.full_like(micro_batch['responses'],-3.,dtype=torch.float32)
        return value*0,value
    worker=SimpleNamespace(config=SimpleNamespace(),actor_module=torch.nn.Linear(1,1),_forward_micro_batch=forward)
    result=DataParallelPPOActor.compute_log_prob(worker,data)
    assert calls==[logical_rank]*4 and result.shape==(16,7)
    assert torch.equal(result,torch.full((16,7),-3.))


class BoundaryReached(Exception):pass

@pytest.mark.parametrize('method_name',['update_actor','compute_log_prob','generate_sequences','compute_ref_log_prob'])
@pytest.mark.parametrize('logical_rank',[0,6])
def test_actual_fsdp_actor_rollout_ref_boundary_explicit_device(method_name,logical_rank,monkeypatch):
    # Execute actual worker AST up to sharding-manager boundary. No CUDA/FSDP
    # object is created; spy preserves CPU input while checking requested routes.
    source=Path(__file__).resolve().parents[2]/'verl/workers/fsdp_workers.py'
    tree=ast.parse(source.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ActorRolloutRefWorker')
    method=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==method_name))
    method.decorator_list=[];unit=ast.Module(body=[method],type_ignores=[]);ast.fix_missing_locations(unit)
    calls=[];monkeypatch.setattr(torch.cuda,'current_device',lambda:logical_rank)
    class Batch:
        def cuda(self,device=None):
            assert device==logical_rank;calls.append(('batch.cuda',device));return self
    class Input:
        def __init__(self):self.batch=Batch();self.meta_info={}
        def to(self,device):
            assert device==torch.device('cuda',logical_rank);calls.append(('proto.to',str(device)));return self
    class Manager:
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def preprocess_data(self,*args,**kwargs):raise BoundaryReached()
    worker=SimpleNamespace(_is_actor=True,_is_rollout=True,_is_ref=True,_is_offload_param=False,
        _is_offload_optimizer=False,ulysses_sharding_manager=Manager(),rollout_sharding_manager=Manager(),
        tokenizer=SimpleNamespace(eos_token_id=2,pad_token_id=0),config=SimpleNamespace(
            ref=SimpleNamespace(log_prob_micro_batch_size=4,log_prob_max_token_len_per_gpu=64,
                log_prob_use_dynamic_bsz=False),rollout=SimpleNamespace(temperature=1.)))
    namespace=dict(torch=torch,DataProto=DataProto,logger=None,log_gpu_memory_usage=lambda *a,**k:None)
    exec(compile(unit,str(source),'exec'),namespace)
    with pytest.raises(BoundaryReached):namespace[method_name](worker,Input())
    assert calls[0]==('proto.to',f'cuda:{logical_rank}')
    assert calls[1:]==([] if method_name=='compute_ref_log_prob' else [('batch.cuda',logical_rank)])


def test_installed_tensordict_parser_explicit_index_overrides_ambient_device(monkeypatch):
    # Device parsing only; no CUDA allocation. Local 0.10 resolves unspecified
    # CUDA differently from official 0.5; explicit indices work in both APIs.
    from tensordict.utils import _parse_to
    for ambient in (0,3,7):
        monkeypatch.setattr(torch.cuda,'current_device',lambda:ambient)
        assert _parse_to(torch.device('cuda',3))[0]==torch.device('cuda',3)
