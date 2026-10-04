"""Scalar routing tests; actual NCCL eight-rank evidence remains a GPU gate."""
import json
from pathlib import Path
import importlib.util
import torch
import pytest
from search_r1.baselines.distillation import global_sum_count, global_token_count, global_token_denominator
from search_r1.baselines.objectives import row_denominator
from tensordict import TensorDictBase


def test_uninitialized_cpu_preserves_value_device(monkeypatch):
    monkeypatch.setattr(torch.distributed,'is_initialized',lambda:False)
    def forbidden(*a,**k):raise AssertionError('unexpected CUDA/backend/collective')
    monkeypatch.setattr(torch.distributed,'get_backend',forbidden)
    monkeypatch.setattr(torch.cuda,'current_device',forbidden)
    count=torch.tensor(7.)
    total,world=global_sum_count(count)
    assert total is count and total.item()==7 and world==1
    assert row_denominator(0,'cpu').item()==1
    assert row_denominator(13,'cpu').item()==13


def mock_backend(monkeypatch,backend,world=8):
    transfers=[];collectives=[];routed_ids=set()
    original_to=torch.Tensor.to
    def routed_to(self,*a,**kw):
        device=a[0] if a else kw.get('device')
        if isinstance(device,torch.device) and device.type=='cuda':
            assert self.ndim==0  # Entire CPU DataProto must not move here.
            transfers.append(str(device));routed_ids.add(id(self));return self
        return original_to(self,*a,**kw)
    monkeypatch.setattr(torch.Tensor,'to',routed_to)
    monkeypatch.setattr(torch.distributed,'is_initialized',lambda:True)
    monkeypatch.setattr(torch.distributed,'get_backend',lambda:backend)
    monkeypatch.setattr(torch.distributed,'get_world_size',lambda:world)
    monkeypatch.setattr(torch.cuda,'current_device',lambda:3)
    def reduction(value,op=None):
        assert op==torch.distributed.ReduceOp.SUM
        if backend=='nccl':
            if id(value) not in routed_ids:raise RuntimeError('No backend type associated with device type cpu')
            routed_ids.remove(id(value))
        collectives.append(float(value));value.mul_(world)
    monkeypatch.setattr(torch.distributed,'all_reduce',reduction)
    return transfers,collectives


@pytest.mark.parametrize('backend',['nccl','gloo','undefined'])
@pytest.mark.parametrize('empty',[False,True])
def test_backend_routes_only_scalar_exact_counts(monkeypatch,backend,empty):
    transfers,collectives=mock_backend(monkeypatch,backend)
    mask=torch.zeros(3,4) if empty else torch.tensor([[1.,0,1,0],[0,1,0,0],[0,0,0,0]])
    clone=mask.clone();total,world=global_token_count(mask)
    assert total.item()==(0 if empty else 24) and world==8
    denominator=row_denominator(3,'cpu')
    assert denominator.item()==3  # Empty rows remain in row measure.
    assert torch.equal(mask,clone) and mask.device.type=='cpu'
    assert transfers==(['cuda:3','cuda:3'] if backend=='nccl' else [])
    assert collectives==[0. if empty else 3.,3.]


def test_actual_actor_incoming_cpu_dataproto_uses_nccl_scalar_route(monkeypatch):
    # Reuse the actual actor CPU fixture, retaining DataProto on CPU. Only CUDA
    # kernels/device transfer are shimmed; real update_policy count paths run.
    path=Path(__file__).with_name('test_sequence_mean_actor.py')
    spec=importlib.util.spec_from_file_location('actor_count_fixture',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(TensorDictBase,'cuda',lambda self,*a,**kw:self)
    transfers,_=mock_backend(monkeypatch,'nccl')
    data=module.fixture();assert all(t.device.type=='cpu' for t in data.batch.values())
    gradients,metrics=module.actor(data,16,2)
    assert transfers==['cuda:3','cuda:3']  # Token SUM then global row SUM.
    assert metrics['actor/optimizer_steps']==1 and torch.isfinite(gradients[0]).all()
    torch.testing.assert_close(gradients[0].reshape(()),module.oracle(data),atol=2e-7,rtol=2e-7)


def gloo_counts_worker(rank,rendezvous,output):
    torch.distributed.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2)
    try:
        results=[]
        for tokens,rows in [((0,7),(3,5)),((0,0),(3,5)),((2,9),(0,11))]:
            mask=torch.ones(tokens[rank]);total,world=global_token_count(mask)
            row=row_denominator(rows[rank],'cpu')
            den=global_token_denominator(mask)
            assert world==2 and total.item()==sum(tokens) and total.device.type=='cpu'
            assert row.item()==max(sum(rows),1)/2
            assert den.item()==max(sum(tokens),1)/2
            results.append(dict(tokens=total.item(),rows_per_rank=row.item(),world=world))
        Path(output+str(rank)+'.json').write_text(json.dumps(results))
    finally:torch.distributed.destroy_process_group()


def test_real_gloo_unequal_and_empty_rank_counts(tmp_path):
    torch.multiprocessing.spawn(gloo_counts_worker,args=(str(tmp_path/'gloo'),str(tmp_path/'rank')),nprocs=2,join=True)
    assert json.loads((tmp_path/'rank0.json').read_text())==json.loads((tmp_path/'rank1.json').read_text())


def test_previous_cpu_nccl_count_paths_reproduce_failure(monkeypatch):
    mock_backend(monkeypatch,'nccl')
    for scalar in (torch.ones(3).sum().float(),torch.tensor(float(4))):
        with pytest.raises(RuntimeError,match='No backend type associated with device type cpu'):
            torch.distributed.all_reduce(scalar,op=torch.distributed.ReduceOp.SUM)
    assert global_token_count(torch.ones(3))[0].item()==24
    assert row_denominator(4,'cpu').item()==4
