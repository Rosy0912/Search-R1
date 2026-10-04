"""Keep the raw token denominator while routing scalar collectives correctly."""
import os
import tempfile
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pytest
from search_r1.baselines.distillation import global_token_count


def test_nccl_routes_only_count_to_current_device(monkeypatch):
    moves=[]
    original=torch.Tensor.to
    def intercepted(self, *args, **kwargs):
        if args and isinstance(args[0],torch.device) and args[0].type=='cuda':
            moves.append((self.numel(),args[0]));return self
        return original(self,*args,**kwargs)
    monkeypatch.setattr(torch.Tensor,'to',intercepted)
    monkeypatch.setattr(dist,'is_initialized',lambda:True)
    monkeypatch.setattr(dist,'get_backend',lambda:'nccl')
    monkeypatch.setattr(torch.cuda,'current_device',lambda:7)
    monkeypatch.setattr(dist,'get_world_size',lambda:8)
    monkeypatch.setattr(dist,'all_reduce',lambda value,op:value.mul_(8))
    mask=torch.tensor([[1.,0.,1.],[0.,1.,0.]])
    count,world=global_token_count(mask)
    assert count.item()==24 and world==8
    assert moves==[(1,torch.device('cuda',7))]
    assert torch.equal(mask,torch.tensor([[1.,0.,1.],[0.,1.,0.]]))


def _gloo_counts(rank, path, output):
    dist.init_process_group('gloo',init_method='file://'+path,rank=rank,world_size=2)
    try:
        mask=torch.tensor([[1.,1.,1.]]) if rank==0 else torch.tensor([[1.,0.,0.]])
        count,world=global_token_count(mask)
        empty,empty_world=global_token_count(torch.zeros_like(mask))
        torch.save({'count':count,'world':world,'empty':empty,'empty_world':empty_world},output+str(rank))
    finally:dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(),reason='Gloo unavailable')
def test_real_gloo_unequal_counts_and_empty_are_collective(tmp_path):
    mp.spawn(_gloo_counts,args=(str(tmp_path/'rendezvous'),str(tmp_path/'result')),nprocs=2,join=True)
    for rank in range(2):
        value=torch.load(str(tmp_path/'result')+str(rank),map_location='cpu')
        assert value['count'].device.type=='cpu'
        assert value['count'].item()==4 and value['world']==2
        assert value['empty'].item()==0 and value['empty_world']==2
