"""CPU transfer regression; no CUDA native training/release claim."""
import copy
import pytest
import torch
from verl.utils.fsdp_utils import offload_fsdp_optimizer,load_fsdp_optimizer
from search_r1.baselines.optimizer_transaction import transactional_adam_step


def assert_equal(a,b):
    if torch.is_tensor(a):
        assert torch.equal(a,b) and a.dtype==b.dtype and a.device==b.device
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:assert_equal(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):assert_equal(x,y)
    else:assert a==b


def test_uninitialized_adam_transfer_does_not_create_lazy_state():
    params=[torch.nn.Parameter(torch.ones(2)),torch.nn.Parameter(torch.ones(3))]
    optimizer=torch.optim.AdamW(params,lr=1e-6/57)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:1.)
    before=copy.deepcopy(optimizer.state_dict());epoch=scheduler.last_epoch
    for _ in range(3):
        offload_fsdp_optimizer(optimizer);load_fsdp_optimizer(optimizer,torch.device('cpu'))
        assert len(optimizer.state)==0
        assert_equal(optimizer.state_dict(),before)
        assert scheduler.last_epoch==epoch and optimizer.param_groups[0]['lr']==1e-6/57


@pytest.mark.parametrize('amsgrad',[False,True])
@pytest.mark.parametrize('shadow',[False,True])
def test_existing_native_adam_state_transfer_preserves_all_values_and_next_step(monkeypatch,amsgrad,shadow):
    p=torch.nn.Parameter(torch.tensor([1.,-2.],dtype=torch.float64))
    q=torch.nn.Parameter(p.detach().clone());unused=torch.nn.Parameter(torch.ones(1))
    optimizer=torch.optim.AdamW([p,unused],lr=.003,amsgrad=amsgrad)
    oracle=torch.optim.AdamW([q],lr=.003,amsgrad=amsgrad)
    p.grad=torch.tensor([.2,.3],dtype=p.dtype);q.grad=p.grad.clone()
    if shadow:assert transactional_adam_step(optimizer)['committed']
    else:optimizer.step()
    oracle.step()
    assert unused not in optimizer.state
    state=optimizer.state[p];state['transfer_metadata']='preserved'
    saved=copy.deepcopy(state);calls=[];original=torch.Tensor.to
    def transfer(tensor,*args,**kwargs):
        calls.append((tensor,args,kwargs));return original(tensor,*args,**kwargs)
    monkeypatch.setattr(torch.Tensor,'to',transfer)
    offload_fsdp_optimizer(optimizer)
    load_fsdp_optimizer(optimizer,torch.device('cpu'))
    assert optimizer.state[p] is state and unused not in optimizer.state
    assert_equal(state,saved)
    tensors=[value for value in state.values() if torch.is_tensor(value)]
    assert len(calls)==2*len(tensors)
    assert all(kwargs=={'non_blocking':True} for _,_,kwargs in calls)
    assert all(args==('cpu',) for _,args,_ in calls[:len(tensors)])
    assert all(args==(torch.device('cpu'),) for _,args,_ in calls[len(tensors):])
    assert state['step'].item()==1 and 'exp_avg' in state and 'exp_avg_sq' in state
    if amsgrad:assert 'max_exp_avg_sq' in state
    # The same lazy/native shadow implementation remains exact after transfer.
    p.grad=torch.tensor([-.1,.4],dtype=p.dtype);q.grad=p.grad.clone()
    if shadow:assert transactional_adam_step(optimizer)['committed']
    else:optimizer.step()
    oracle.step()
    assert_equal(p,q)
    for key,value in oracle.state[q].items():assert_equal(optimizer.state[p][key],value)
    assert optimizer.state[p]['transfer_metadata']=='preserved' and unused not in optimizer.state


@pytest.mark.parametrize('transfer',[offload_fsdp_optimizer,lambda optimizer:load_fsdp_optimizer(optimizer,'cpu')])
def test_existing_empty_state_retained_and_invalid_none_state_not_hidden(transfer):
    p=torch.nn.Parameter(torch.ones(1));optimizer=torch.optim.AdamW([p])
    existing={};optimizer.state[p]=existing
    transfer(optimizer)
    assert optimizer.state[p] is existing and len(optimizer.state)==1
    optimizer.state[p]=None
    with pytest.raises(AttributeError):transfer(optimizer)
    assert optimizer.state[p] is None


def test_actual_offload_helper_bytes_in_critical_source_binding():
    import hashlib
    from pathlib import Path
    from search_r1.baselines.fresh200_contract import critical_source_hashes
    source=Path(__file__).resolve().parents[2]/'verl/utils/fsdp_utils.py'
    assert critical_source_hashes()['verl/utils/fsdp_utils.py']==hashlib.sha256(source.read_bytes()).hexdigest()
