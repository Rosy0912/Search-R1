import ast
import importlib.util
from pathlib import Path
import subprocess
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[2]/'work/opsd-e1-300'
LOSS=ROOT/'search_r1/baselines/distillation.py'
spec=importlib.util.spec_from_file_location('author200_independent_loss',LOSS)
loss_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(loss_module)


def author_function():
    path=ROOT/'reproduction/baselines/opsd-validation-20261002/evidence/author_opsd_trainer.py'
    tree=ast.parse(path.read_text())
    matches=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='generalized_jsd_loss']
    assert len(matches)==1
    func=matches[0];func.decorator_list=[]
    module=ast.fix_missing_locations(ast.Module(body=[func],type_ignores=[]))
    namespace={'torch':torch,'F':F}
    exec(compile(module,str(path),'exec'),namespace)
    return namespace['generalized_jsd_loss']


def test_full_vocab_component_clip_matches_fixed_author_values_and_gradients():
    torch.manual_seed(400)
    source=(torch.randn(32,5,13)*3).requires_grad_()
    teacher=torch.randn(32,5,13)*3
    mask=(torch.arange(5)[None,:] < (torch.arange(32)%5+1)[:,None])
    labels=torch.where(mask,torch.zeros_like(mask,dtype=torch.long),-100)
    actual=(loss_module.generalized_jsd(source,teacher,beta=0.,temperature=1.,component_clip=.05)*mask).sum()/mask.sum()
    reference=author_function()(source,teacher,labels=labels,beta=0.,temperature=1.,token_clip=.05)
    torch.testing.assert_close(actual,reference,atol=1e-7,rtol=1e-6)
    actual_grad=torch.autograd.grad(actual,source,retain_graph=True)[0]
    ref_grad=torch.autograd.grad(reference,source)[0]
    torch.testing.assert_close(actual_grad,ref_grad,atol=1e-8,rtol=1e-6)
    assert not torch.equal(actual_grad,torch.autograd.grad((loss_module.generalized_jsd(source,teacher,component_clip=None)*mask).sum()/mask.sum(),source)[0])


def test_eight_rank_two_micro_accumulations_equal_global32_token_mean():
    torch.manual_seed(401)
    features=torch.randn(32,7,3)
    teacher=torch.randn(32,7,11)*2
    mask=(torch.arange(7)[None,:] < (torch.arange(32)%7+1)[:,None])
    weight=torch.randn(3,11,requires_grad=True)
    logits=features@weight
    target=(loss_module.generalized_jsd(logits,teacher,component_clip=.05)*mask).sum()/mask.sum()
    expected=torch.autograd.grad(target,weight)[0]
    grads=[]
    observed_loss=[]
    for rank in range(8):
        local_weight=weight.detach().clone().requires_grad_()
        rank_loss=0.
        for micro in range(2):
            begin=rank*4+micro*2;end=begin+2
            logits=features[begin:end]@local_weight
            loss=(loss_module.generalized_jsd(logits,teacher[begin:end],component_clip=.05)*mask[begin:end]).sum()/(mask.sum()/8)
            loss.backward()
            rank_loss+=float(loss.detach())
        grads.append(local_weight.grad)
        observed_loss.append(rank_loss)
    torch.testing.assert_close(torch.stack(grads).mean(0),expected,atol=1e-8,rtol=1e-5)
    assert abs(sum(observed_loss)/8-float(target.detach()))<1e-7
    assert not torch.allclose(torch.stack(grads).mean(0)/2,expected,atol=1e-8,rtol=1e-5), 'extra accumulation division would change target'


def test_global_gradient_l2_clip_point_one_scales_complete_vector():
    first=torch.nn.Parameter(torch.tensor([1.,2.]))
    second=torch.nn.Parameter(torch.tensor([3.]))
    first.grad=torch.tensor([3.,4.]);second.grad=torch.tensor([12.])
    returned=torch.nn.utils.clip_grad_norm_([first,second],max_norm=.1)
    assert returned.item()==13.
    joined=torch.cat([first.grad,second.grad])
    assert .0999999<float(joined.norm())<=.10000001
    torch.testing.assert_close(joined,torch.tensor([3.,4.,12.])*(.1/(13+1e-6)),atol=1e-8,rtol=1e-6)


def production_scheduler_function(old=False):
    source=(subprocess.check_output(['git','-C',str(ROOT),'show','HEAD:verl/utils/torch_functional.py'],text=True)
            if old else (ROOT/'verl/utils/torch_functional.py').read_text())
    func=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='get_constant_schedule_with_warmup')
    namespace={'Optimizer':torch.optim.Optimizer,'LambdaLR':torch.optim.lr_scheduler.LambdaLR}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[func],type_ignores=[])),'actual_production_scheduler.py','exec'),namespace)
    return namespace['get_constant_schedule_with_warmup']


def test_constant_no_warmup_first_adam_and_scheduler_clock():
    p=torch.nn.Parameter(torch.tensor(1.))
    optim=torch.optim.AdamW([p],lr=5e-6)
    scheduler=production_scheduler_function()(optim,num_warmup_steps=0)
    recorded=[]
    for outer in range(5):
        for update in range(16):
            recorded.append(optim.param_groups[0]['lr'])
            optim.zero_grad();p.grad=torch.ones_like(p);optim.step()
        scheduler.step()
    assert recorded==[5e-6]*80
    assert optim.state[p]['step'].item()==80
    assert scheduler.last_epoch==5 and scheduler._step_count==6


def test_frozen_old_scheduler_counterexample_zero_warmup_first_outer_lr_zero():
    p=torch.nn.Parameter(torch.tensor(1.))
    optim=torch.optim.AdamW([p],lr=5e-6)
    scheduler=production_scheduler_function(old=True)(optim,num_warmup_steps=0)
    before=p.detach().clone()
    for update in range(16):
        assert optim.param_groups[0]['lr']==0.
        optim.zero_grad();p.grad=torch.ones_like(p);optim.step()
    assert torch.equal(p,before)
    assert optim.state[p]['step'].item()==16
    scheduler.step()
    assert optim.param_groups[0]['lr']==5e-6


def test_positive_57_warmup_actual_factory_curve_unchanged_through_horizon():
    previous=torch.nn.Parameter(torch.tensor(1.));current=torch.nn.Parameter(torch.tensor(1.))
    prior_opt=torch.optim.AdamW([previous],lr=1e-6);current_opt=torch.optim.AdamW([current],lr=1e-6)
    prior=production_scheduler_function(old=True)(prior_opt,num_warmup_steps=57)
    now=production_scheduler_function()(current_opt,num_warmup_steps=57)
    for outer in range(301):
        assert prior_opt.param_groups[0]['lr']==current_opt.param_groups[0]['lr']==1e-6*min(1,outer/57)
        for opt,p in [(prior_opt,previous),(current_opt,current)]:
            opt.zero_grad();p.grad=torch.ones_like(p);opt.step()
        prior.step();now.step()
    assert torch.equal(previous,current)
