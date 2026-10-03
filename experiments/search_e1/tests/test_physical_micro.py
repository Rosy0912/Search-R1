"""Compare actual archived and patched actor updates with real PPO reductions."""
import ast, os, sys, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from tensordict import TensorDict
CONTROL=Path(__file__).resolve().parents[1]/'control'
sys.path.insert(0,str(CONTROL))
from physical_micro import patch_actor
UP=CONTROL.parent/'archive/upstream/Search-R1/verl'
def extracted(source,name,namespace):
    tree=ast.parse(source)
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
    node.returns=None
    for arg in node.args.args:arg.annotation=None
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<actual-source>','exec'),namespace)
    return namespace[name]
functional={}
extracted((UP/'utils/torch_functional.py').read_text(),'masked_mean',functional)
ns=dict(torch=torch,verl_F=SimpleNamespace(**functional))
for name in ['compute_policy_loss','kl_penalty']:
    extracted((UP/'trainer/ppo/core_algos.py').read_text(),name,ns)
algos=SimpleNamespace(compute_policy_loss=ns['compute_policy_loss'],kl_penalty=ns['kl_penalty'])
def append(metrics,data):
    for k,v in data.items():metrics.setdefault(k,[]).append(v)
def method(source,name):
    result = extracted(source,name,dict(torch=torch,os=os,core_algos=algos,verl_F=SimpleNamespace(**functional),masked_mean=functional['masked_mean'],append_to_dict=append))
    if os.environ.get('E1_CHECK_TRAJECTORY') == '1':
        from types import FunctionType
        def trajectory_mean(values, mask, axis=None):
            if axis is not None:return functional['masked_mean'](values,mask,axis=axis)
            return functional['masked_mean'](values,mask,axis=-1).mean()
        vf=SimpleNamespace(masked_mean=trajectory_mean)
        policy=FunctionType(algos.compute_policy_loss.__code__,dict(algos.compute_policy_loss.__globals__,verl_F=vf))
        result=FunctionType(result.__code__,dict(result.__globals__,core_algos=SimpleNamespace(compute_policy_loss=policy,kl_penalty=algos.kl_penalty),verl_F=vf,masked_mean=trajectory_mean))
    return result
class PhysicalMicroTests(unittest.TestCase):
    def test_actual_update_gradients_weights_and_metrics(self):
        torch.manual_seed(42)
        source=(UP/'workers/actor/dp_actor.py').read_text()
        original=method(source,'update_policy');adapted=method(patch_actor(source),'update_policy')
        mask=(torch.rand(32,7)>.45).double();mask[:4]=0
        if os.environ.get('E1_CHECK_TRAJECTORY') == '1':mask[:,0]=1
        batch=TensorDict(dict(responses=torch.zeros(32,7,dtype=torch.long),input_ids=torch.randn(32,7,dtype=torch.double),attention_mask=mask,position_ids=torch.zeros(32,7,dtype=torch.long),loss_mask=mask,old_log_probs=torch.randn(32,7,dtype=torch.double)*.4,ref_log_prob=torch.randn(32,7,dtype=torch.double)*.4,advantages=torch.randn(32,7,dtype=torch.double)),batch_size=[32])
        class Data:
            meta_info={'temperature':1.}
            def select(self,**kwargs):return SimpleNamespace(batch=batch)
        def actor():
            model=torch.nn.Linear(1,1,dtype=torch.double)
            with torch.no_grad():model.weight.fill_(.17);model.bias.fill_(-.2)
            opt=torch.optim.AdamW(model.parameters(),lr=1e-3)
            gradients=[]
            calls=[]
            def forward(micro_batch,temperature):
                calls.append(len(micro_batch))
                lp=model(micro_batch['input_ids'].unsqueeze(-1)).squeeze(-1)
                return torch.sigmoid(lp),lp
            def step():
                gradients.append(torch.cat([p.grad.flatten() for p in model.parameters()]).clone())
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                opt.step();return norm
            return SimpleNamespace(config=SimpleNamespace(ppo_mini_batch_size=16,ppo_micro_batch_size=8,use_dynamic_bsz=False,state_masking=True,use_kl_loss=True,clip_ratio=.2,entropy_coeff=.001,kl_loss_type='low_var_kl',kl_loss_coef=.001),actor_module=model,actor_optimizer=opt,_forward_micro_batch=forward,_optimizer_step=step,gradients=gradients,calls=calls)
        cap=int(os.environ.get("E1_CHECK_PHYSICAL", "4"));assert cap in [1,2,4,8]
        a,b=actor(),actor()
        with patch.object(TensorDict,'cuda',lambda self:self),patch.dict(os.environ,{'E1_PHYSICAL_MICRO':str(cap),'SEARCH_E1_TRAJECTORY_BATCHING':os.environ.get('E1_CHECK_TRAJECTORY','0')}):
            ma=original(a,Data());mb=adapted(b,Data())
        self.assertEqual(len(a.calls),4);self.assertEqual(len(b.calls),32//cap)
        self.assertEqual(len(a.gradients),2);self.assertEqual(len(b.gradients),2)
        for x,y in zip(a.gradients,b.gradients):torch.testing.assert_close(x,y,atol=1e-12,rtol=1e-12)
        for x,y in zip(a.actor_module.parameters(),b.actor_module.parameters()):torch.testing.assert_close(x,y,atol=1e-12,rtol=1e-12)
        for k in ma:torch.testing.assert_close(torch.tensor(ma[k]),torch.tensor(mb[k]),atol=1e-7,rtol=1e-7)
    def test_logical_micro_configuration_stays_unchanged(self):
        source=patch_actor((UP/'workers/actor/dp_actor.py').read_text())
        self.assertIn('self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size',source)
        self.assertIn('mini_batch.split(self.config.ppo_micro_batch_size)',source)
        ast.parse(source)
if __name__=='__main__':unittest.main()
