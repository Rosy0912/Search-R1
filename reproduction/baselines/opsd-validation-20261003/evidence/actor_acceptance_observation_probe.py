import ast,json,torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
tree=ast.parse((ROOT/'search_r1/baselines/actor.py').read_text())
method=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DistillationActor').body
method=next(n for n in method if isinstance(n,ast.FunctionDef) and n.name=='update_policy')
start=next(i for i,n in enumerate(method.body) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=="observations['actor/samples']")
end=next(i for i,n in enumerate(method.body) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=="observations['actor/optimizer_steps']")
code=compile(ast.Module(body=method.body[start:end],type_ignores=[]),'actual_actor_observation_block','exec')
class Data:
 def __init__(self,batch):self.batch=batch
 def __len__(self):return self.batch['responses'].shape[0]
results={}
for name in ['valid','fractional_mask','negative_mask','attention_excluded','info_mismatch','nan_old_lp','absent_old_lp']:
 mask=torch.tensor([[1.,0.,1.],[0.,1.,0.]])
 attention=torch.tensor([[1,1,0,1],[1,0,1,0]])
 info=torch.cat([torch.ones(2,1),mask],dim=1)
 lp=torch.zeros_like(mask)
 if name=='fractional_mask':mask[0,0]=.5
 if name=='negative_mask':mask[0,0]=-1
 if name=='attention_excluded':attention[0,-3]=0
 if name=='info_mismatch':info[0,-3]=0
 if name=='nan_old_lp':lp[0,0]=float('nan')
 batch={'responses':torch.zeros(2,3,dtype=torch.long),'loss_mask':mask,'attention_mask':attention,'info_mask':info}
 if name!='absent_old_lp':batch['old_log_probs']=lp
 snapshots={k:(v.clone(),v._version) for k,v in batch.items()}
 ns={'torch':torch,'data':Data(batch),'observations':{}};exec(code,ns);obs=ns['observations']
 assert obs['actor/loss_mask_valid']==int(name not in ['fractional_mask','negative_mask','attention_excluded','info_mismatch'])
 if name=='nan_old_lp':assert obs['actor/old_log_probs_finite']==0
 elif name=='absent_old_lp':assert obs['actor/old_log_probs_present']==0 and 'actor/old_log_probs_finite' not in obs
 else:assert obs['actor/old_log_probs_finite']==1
 for k,(before,version) in snapshots.items():
  assert batch[k]._version==version
  torch.testing.assert_close(batch[k],before,rtol=0,atol=0,equal_nan=True)
 results[name]=obs
print(json.dumps({'cases':results,'input_versions_and_values_unchanged':True,'scope':'actual AST observation block; CPU only; no optimizer/GPU'},indent=2))
