"""Verify complete finite checkpoint tensors and actual actor weight changes."""
import argparse,json,torch
from pathlib import Path
from safetensors import safe_open
p=argparse.ArgumentParser();p.add_argument('--initial',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--report',required=True);a=p.parse_args()
def inventory(root):
 root=Path(root);files=list(root.glob('model*.safetensors'));result={}
 if not files:raise ValueError('No checkpoint shards')
 for path in files:
  with safe_open(path,framework='pt',device='cpu') as f:
   for key in f.keys():
    if key in result:raise ValueError('Duplicate tensor key')
    result[key]=path
 return result
initial=inventory(a.initial);saved=inventory(a.checkpoint)
if set(saved)!=set(initial):raise ValueError('Incomplete/unexpected checkpoint tensor keys')
nonfinite=[];changed=[]
for key,path in saved.items():
 with safe_open(path,framework='pt',device='cpu') as f: t=f.get_tensor(key)
 with safe_open(initial[key],framework='pt',device='cpu') as f:
  if list(t.shape)!=f.get_slice(key).get_shape():raise ValueError('Checkpoint shape mismatch: '+key)
 if not torch.isfinite(t).all():nonfinite.append(key)
 if key.endswith('self_attn.q_proj.weight') and len(changed)<4:
  with safe_open(initial[key],framework='pt',device='cpu') as f: original=f.get_tensor(key)
  if t.shape!=original.shape:raise ValueError('Checkpoint shape mismatch')
  delta=(t.float()-original.float()).abs().max().item()
  if delta>0:changed.append(dict(key=key,max_abs_delta=delta))
report=dict(checkpoint=a.checkpoint,tensor_count=len(saved),nonfinite=nonfinite,changed=changed,accepted=not nonfinite and bool(changed))
Path(a.report).write_text(json.dumps(report,indent=2))
if not report['accepted']:raise SystemExit('Checkpoint rejected: inspect '+a.report)
print(json.dumps(report),flush=True)
