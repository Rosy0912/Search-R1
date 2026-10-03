"""Verify finite actor tensors and changes, including verified tied-head aliases."""
import argparse,json,torch
from pathlib import Path
from safetensors import safe_open

def inventory(root):
 files=list(Path(root).glob('model*.safetensors'));result={}
 if not files:raise ValueError('No checkpoint shards')
 for path in files:
  with safe_open(path,framework='pt',device='cpu') as f:
   for key in f.keys():
    if key in result:raise ValueError('Duplicate tensor key')
    result[key]=path
 return result

def tensor(items,key):
 with safe_open(items[key],framework='pt',device='cpu') as f:return f.get_tensor(key)

def verify(initial_path,saved_path):
 initial=inventory(initial_path);saved=inventory(saved_path)
 configs=[json.loads((Path(p)/'config.json').read_text()) for p in [initial_path,saved_path]]
 tied=all(c.get('tie_word_embeddings') is True for c in configs)
 aliases={};missing=set(initial)-set(saved);extra=set(saved)-set(initial)
 if missing or extra:
  if missing or extra!={'lm_head.weight'} or not tied or 'model.embed_tokens.weight' not in initial or 'model.embed_tokens.weight' not in saved:
   raise ValueError('Incomplete/unexpected checkpoint tensor keys')
  aliases['lm_head.weight']='model.embed_tokens.weight'
 # FSDP may materialize a second safe-tensor entry for an originally shared head.
 # Accept only the declared alias with exact values in the saved checkpoint.
 if tied:
  for items in [initial,saved]:
   if {'lm_head.weight','model.embed_tokens.weight'}<=set(items):
    if not torch.equal(tensor(items,'lm_head.weight'),tensor(items,'model.embed_tokens.weight')):
     raise ValueError('Declared tied head differs from token embedding')
 nonfinite=[];changed=[]
 for key,path in saved.items():
  t=tensor(saved,key);original_key=aliases.get(key,key)
  with safe_open(initial[original_key],framework='pt',device='cpu') as f:
   if list(t.shape)!=f.get_slice(original_key).get_shape():raise ValueError('Checkpoint shape mismatch: '+key)
  if not torch.isfinite(t).all():nonfinite.append(key)
  if key.endswith('self_attn.q_proj.weight') and len(changed)<4:
   original=tensor(initial,key);delta=(t.float()-original.float()).abs().max().item()
   if delta>0:changed.append(dict(key=key,max_abs_delta=delta))
 return dict(checkpoint=str(saved_path),tensor_count=len(saved),initial_tensor_count=len(initial),verified_aliases=aliases,nonfinite=nonfinite,changed=changed,accepted=not nonfinite and bool(changed))

def main():
 p=argparse.ArgumentParser();p.add_argument('--initial',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--report',required=True);a=p.parse_args()
 report=verify(a.initial,a.checkpoint);Path(a.report).write_text(json.dumps(report,indent=2))
 if not report['accepted']:raise SystemExit('Checkpoint rejected: inspect '+a.report)
 print(json.dumps(report),flush=True)
if __name__=='__main__':main()
