"""CPU-only final saved-state acceptance after five complete native online outers."""
import os,json,sys,hashlib,time
from pathlib import Path
B=Path('/data/shaoshitong/search-e1-opsd-j-minus-p-bf16-20261010');cfgpath=B/'receipts/training-config.json';raw=cfgpath.read_bytes();cfg=json.loads(raw);cfg['_config_sha256']=hashlib.sha256(raw).hexdigest();sys.path.insert(0,cfg['paths']['helper_root'])
from checkpoint import check_checkpoint,checked_json,fileproof,require
from trainer import verify_rank_state,optimizer_tensor_digest
import torch
from safetensors.torch import load_file
require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only acceptance required')
closed=[checked_json(fileproof(B/'runs/outers'/f'outer-{i:05d}'/'training/outer-complete.json'))[0] for i in range(5)]
updates=sum(x['optimizer_updates_delta'] for x in closed);pairs=sum(x['K'] for x in closed)
require(all(x['accepted'] and x['outer_index']==i for i,x in enumerate(closed)),'Five complete native rounds missing')
require(updates==sum((x['K']+31)//32 for x in closed),'Actual update accounting differs')
cp,proof=check_checkpoint(closed[-1]['checkpoint'],cfg,cfg['run_id'],updates,4)
shared=torch.load(cp['shared_optimizer']['path'],map_location='cpu',weights_only=False);digest=optimizer_tensor_digest(shared['optimizer']);require(digest==cp['all_rank_optimizer_tensor_sha256'][0],'Full optimizer digest mismatch')
for rank in cp['ranks']:
 state=torch.load(rank['file']['path'],map_location='cpu',weights_only=False);require(state['optimizer_ref']==cp['shared_optimizer'],'Shared state reference differs');state['optimizer']=shared['optimizer'];verify_rank_state(state,cp,rank['rank'],cfg)
weights=load_file(str(Path(cp['adapter']['path'])/'adapter_model.safetensors'),device='cpu');require(len(weights)==504 and all(t.dtype==torch.float32 and torch.isfinite(t).all().item() for t in weights.values()),'Actual saved 504 FP32 LoRA invalid')
initial=json.loads(Path(cfg['common_initial_adapter_manifest']['path']).read_text());initialpath=Path(initial['adapter_model']['path']);before=load_file(str(initialpath),device='cpu');require(set(before)==set(weights),'Adapter keys differ');changed=sum(not torch.equal(weights[k],before[k]) for k in weights);require(changed==504,'Not all LoRA tensors changed')
require(not torch.cuda.is_initialized(),'CPU audit initialized CUDA')
print(json.dumps(dict(accepted=True,run_id=cfg['run_id'],precision='bf16',mechanisms=['R','W','M'],outers=5,real_Adam=updates,pair_exposure=pairs,each_outer=[dict(outer=x['outer_index'],K=x['K'],Adam=x['optimizer_updates_delta']) for x in closed],final_checkpoint=closed[-1]['checkpoint'],eight_rank_saved_state_finite_equal=True,adapter_FP32_tensors=504,adapter_finite_changed=changed,CUDA_initialized=False,observed_unix=time.time())))
