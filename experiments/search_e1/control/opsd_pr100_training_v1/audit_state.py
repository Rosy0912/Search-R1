"""Explicit CPU-only numerical state readback; no model, signals or RNG restore."""
import argparse
import json
import os
from pathlib import Path
import time

from checkpoint import checked_json,checked_file,check_checkpoint,require


def read(config_ref,checkpoint_ref,updates,outer):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU audit requires explicit empty CUDA visibility')
    import torch
    from safetensors.torch import load_file
    from state_storage import hydrate_rank_optimizer
    from trainer import optimizer_tensor_digest,tensor_digest,verify_rank_state,actual_adam_state
    require(not torch.cuda.is_initialized(),'CUDA was already initialized')
    config,cf=checked_json(config_ref);config['_config_sha256']=cf['sha256']
    value,evidence=check_checkpoint(checkpoint_ref,config,config['run_id'],updates,outer)
    payload=None
    def loader(path):
        nonlocal payload
        if payload is None:payload=torch.load(path,map_location='cpu',weights_only=False)
        return payload
    ranks=[]
    for entry in sorted(value['ranks'],key=lambda x:x['rank']):
        state=torch.load(entry['file']['path'],map_location='cpu',weights_only=False)
        expanded=hydrate_rank_optimizer(state,value,loader,optimizer_tensor_digest)
        verify_rank_state(expanded,value,entry['rank'],config)
        actual_adam_state(expanded['optimizer'],updates,504)
        require(all(t.dtype==torch.float32 for item in expanded['optimizer']['state'].values()
                    for name,t in item.items() if name in ('exp_avg','exp_avg_sq')),'Adam moments must be FP32')
        ranks.append(dict(rank=entry['rank'],actual_adam_step=updates,optimizer_tensor_sha256=optimizer_tensor_digest(expanded['optimizer']),
                          full_rng_validated=True,scaler=state['scaler']))
    adapter_ref=next(f for f in value['adapter']['files'] if Path(f['path']).name=='adapter_model.safetensors')
    adapter=load_file(adapter_ref['path'],device='cpu')
    initial,initial_proof=checked_json(config['common_initial_adapter_manifest']);checked_file(initial['adapter_model'])
    original=load_file(initial['adapter_model']['path'],device='cpu')
    require(set(adapter)==set(original) and len(adapter)==504,'Common/default504 adapter inventory differs')
    require(all(t.dtype==torch.float32 and torch.isfinite(t).all().item() for t in adapter.values()),'Adapter must be finite FP32')
    digest=tensor_digest(adapter);require(digest==value['all_rank_adapter_sha256'][0],'Actual adapter tensor digest differs')
    changed=sum(not torch.equal(t,original[k]) for k,t in adapter.items())
    require((updates==0 and changed==0) or (updates>0 and changed>0),'Actual zero-update/fresh changed-weight relation differs')
    require(not torch.cuda.is_initialized(),'CPU-only audit initialized CUDA')
    return dict(schema='opsd-ablation-independent-cpu-state-readback-v1',accepted=True,checked_unix=time.time(),
        run_id=config['run_id'],variant=config['variant'],config=cf,checkpoint=evidence,optimizer_updates=updates,outer_index=outer,
        ranks=ranks,all_eight_exact_full_adam=True,adapter_tensors=504,adapter_finite_fp32=True,
        changed_tensors_vs_common_initial=changed,adapter_tensor_sha256=digest,common_initial=initial_proof,cuda_initialized=False,
        scope='Saved actual full Adam payload and eight distinct RNG/scaler/cursor bindings; no tokenizer/logits or all-history token recomputation')


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--config-sha256',required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--checkpoint-sha256',required=True)
    p.add_argument('--expected-updates',required=True,type=int);p.add_argument('--expected-outer',required=True,type=int)
    a=p.parse_args();print(json.dumps(read(dict(path=a.config,sha256=a.config_sha256),
        dict(path=a.checkpoint,sha256=a.checkpoint_sha256),a.expected_updates,a.expected_outer),sort_keys=True))


if __name__=='__main__':main()
