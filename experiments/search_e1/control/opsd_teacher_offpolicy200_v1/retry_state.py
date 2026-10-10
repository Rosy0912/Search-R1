"""Durable overflow state is separate from accepted Adam checkpoints."""
from pathlib import Path
import json

try:
    from .checkpoint import checked_json, checked_file, fileproof, require, science_contract
except ImportError:
    from checkpoint import checked_json, checked_file, fileproof, require, science_contract


def select_retry_capsule(training, checkpoint_ref, config):
    """Return an explicit bound ref; never restore from an uncommitted PT file."""
    cp, _ = checked_json(checkpoint_ref)
    candidates=[]
    for path in Path(training).glob('attempt-*/retry-update-*-attempt-*/manifest.json'):
        value=json.loads(path.read_text())
        if (value.get('accepted') is True and value.get('failure_kind')=='gradient_overflow'
                and value.get('checkpoint',{}).get('sha256')==checkpoint_ref['sha256']):
            validate_manifest(value,cp,checkpoint_ref,config)
            candidates.append((value['created_unix'],fileproof(path)))
    return max(candidates,key=lambda x:x[0])[1] if candidates else None


def validate_manifest(value, cp, cp_ref, config):
    require(value['schema']=='opsd-ablation-overflow-retry-v1' and value['accepted'] is True,'Incomplete retry capsule')
    require(value['failure_kind']=='gradient_overflow','Forward failure must not be replayed as scaler recovery')
    require(value['config_sha256']==config['_config_sha256'] and value['run_id']==config['run_id'], 'Retry run/config differs')
    require(value['contract']==science_contract(config) and value['checkpoint']==cp_ref,'Retry accepted checkpoint binding differs')
    require(value['cursor']==cp['cursor'] and value['outer_old_adapter']==cp['outer_old_adapter'],'Retry cursor/old differs')
    require(len(value['ranks'])==8 and {x['rank'] for x in value['ranks']}==set(range(8)),'Eight retry states required')
    require(len({x['file']['path'] for x in value['ranks']})==8,'Retry rank files aliased')
    require(value['scaler_after']['enabled'] is True,'BF16 retry is not an authorized overflow recovery')
    return True


def load_rank_capsule(ref, cp, cp_ref, config, rank, current_adapter_digest, current_optimizer_digest):
    import torch
    value,_=checked_json(ref);validate_manifest(value,cp,cp_ref,config)
    for item in value['ranks']:checked_file(item['file'])
    file=next(x['file'] for x in value['ranks'] if x['rank']==rank)
    state=torch.load(file['path'],map_location='cpu',weights_only=False)
    require(state['rank']==rank and state['run_id']==config['run_id'] and state['contract']==science_contract(config),'Actual retry state identity differs')
    require(state['optimizer_updates']==cp['optimizer_updates'] and state['cursor']==cp['cursor']['next_pair_cursor'],'Actual retry Adam/cursor differs')
    require(state['order_sha256']==cp['cursor']['order_sha256'] and state['outer_old_adapter']==cp['outer_old_adapter'],'Actual retry order/old differs')
    require(state['adapter_tensor_sha256']==current_adapter_digest==cp['all_rank_adapter_sha256'][rank], 'Retry adapter changed without accepted Adam')
    require(state['optimizer_tensor_sha256']==current_optimizer_digest==cp['all_rank_optimizer_tensor_sha256'][rank], 'Retry actual Adam changed')
    require(state['step']['scaler_after']==value['scaler_after'] and state['step']['failure_kind']=='gradient_overflow', 'Actual retry scaler differs')
    return state
