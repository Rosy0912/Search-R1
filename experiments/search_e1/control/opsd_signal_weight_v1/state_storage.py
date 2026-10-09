"""One exact optimizer payload, four distinct rank RNG/cursor files.

This is physical deduplication after equality of all four actual Adam digests.
Restoration expands the explicit file reference before normal rank validation.
"""
from pathlib import Path

try:
    from .checkpoint import checked_file, require
except ImportError:
    from checkpoint import checked_file, require

STORAGE='shared_exact_four_rank_optimizer_v1'


def optimizer_payload(optimizer,proofs,run_id,config_sha,updates,digest):
    require(len(proofs)==4 and {p['rank'] for p in proofs}==set(range(4)), 'Four actual rank proofs required before deduplication')
    require(all(p['optimizer_digest']==digest for p in proofs),'Actual four-rank Adam tensors/groups differ')
    return dict(schema=STORAGE,run_id=run_id,config_sha256=config_sha,optimizer_updates=updates,
                optimizer_tensor_sha256=digest,actual_rank_optimizer_digests=[p['optimizer_digest'] for p in sorted(proofs,key=lambda x:x['rank'])],optimizer=optimizer)


def hydrate_rank_optimizer(state,checkpoint,loader,digest_reader):
    require(state.get('optimizer_storage')==STORAGE and 'optimizer' not in state,'Rank file storage contract differs')
    reference=state.get('optimizer_ref')
    require(reference==checkpoint.get('shared_optimizer'),'Rank optimizer reference differs from complete checkpoint')
    checked_file(reference)
    value=loader(reference['path'])
    require(value.get('schema')==STORAGE,'Shared payload format differs')
    for key in ('run_id','config_sha256','optimizer_updates','optimizer_tensor_sha256'):
        require(value.get(key)==state.get(key),'Shared optimizer identity mismatch: '+key)
    expected=checkpoint['all_rank_optimizer_tensor_sha256']
    require(value['actual_rank_optimizer_digests']==expected and len(expected)==4 and len(set(expected))==1,'Four actual optimizer identities differ')
    require(digest_reader(value['optimizer'])==state['optimizer_tensor_sha256'],'Shared full Adam tensor/groups digest changed')
    expanded=dict(state);expanded['optimizer']=value['optimizer']
    return expanded
