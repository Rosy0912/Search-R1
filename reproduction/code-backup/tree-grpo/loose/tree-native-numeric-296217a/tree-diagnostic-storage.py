"""Synthetic full-window Tree capture storage probe; not 3B model peak proof."""
import os,time
from pathlib import Path
from types import SimpleNamespace
import torch
from search_r1.baselines.tree_rank_audit import TreeRankAudit


def probe(device,rank):
    records=[]
    for enabled in (False,True):
        torch.cuda.empty_cache()
        model=torch.nn.Linear(1,1,bias=False,device=device)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats(device)
        start=time.perf_counter();cpu=time.process_time();before_rng=torch.cuda.get_rng_state(device).clone()
        audit=TreeRankAudit(model,torch.empty(32),{'baseline_outer_step':1},
            SimpleNamespace(ppo_mini_batch_size=32,ppo_micro_batch_size=8,ppo_epochs=1),__file__) if enabled else None
        before=torch.cuda.memory_allocated(device)
        for i in range(4):
            full=torch.full((8,6144),-.1,device=device)
            logp=full[:,-4096:]
            old=torch.full_like(logp,-.2);ref=torch.full_like(logp,-.15)
            mask=torch.ones_like(logp);adv=torch.ones_like(logp);weight=torch.ones_like(logp)/131072
            ids=torch.ones((8,6144),device=device,dtype=torch.int64)
            loss=(logp*weight*adv).sum()
            if enabled:audit.micro(logp,old,mask,loss,advantages=adv,ref_logp=ref,
                diagnostic_data=dict(input_ids=ids,policy_weights=weight))
            del full,logp,old,ref,mask,adv,weight,ids,loss
        torch.cuda.synchronize();retained=torch.cuda.memory_allocated(device)
        tensors=[] if not enabled else [t for m in audit.pending_microbatches for t in m['tensors'].values()]
        unique={t.untyped_storage().data_ptr():t.untyped_storage().nbytes() for t in tensors}
        logical=sum(t.numel()*t.element_size() for t in tensors)
        del tensors
        if enabled:
            assert len(audit.pending_microbatches)==4
            audit.gradient(torch.tensor(1.,device=device))
            assert not audit.pending_microbatches
        torch.cuda.synchronize();cleared=torch.cuda.memory_allocated(device)
        assert torch.equal(torch.cuda.get_rng_state(device),before_rng)
        records.append(dict(enabled=enabled,rank=rank,local_micro=8,local_mini=32,
            synthetic_response_window=4096,synthetic_backing_window=6144,
            before_allocated=before,retained_allocated=retained,after_clear_allocated=cleared,
            unique_storage_bytes=sum(unique.values()),logical_tensor_bytes=logical,
            peak_allocated=torch.cuda.max_memory_allocated(device),peak_reserved=torch.cuda.max_memory_reserved(device),
            wall_seconds=time.perf_counter()-start,process_cpu_seconds=time.process_time()-cpu,
            scope='Synthetic fixed full-window storage probe; no real3B peak or throughput acceptance'))
        del audit,model
    assert records[0]['unique_storage_bytes']==0
    assert records[1]['unique_storage_bytes']>0
    assert records[1]['after_clear_allocated']<records[1]['retained_allocated']
    return records
