"""Same-world FSDP continuation; original reference and logical schedule remain."""
import os,json,hashlib
from pathlib import Path
import ray

def restore_worker(w, directory):
    import torch,random,numpy as np
    from pathlib import Path
    root=Path(directory)
    state=torch.load(root/('full-rank-'+str(w.rank)+'.pt'),map_location='cpu',weights_only=False)
    assert state['rank']==w.rank and state['world_size']==w.world_size==8
    assert state['global_step']==2 and state['scheduler']['last_epoch']==2
    current=[(n,list(p.shape),str(p.dtype)) for n,p in w.actor_module_fsdp.named_parameters()]
    assert current==state['parameter_shapes'],'FSDP local layout differs'
    assert set(state['model'])=={n for n,p in w.actor_module_fsdp.named_parameters()}
    with torch.no_grad():
        for name,param in w.actor_module_fsdp.named_parameters():
            saved=state['model'][name]
            assert saved.shape==param.shape and saved.dtype==param.dtype and torch.isfinite(saved).all()
            param.copy_(saved.to(param.device))
            if hasattr(param, '_local_shard'):
                assert param._local_shard.shape == saved.shape
                param._local_shard.copy_(saved.to(param._local_shard.device))
                assert torch.equal(param._local_shard.cpu(), saved)
            assert torch.equal(param.detach().cpu(),saved)
    w.actor_optimizer.load_state_dict(state['optimizer'])
    w.actor_lr_scheduler.load_state_dict(state['scheduler'])
    actual=w.actor_optimizer.state_dict()
    assert actual['param_groups']==state['optimizer']['param_groups']
    assert set(actual['state'])==set(state['optimizer']['state'])
    for key,values in actual['state'].items():
        for name,value in values.items():
            saved=state['optimizer']['state'][key][name]
            if torch.is_tensor(value):
                assert torch.isfinite(value).all() and torch.equal(value.cpu(),saved.cpu())
            else:assert value==saved
    assert w.actor_lr_scheduler.get_last_lr()==[g['lr'] for g in w.actor_optimizer.param_groups]
    # load_state_dict follows parameter device. Restore original CPU staging
    # before rollout so all Adam moments do not remain on the training GPU.
    if getattr(w, '_is_offload_optimizer', False):
        from verl.utils.fsdp_utils import offload_fsdp_optimizer
        offload_fsdp_optimizer(w.actor_optimizer)
        torch.cuda.synchronize()
    torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state(state['cuda_rng'])
    np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng'])
    report=dict(rank=w.rank,pid=os.getpid(),completed_steps=2,optimizer_steps=sorted(set(float(v['step']) for v in actual['state'].values())),scheduler=w.actor_lr_scheduler.state_dict(),weights_exact=True,optimizer_exact=True)
    audit=Path(os.environ['E1_BASE'])/'receipts'/('resume-restored-'+os.environ['E1_RUN_ID']+'-rank-'+str(w.rank)+'.json')
    audit.write_text(json.dumps(report,indent=2));return report

def restore_group(group,directory):
    root=Path(directory);manifest=json.loads((root/'full-complete.json').read_text())
    assert manifest['global_step']==2 and manifest['world_size']==8
    workers=group.workers;assert len(workers)==8
    results=ray.get([w.actor_rollout_execute_with_func_generator.remote(restore_worker,str(root)) for w in workers])
    assert sorted(r['rank'] for r in results)==list(range(8))
    print('FULL_STATE_RESTORED_CONTINUE_AT_STEP_3',json.dumps(results),flush=True)
    return 2
