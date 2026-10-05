"""Native eight-GPU COMPONENT fixture, never production calibration evidence.

Runs the actual actor update loop/FSDP/Adam, with a tiny synthetic forward and
an explicitly replaced threshold admission. No Qwen rollout, EM or GRPO update
is claimed by this fixture. Launch only within a native Slurm GPU allocation.
"""
import os,json,gc,hashlib
from pathlib import Path
from unittest.mock import patch
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from omegaconf import OmegaConf
from verl import DataProto
from verl.workers.actor.dp_actor import DataParallelPPOActor


class Tiny(nn.Module):
    def __init__(self):
        super().__init__();self.linear=nn.Linear(16,16,bias=False)
        nn.init.zeros_(self.linear.weight)
    def forward(self,x):return self.linear(x).sum(-1)/16


class FixtureActor(DataParallelPPOActor):
    def _forward_micro_batch(self,micro_batch,temperature):
        rows=len(micro_batch['responses'])
        value=self.actor_module(torch.ones(rows,16,device=torch.cuda.current_device()))
        logs=value[:,None].expand_as(micro_batch['old_log_probs'])
        return logs*0,logs


def scenario(name,threshold,expected_writes,expected_reject,inject=False):
    base=Path(os.environ['SEARCH_G1_BASE']);rank=dist.get_rank()
    run_id='component-fixture-'+name
    if rank==0:
        (base/'logs/run-manifest.json').write_text(json.dumps(dict(profile='common8',run_id=run_id,
            role='synthetic_native_component_fixture_not_production',policy_safety_required=True)))
        for p in (base/'logs').glob('optimizer-rank-*.jsonl'):p.unlink()
    dist.barrier()
    module=FSDP(Tiny().cuda(),device_id=torch.cuda.current_device(),use_orig_params=False)
    optimizer=torch.optim.Adam(module.parameters(),lr=1e-6)
    config=OmegaConf.create(dict(ppo_mini_batch_size=16,ppo_micro_batch_size=2,ppo_epochs=1,
        use_dynamic_bsz=False,ulysses_sequence_parallel_size=1,state_masking=True,
        use_kl_loss=False,entropy_coeff=0.,clip_ratio=.2,grad_clip=1.,
        policy_safety=dict(enabled=True,dual_clip=3.,target_old_policy_kl=threshold,
            kl_log_bound=30.,monitor_trajectories_per_rank=4,calibration_manifest=None)))
    actor=FixtureActor(config,module,optimizer)
    mask=(torch.arange(4)[None,:]<(torch.arange(320)%4+1)[:,None]).float().cuda()
    old=torch.zeros(320,4,device=torch.cuda.current_device())
    if inject and rank==7:old[0,0]=float('nan')
    data=DataProto.from_dict(tensors=dict(responses=torch.ones(320,4,dtype=torch.long,device=old.device),
        input_ids=torch.ones(320,8,dtype=torch.long,device=old.device),
        attention_mask=torch.ones(320,8,dtype=torch.long,device=old.device),
        position_ids=torch.arange(8,device=old.device).expand(320,-1),loss_mask=mask,
        old_log_probs=old,advantages=torch.ones_like(old)),meta_info=dict(temperature=1.,logical_update=1))
    before=[p.detach().clone() for p in module.parameters()]
    rejected=False;reason=None
    # Explicitly replace only admission for this isolated synthetic fixture.
    with patch('search_g1.policy_safety.bind_thresholds',return_value=(.25,'fixture-not-calibration')):
        try:actor.update_policy(data)
        except RuntimeError as exc:
            rejected=True;reason=str(exc)
    writes=getattr(actor,'_optimizer_writes',0)
    evidence=[json.loads(s) for s in (base/'logs'/f'optimizer-rank-{rank}.jsonl').read_text().splitlines()]
    monitors=[x for x in evidence if x['kind']=='POLICY_MONITOR']
    budgets=[x for x in evidence if x['kind']=='OUTER_BUDGET']
    actual=[x for x in evidence if x['kind']=='OPTIMIZER_WRITE']
    assert writes==expected_writes and len(actual)==writes and rejected==expected_reject
    assert len(budgets)==(0 if rejected else 1)
    assert module.training and all(torch.isfinite(p).all() for p in module.parameters())
    assert all(torch.isfinite(v).all() for s in optimizer.state.values() for v in s.values() if torch.is_tensor(v))
    changed=any(not torch.equal(a,b) for a,b in zip(before,module.parameters()))
    assert changed==(writes>0)
    if not rejected:assert len(monitors)==40 and all(x['allowed'] for x in monitors)
    else:
        assert monitors[-1]['allowed'] is False
        assert monitors[-1]['stage']==('after_minibatch' if writes else 'before_minibatch')
    assert all(x['monitor_identity']==monitors[0]['monitor_identity'] for x in monitors)
    if writes:
        assert monitors[1]['sampled_old_policy_kl']>monitors[0]['sampled_old_policy_kl']
    result=dict(scenario=name,rank=rank,rejected=rejected,reason=reason,actual_optimizer_writes=writes,
        outer_budget_records=len(budgets),monitor_records=len(monitors),
        initial_monitor_kl=monitors[0]['sampled_old_policy_kl'],final_monitor_kl=monitors[-1]['sampled_old_policy_kl'],
        frozen_old_identity_unchanged=True,actual_parameter_change=changed,all_parameters_and_adam_finite=True,
        gradient_norm_before=[x['gradient_norm_before'] for x in actual],
        gradient_norm_after=[x['gradient_norm_after'] for x in actual],production_admission=False)
    (base/'logs'/f'{name}-rank-{rank}.json').write_text(json.dumps(result,indent=2))
    dist.barrier()
    for p in (base/'logs').glob('optimizer-rank-'+str(rank)+'.jsonl'):
        p.rename(base/'logs'/(name+'-'+p.name))
    del actor,optimizer,module;gc.collect();torch.cuda.empty_cache();dist.barrier()


def main():
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SEARCH_G1_COMPONENT_FIXTURE')=='1'
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']));dist.init_process_group('nccl')
    assert dist.get_world_size()==8
    scenario('healthy_synthetic_outer',.01,20,False)
    scenario('rank7_invalid_old',.01,0,True,inject=True)
    scenario('post_write_kl_brake',1e-15,1,True)
    dist.destroy_process_group()


if __name__=='__main__':main()
