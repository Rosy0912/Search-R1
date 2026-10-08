"""Replays the independent rejected F04 harness against immutable 4ff source.

Actual clip/actor/worker methods, real Gloo and tiny Adam; synthetic FSDP layout.
Owned tiny state files are retained. No GPU/FSDP acceptance claim is made.
"""
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace as N
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

OUT = Path(__file__).resolve().parent
SOURCE = OUT/'immutable-source-4ff'
OLD_HARNESS = OUT.parent/'tree-fresh200-repaired-freeze-review/reproduce_f04_cpu.py'
spec = importlib.util.spec_from_file_location('rejected_review_harness', OLD_HARNESS)
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)
harness.SOURCE = SOURCE
sys.path.insert(0, str(SOURCE))


class Proto:
    def __init__(self, meta_info): self.meta_info = meta_info
    def to(self, device): return self


def rank_main(rank, mode):
    run = OUT/f'independent-f04-{mode}'
    dist.init_process_group('gloo', init_method='file://'+str(run/'gloo-init'), rank=rank, world_size=2)
    from search_r1.baselines import gradient_norm
    try:
        env = dict(SLURM_JOB_ID='owned-local-review', SEARCH_R1_BASELINE_RUN_ID='independent-f04-'+mode,
            SEARCH_R1_BASELINE_COMMIT='4ff6f9dddce24572778124717ce73dfc7b655541',
            SEARCH_R1_BASELINE_CONFIG_SHA256='tiny-cpu-config', SEARCH_R1_BASELINE_DATA_IDENTITY='tiny-cpu-data',
            SEARCH_R1_MODEL_IDENTITY='tiny-cpu-model', SEARCH_R1_BASELINE_OUTPUT=str(run/'output'))
        with patch.dict(os.environ, env):
            model = harness.SyntheticFSDP()
            optimizer = torch.optim.Adam(model.parameters(), lr=.01)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.)
            safety = N(committed=0, sync_valid=lambda *a: None, publish=lambda *a: None)
            actor = N(actor_module=model, actor_optimizer=optimizer, config=harness.Config(grad_clip=1), _policy_safety=safety)
            step = harness.actual_method('verl/workers/actor/dp_actor.py','DataParallelPPOActor','_optimizer_step',dict(torch=torch))
            native_clip = gradient_norm.clip_grad_norm64
            clip_count = 0
            def observed_clip(module, maximum):
                nonlocal clip_count
                clip_count += 1
                if mode=='postclip' and clip_count==4 and rank==1:
                    module.weight.grad.fill_(float('nan'))
                return native_clip(module, maximum)
            def update_policy(data):
                model.weight.grad = torch.ones_like(model.weight)
                step(actor)
                model.weight.grad = torch.full_like(model.weight, float('nan') if mode=='preclip' and rank==1 else 1.)
                step(actor)
                raise AssertionError('Expected controlled refusal before second optimizer write')
            actor.update_policy = update_policy
            worker = N(actor=actor,actor_optimizer=optimizer,actor_lr_scheduler=scheduler,
                config=harness.Config(actor=harness.Config(fresh200_protocol='tree_fresh200_five_leaf_v1')),
                _is_actor=True,_is_offload_param=False,_is_offload_optimizer=False,
                ulysses_sharding_manager=harness.Sharding())
            identity = {key:env[value] for key,value in dict(job_id='SLURM_JOB_ID',run_id='SEARCH_R1_BASELINE_RUN_ID',
                code_commit='SEARCH_R1_BASELINE_COMMIT',config_sha256='SEARCH_R1_BASELINE_CONFIG_SHA256',
                data_identity='SEARCH_R1_BASELINE_DATA_IDENTITY',model_identity='SEARCH_R1_MODEL_IDENTITY').items()}
            identity.update(training_protocol='tree_fresh200_five_leaf_v1',attempted_outer=1,completed_outer=0)
            data = N(batch=harness.TinyBatch(),meta_info=dict(baseline_outer_step=1,
                tree_partial_context=dict(protocol='tree_partial_state_v1',identity=identity,
                    directory=str(run/'training_state/partial_outer_000001'))))
            data.to = lambda device:data
            update = harness.actual_method('verl/workers/fsdp_workers.py','ActorRolloutRefWorker','update_actor',
                dict(torch=torch,os=os,DataProto=Proto,Timer=harness.Timer,logger=None,
                    log_gpu_memory_usage=lambda *a,**k:None))
            before_rng = torch.get_rng_state().clone()
            with patch.object(gradient_norm,'FSDP',harness.SyntheticFSDP), \
                 patch('torch.distributed.fsdp._runtime_utils._lazy_init',lambda *a:None), \
                 patch.object(gradient_norm,'clip_grad_norm64',observed_clip), \
                 patch.object(torch.cuda,'empty_cache',lambda:None):
                result = update(worker,data)
            envelope = result.meta_info['tree_partial_state']
            own = envelope['worker_states'][rank]
            saved_path = Path(own['path'])
            saved = torch.load(saved_path,map_location='cpu',weights_only=False)
            assert envelope['status']=='PARTIAL_WORKER_STATE_SAVED'
            assert envelope['outer_completed'] is False and envelope['automatic_rollback'] is False
            assert envelope['resume_permitted'] is False
            assert envelope['scheduler_advanced_actual']==[False,False]
            assert actor._actual_committed_optimizer_calls==1
            assert int(next(iter(optimizer.state.values()))['step'])==1
            assert model.weight.item()!=1. and scheduler.last_epoch==0
            assert saved['committed_optimizer_calls']==1 and saved['scheduler']['last_epoch']==0
            assert torch.equal(saved['parameters']['weight'],model.weight.detach())
            assert int(next(iter(saved['optimizer']['state'].values()))['step'])==1
            torch.testing.assert_close(saved['gradients']['weight'],model.weight.grad,equal_nan=True,rtol=0,atol=0)
            assert bool(torch.isnan(saved['gradients']['weight']).any()) is (rank==1)
            assert saved['state_finite'] is (rank==0)
            assert torch.equal(saved['rng']['torch'],before_rng) and torch.equal(torch.get_rng_state(),before_rng)
            assert hashlib.sha256(saved_path.read_bytes()).hexdigest()==own['sha256']
            assert result.meta_info['metrics']['actor/optimizer_steps']==1
            assert result.meta_info['metrics']['actor/partial_outer_refused']==1
            record = dict(rank=rank,mode=mode,actual_optimizer_returns=1,actual_adam_step=1,
                actual_weight=model.weight.item(),scheduler_last_epoch=0,saved_weight=saved['parameters']['weight'].item(),
                saved_gradient_has_nan=bool(torch.isnan(saved['gradients']['weight']).any()),
                snapshot_sha256=own['sha256'],state_finite=saved['state_finite'],rng_unchanged=True,
                partial_envelope=envelope,clip_count=clip_count,
                scope='Independent rejected harness replay: actual methods, real tiny CPU Adam/Gloo, synthetic FSDP layout/lazy-init bypass; no GPU/model asset execution')
            (run/f'RANK_{rank}.json').write_text(json.dumps(record,indent=2,allow_nan=False))
            print(json.dumps({key:record[key] for key in ('rank','mode','actual_optimizer_returns','actual_adam_step',
                'actual_weight','scheduler_last_epoch','saved_gradient_has_nan','state_finite','snapshot_sha256')},allow_nan=False),flush=True)
    finally:
        dist.destroy_process_group()


if __name__=='__main__':
    for mode in ('preclip','postclip'):
        run = OUT/f'independent-f04-{mode}'
        run.mkdir(exist_ok=False)
        mp.spawn(rank_main,args=(mode,),nprocs=2,join=True)
        records = [json.loads((run/f'RANK_{rank}.json').read_bytes()) for rank in (0,1)]
        assert records[0]['partial_envelope']==records[1]['partial_envelope']
        (run/'ENVELOPE_IDENTICAL.json').write_text(json.dumps(dict(mode=mode,envelopes_identical=True,
            outer_completed=False,automatic_rollback=False,actual_optimizer_calls=[1,1])))
