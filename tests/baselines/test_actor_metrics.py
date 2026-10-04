"""Real Gloo workers + real DataProto concat, global objective and clocks."""
import copy
import json
import torch
import pytest
from verl import DataProto
from search_r1.baselines.actor_metrics import (
    merge_actor_metric_records, aggregate_sequence_mean_actor_metrics,
    validate_actor_metric_topology)
from search_r1.baselines.objectives import weighted_policy_loss


def record(rank):
    return {'actor/pg_loss': [float(rank + 1)] * 20,
            'actor/kl_loss': [float(rank + 2)] * 20,
            'actor/entropy_loss': [float(rank + 3)] * 20,
            'actor/ppo_kl': [float(rank + 4)] * 20,
            'actor/pg_clipfrac': [rank * .1] * 20,
            'actor/grad_norm': [7.] * 20, 'bapo/grad_norm_min': 7., 'bapo/grad_norm_max': 7.,
            'actor/optimizer_steps': 20, 'actor/skipped_empty_minibatches': 0,
            'actor/lr': 1e-6, 'actor/lr_before_outer': 1e-6,
            'actor/scheduler_outer_steps': 80, 'actor/kl_coef': .001,
            'bapo/loss_tokens': (rank + 1) * 11,
            'bapo/old_logprob_nonfinite': 0, 'bapo/new_logprob_nonfinite': 0,
            'mfu/actor': .2 + rank * .1}


def test_eight_rank_metrics_not_first_rank_or_sum_clocks():
    originals = [record(i) for i in range(8)]
    before = copy.deepcopy(originals)
    merged = merge_actor_metric_records(originals)
    assert merged['actor/pg_loss'] == [4.5] * 20
    assert merged['actor/kl_loss'] == [5.5] * 20
    assert merged['bapo/loss_tokens'] == 11 * 36
    assert merged['actor/optimizer_steps'] == 20
    assert merged['actor/scheduler_outer_steps'] == 80
    assert merged['actor/grad_norm'] == [7.] * 20
    assert originals == before


@pytest.mark.parametrize('key,value,match',[
    ('actor/optimizer_steps',19,'Replicated'), ('actor/lr',2e-6,'Replicated'),
    ('actor/grad_norm',[8.]*20,'Replicated'), ('actor/pg_loss',[1.]*19,'mini counts'),
    ('actor/pg_loss',[float('nan')]*20,'finite'),
])
def test_fail_loudly_on_misleading_rank_metrics(key,value,match):
    records = [record(0), record(1)]
    records[1][key] = value
    with pytest.raises(ValueError, match=match):
        merge_actor_metric_records(records)


def test_unclassified_or_mismatched_metrics_rejected():
    records=[record(0), record(1)]
    records[1]['future/ambiguous_metric']=1
    with pytest.raises(ValueError,match='keys disagree'):
        merge_actor_metric_records(records)
    records[0]['future/ambiguous_metric']=1
    with pytest.raises(ValueError,match='Unclassified'):
        merge_actor_metric_records(records)


@pytest.mark.parametrize('sp,tp',[(2,1),(1,2)])
def test_unapproved_parallel_topology_rejected_before_update(sp,tp):
    with pytest.raises(ValueError,match='TP1.*SP1'):
        validate_actor_metric_topology(sp,tp)


def gloo_worker(rank,rendezvous,output):
    torch.distributed.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2)
    # Uneven rows: the local objective already uses global_rows/world.
    values = [1.] if rank == 0 else [2.,3.,4.]
    a=torch.tensor(values,dtype=torch.float64)[:,None]
    loss,cf,kl=weighted_policy_loss(torch.zeros_like(a),torch.zeros_like(a),
                                  a,torch.ones_like(a),torch.ones_like(a),.2,torch.tensor(2.))
    local=record(rank)
    local['actor/pg_loss']=[loss.item()]*20
    local['actor/pg_clipfrac']=[cf.item()*len(values)/2.]*20
    local['actor/ppo_kl']=[kl.item()*len(values)/2.]*20
    merged=aggregate_sequence_mean_actor_metrics(local)
    # Test the actual dangerous concat route, not an imaginary meta reducer.
    objs=[None,None]
    torch.distributed.all_gather_object(objs,DataProto(meta_info={'metrics':merged}))
    concatenated=DataProto.concat(objs)
    m=concatenated.meta_info['metrics']
    assert m['actor/pg_loss']==[-2.5]*20 # mean of all four trajectories
    assert m['actor/optimizer_steps']==20 # never 40, never 160
    assert m['bapo/loss_tokens']==33
    assert m['actor/grad_norm']==[7.]*20
    assert objs[0].meta_info==objs[1].meta_info
    assert local['actor/pg_loss']==([-.5]*20 if rank==0 else [-4.5]*20)
    # Empty-rank local numerator still contributes zero to full denominator.
    local['actor/pg_loss']=[0. if rank==0 else -5.]*20
    assert aggregate_sequence_mean_actor_metrics(local)['actor/pg_loss']==[-2.5]*20
    # Collective gathers first, so every rank rejects inconsistent clocks.
    local['actor/scheduler_outer_steps']=80+rank
    with pytest.raises(ValueError,match='Replicated'):
        aggregate_sequence_mean_actor_metrics(local)
    torch.distributed.destroy_process_group()
    with open(output+str(rank)+'.json','w') as f:json.dump(m,f)


def test_real_gloo_dataproto_concat_with_uneven_rows(tmp_path):
    torch.multiprocessing.spawn(gloo_worker,args=(str(tmp_path/'rendezvous'),str(tmp_path/'rank')),nprocs=2,join=True)
    assert json.loads((tmp_path/'rank0.json').read_text()) == json.loads((tmp_path/'rank1.json').read_text())


def test_actual_worker_hook_preserves_local_record_and_global_return(monkeypatch,capsys):
    # Execute the production worker update method with CPU CUDA shims, not an
    # AST assertion. The actor itself is covered by sequence-mean actor tests.
    from types import SimpleNamespace
    from omegaconf import OmegaConf
    from tensordict import TensorDictBase
    import verl.workers.fsdp_workers as worker_module
    monkeypatch.setattr(torch.cuda,'current_device',lambda:0)
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    monkeypatch.setattr(DataProto,'to',lambda self,*args,**kwargs:self)
    monkeypatch.setattr(TensorDictBase,'cuda',lambda self,*args,**kwargs:self)
    monkeypatch.setattr(worker_module,'log_gpu_memory_usage',lambda *args,**kwargs:None)
    import search_r1.baselines.actor_metrics as reporter
    local=record(0)
    seen=[]
    def aggregate(metrics,sp,tp):
        seen.append(copy.deepcopy(metrics))
        return merge_actor_metric_records([metrics,dict(metrics,**{'actor/pg_loss':[3.]*20})])
    monkeypatch.setattr(reporter,'aggregate_sequence_mean_actor_metrics',aggregate)
    class Sharding:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def preprocess_data(self,data):return data
        def postprocess_data(self,data):return data
    class Scheduler:
        last_epoch=79
        def get_last_lr(self):return [1e-6]
        def step(self):self.last_epoch+=1
    instance=SimpleNamespace(_is_actor=True,_is_offload_param=False,_is_offload_optimizer=False,
       config=OmegaConf.create({'actor':{'baseline_method':'bapo','loss_reduction':'sequence_mean','ppo_epochs':1},
                               'rollout':{'tensor_model_parallel_size':1}}),
       ulysses_sequence_parallel_size=1,ulysses_sharding_manager=Sharding(),world_size=2,rank=0,
       actor=SimpleNamespace(update_policy=lambda data:copy.deepcopy(local)),actor_lr_scheduler=Scheduler(),
       flops_counter=SimpleNamespace(estimate_flops=lambda tokens,seconds:(1.,10.)))
    data=DataProto.from_dict({'responses':torch.zeros(1,1)},meta_info={'global_token_num':[1]})
    out=worker_module.ActorRolloutRefWorker.update_actor(instance,data)
    printed=capsys.readouterr().out
    original=json.loads(printed.split('ACTOR_RANK_UPDATE_JSON=',1)[1].splitlines()[0])
    assert original['metrics']['actor/pg_loss']==[1.]*20
    assert out.meta_info['metrics']['actor/pg_loss']==[2.]*20
    assert len(seen)==1 and out.meta_info['metrics']['actor/optimizer_steps']==20
    assert instance.actor_lr_scheduler.last_epoch==80
    # Unsupported topology rejected before actor or scheduler can update.
    instance.config.rollout.tensor_model_parallel_size=2
    with pytest.raises(ValueError,match='TP1.*SP1'):
        worker_module.ActorRolloutRefWorker.update_actor(instance,data)
    assert instance.actor_lr_scheduler.last_epoch==80
