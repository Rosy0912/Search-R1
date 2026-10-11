"""Inference only: existing BAPO validation protocol, eight rollout ranks, no optimizer."""
import json,os,time
from pathlib import Path
import ray
from omegaconf import OmegaConf
from types import SimpleNamespace
from verl.workers.fsdp_workers import ActorRolloutRefWorker

@ray.remote
def evaluate(config,run):
    from torch.utils.data import DataLoader
    from verl.utils.dataset.rl_dataset import RLHFDataset,collate_fn
    from verl.utils import hf_tokenizer
    from verl.trainer.main_ppo import RewardManager
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer,Role,ResourcePoolManager
    from verl.single_controller.ray import RayWorkerGroup
    tokenizer=hf_tokenizer(config.actor_rollout_ref.model.path)
    dataset=RLHFDataset(config.data.val_files,tokenizer,prompt_key=config.data.prompt_key,
        max_prompt_length=config.data.max_prompt_length,filter_prompts=True,truncation='error')
    assert len(dataset)==51713, f'Unexpected final dataset size {len(dataset)}'
    loader=DataLoader(dataset,batch_size=config.data.val_batch_size,shuffle=False,drop_last=False,collate_fn=collate_fn)
    def batches():
        done=0
        for i,batch in enumerate(loader):
            start=time.monotonic()
            yield batch
            done+=len(batch['input_ids'])
            record=dict(batch=i+1,completed=done,total=len(dataset),batch_size=len(batch['input_ids']),seconds=time.monotonic()-start,time=time.time())
            with (Path(run)/'progress.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            print('EVAL150_BATCH_COMPLETE='+json.dumps(record),flush=True)
    t=RayPPOTrainer.__new__(RayPPOTrainer)
    t.config=config;t.tokenizer=tokenizer;t.hybrid_engine=True
    t.use_reference_policy=False;t.use_rm=False
    t.role_worker_mapping={Role.ActorRollout:ray.remote(ActorRolloutRefWorker)}
    t.resource_pool_manager=ResourcePoolManager({'eval':[8]},{Role.ActorRollout:'eval'})
    t.ray_worker_group_cls=RayWorkerGroup
    t.baseline_controller=SimpleNamespace(method='bapo',ranker=None)
    t.val_reward_fn=RewardManager(tokenizer,num_examine=0)
    t.val_dataloader=batches();t.global_steps=150
    # Construct the unchanged worker directly as rollout-only. The colocated
    # factory requires its direct Worker parent, so do not subclass the actor.
    from verl.single_controller.ray import RayClassWithInitArgs
    from verl.single_controller.ray.base import create_colocated_worker_cls
    t.resource_pool_manager.create_resource_pool()
    pool=t.resource_pool_manager.get_resource_pool(Role.ActorRollout)
    cls=RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker),config=config.actor_rollout_ref,role='rollout')
    wg=RayWorkerGroup(resource_pool=pool,ray_cls_with_init=create_colocated_worker_cls({'actor_rollout':cls}))
    t.actor_rollout_wg=wg.spawn(prefix_set={'actor_rollout'})['actor_rollout']
    t.wg_dicts=[wg]
    t.actor_rollout_wg.init_model()
    # Only the validation metadata branch; native training initialization stays disabled.
    config.trainer.fresh200_enabled=True
    result=t._validate()
    assert result['val/qa_count/all']==51713
    (Path(run)/'final51713.json').write_text(json.dumps(dict(checkpoint_outer=150,metrics=result),indent=2))
    return result

def main():
    root=Path('/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v11-20261009')
    old=root/'runs/formal-aligned-em200-16c570e111f947e2b0026983df2f7938-3359'
    run=Path(os.environ['BAPO_RUN_ID'])
    c=OmegaConf.load(old/'resolved-config.yaml')
    c.actor_rollout_ref.model.path=str(old/'checkpoints/actor/global_step_150')
    c.actor_rollout_ref.model.revision=None
    c.actor_rollout_ref.actor.fresh200_enabled=False
    c.trainer.fresh200_enabled=False;c.trainer.fresh200_gate_enabled=False
    c.trainer.resume_full_state=None;c.trainer.resume_permission=None;c.trainer.resume_verify_only=False
    c.data.val_files=c.data.final_val_files;c.data.val_batch_size=512
    c.actor_rollout_ref.rollout.gpu_memory_utilization=0.28
    c.actor_rollout_ref.rollout.max_num_batched_tokens=16384
    c.actor_rollout_ref.rollout.max_num_seqs=128
    c.trainer.default_local_dir=str(run/'unused-training-checkpoints')
    OmegaConf.save(c,run/'eval-config.yaml')
    ray.init(address=os.environ['RAY_ADDRESS'],runtime_env={'env_vars':{'TOKENIZERS_PARALLELISM':'false','NCCL_DEBUG':'WARN'}})
    print(ray.get(evaluate.remote(c,str(run))),flush=True)
if __name__=='__main__':main()
