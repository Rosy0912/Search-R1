"""User-requested isolated 16-question/80-row native iteration, not formal training.

Run against either the immutable historical OPSD checkout or repaired B checkout.
Production processes, optimizers, checkpoints and sources are never attached.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import time

def emit(stage, **values):
    print('COST80='+json.dumps(dict(stage=stage, unix_time=time.time(), pid=os.getpid(),
          case=os.environ['COST80_CASE'], **values), allow_nan=False), flush=True)

@contextmanager
def stage(name):
    start=time.perf_counter(); cpu=time.process_time(); ok=False
    emit(name+'_start')
    try:
        yield
        ok=True
    finally:
        emit(name+'_complete', successful=ok, wall_seconds=time.perf_counter()-start,
             cpu_seconds=time.process_time()-cpu)

def timed_cuda(worker, name, fn):
    import torch
    def call(*args, **kwargs):
        begin=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
        started=time.perf_counter(); stamp=time.time(); begin.record()
        ok=False
        try:
            result=fn(*args, **kwargs); ok=True
            return result
        finally:
            end.record()
            worker._cost_events.append((name, begin, end, time.perf_counter()-started, ok))
            emit(name, rank=worker.rank, start_unix=stamp, end_unix=time.time(),
                 host_seconds=time.perf_counter()-started, successful=ok,
                 host_scope='native asynchronous call; CUDA events resolved at iteration end')
    return call

def worker_type():
    import torch
    from verl.workers.fsdp_workers import ActorRolloutRefWorker
    from verl.single_controller.base.decorator import register, Dispatch
    class ProfileWorker(ActorRolloutRefWorker):
        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def init_model(self):
            with stage('worker_model_load'):
                result=super().init_model()
            self._cost_events=[]
            if os.environ['COST80_CASE']=='repaired_b':
                manager=self.rollout_sharding_manager
                original=manager.prepare_policy_cache
                # This isolated worker uses the production cache implementation.
                # Its fresh diagnostic state has no production resume binding.
                manager.prepare_policy_cache=lambda enabled, version: original(True,
                    (self.actor._fresh_total_committed_optimizer_calls,
                     self.actor_lr_scheduler.last_epoch))
            self.actor.actor_module.forward=timed_cuda(self,'student_forward',self.actor.actor_module.forward)
            teacher=getattr(self.actor,'teacher',None)
            if teacher is not None:
                teacher.model.base_model.forward=timed_cuda(self,'teacher_backbone_forward',teacher.model.base_model.forward)
                head=teacher.model.get_output_embeddings()
                head.forward=timed_cuda(self,'teacher_vocab_forward',head.forward)
            torch.Tensor.backward=timed_cuda(self,'backward',torch.Tensor.backward)
            self.actor_optimizer.step=timed_cuda(self,'adam_step',self.actor_optimizer.step)
            torch.cuda.reset_peak_memory_stats()
            return result

        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def cost_report(self):
            torch.cuda.synchronize()
            totals={}
            for name, begin, end, host, ok in self._cost_events:
                row=totals.setdefault(name,dict(calls=0,host_seconds=0.,cuda_seconds=0.,successful=True))
                row['calls']+=1; row['host_seconds']+=host
                row['cuda_seconds']+=begin.elapsed_time(end)/1000.; row['successful'] &= ok
            return dict(rank=self.rank, pid=os.getpid(), stages=totals,
                adam_steps=sorted({int(v['step']) for v in self.actor_optimizer.state.values()}),
                scheduler_step=self.actor_lr_scheduler.last_epoch,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                cuda_scope='current-stream events; forward/backward stages may be nested; do not sum nested stages')
    return ProfileWorker

def run_driver(config):
    import ray
    import torch
    import numpy as np
    from verl import DataProto
    from verl.utils import hf_tokenizer
    from verl.single_controller.ray import RayWorkerGroup
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role, reduce_metrics
    from verl.trainer.main_ppo import RewardManager
    import search_r1.baselines.controller as controller
    from search_r1.llm_agent.generation import GenerationConfig, LLMGenerationManager
    from search_r1.baselines.rollout import select_rows
    case=os.environ['COST80_CASE']; output=Path(os.environ['COST80_OUTPUT'])
    assert config.data.train_batch_size==16 and config.trainer.n_gpus_per_node==4
    assert config.max_turns==4 and config.actor_rollout_ref.rollout.temperature==1.
    assert config.actor_rollout_ref.actor.optim.lr==1e-6
    # Production's fixed 512/world4/20-opportunity admission is inapplicable to
    # the explicitly requested standalone 80-row diagnostic. Native numerical
    # guards, raw IDs, mu validation, loss and recursive selection stay enabled.
    original_validator=controller.validate_config
    if case=='repaired_b':
        assert config.baselines.tree.branch_sampling=='recursive_inverse_generation_v1'
        assert config.baselines.tree.continuation_budget_per_question==10
        assert config.baselines.tree.behavior_mu.enabled
        assert config.actor_rollout_ref.actor.policy_safety.enabled
        controller.validate_config=lambda cfg: None
    worker=worker_type()
    pool=ResourcePoolManager({'profile':[4]}, {Role.ActorRollout:'profile'})
    try:
        with stage('driver_setup'):
            tokenizer=hf_tokenizer(config.actor_rollout_ref.model.path)
            trainer=RayPPOTrainer(config,tokenizer,{Role.ActorRollout:ray.remote(worker)},pool,
                RayWorkerGroup,RewardManager(tokenizer,0),None)
    finally:
        controller.validate_config=original_validator
    with stage('initialize_workers'):
        trainer.init_workers()
    trainer.global_steps=1
    with stage('fetch_same_16_questions'):
        batch=DataProto.from_single_dict(next(iter(trainer.train_dataloader)))
        question_ids=batch.non_tensor_batch['index'].tolist()
        emit('question_identity',count=len(batch),sha256=hashlib.sha256(json.dumps(question_ids,sort_keys=True).encode()).hexdigest())
    manager=LLMGenerationManager(tokenizer,trainer.actor_rollout_wg,GenerationConfig(
        max_turns=4,max_start_length=2048,max_prompt_length=4096,max_response_length=500,
        max_obs_length=500,num_gpus=4,no_think_rl=config.algorithm.no_think_rl,
        search_url=config.retriever.url,topk=config.retriever.topk,force_final_answer=False))
    metrics={}; manager.timing_raw={}; started=time.perf_counter()
    batch=batch.repeat(repeat_times=config.actor_rollout_ref.rollout.n_agent,interleave=True)
    gen=batch.pop(batch_keys=['input_ids','attention_mask','position_ids'])
    with stage('sampling_and_compose'):
        generated,sources=trainer.baseline_controller.generate(manager,gen,batch,metrics)
    with stage('prepare_old_probability'):
        if case=='repaired_b':
            trainer.actor_rollout_wg.release_rollout_policy_cache()
            generated.meta_info['tree_full_actor_beta_available']=False
            from verl.trainer.ppo.ray_trainer import prepare_behavior_mu_old_policy
            from search_r1.baselines.tree_behavior_mu import BehaviorMuProofCache
            proofs=BehaviorMuProofCache()
            prepare_behavior_mu_old_policy(generated,config.actor_rollout_ref.actor,proofs)
        else:
            generated=generated.union(trainer.actor_rollout_wg.compute_log_prob(generated))
        batch=select_rows(batch,sources).union(generated)
        batch.non_tensor_batch['uid']=batch.non_tensor_batch['index'].copy()
        assert len(batch)==80, 'Requested 16x5 training batch was not actually produced'
    with stage('balance_reward_advantage_mask'):
        trainer._balance_batch(batch,metrics)
        width=batch.batch['responses'].shape[-1]
        if case=='repaired_b':
            from verl.trainer.ppo.ray_trainer import validate_final_behavior_mu_batch
            validate_final_behavior_mu_batch(batch,config.actor_rollout_ref.actor,proofs)
            batch.batch['ref_log_prob']=torch.zeros_like(batch.batch['sampled_old_log_probs'])
        batch.batch['token_level_scores']=trainer.reward_fn(batch)
        batch.batch['token_level_rewards']=batch.batch['token_level_scores']
        batch=trainer.baseline_controller.compute_advantages(batch,metrics)
        batch,metrics=trainer._create_loss_mask(batch,metrics)
        batch.meta_info.update(global_token_num=batch.batch['attention_mask'].sum(-1).tolist(),baseline_outer_step=1)
        if case=='repaired_b':
            validate_final_behavior_mu_batch(batch,config.actor_rollout_ref.actor,proofs)
            trainer.baseline_controller.audit_actor_batch(batch)
    with stage('native_actor_update_total'):
        updated=trainer.actor_rollout_wg.update_actor(batch)
        metrics.update(reduce_metrics(updated.meta_info['metrics']))
    assert metrics['actor/optimizer_steps']>=1, 'No actual Adam update occurred'
    with stage('native_checkpoint_save'):
        trainer.actor_rollout_wg.save_checkpoint(str(output/'checkpoint'),None)
    reports=trainer.actor_rollout_wg.cost_report()
    result=dict(case=case,completed=True,questions=16,training_rows=len(batch),
        iteration_wall_seconds=time.perf_counter()-started,metrics=metrics,workers=reports,
        source_commit=os.environ['SEARCH_R1_BASELINE_COMMIT'],diagnostic_only=True,
        limits='Four GPUs 0-3; one small batch; cold setup separate; not formal 512x5 throughput acceptance')
    (output/'result.json').write_text(json.dumps(result,default=lambda x:float(x),indent=2))
    emit('iteration_complete',**result)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--case',choices=['old_opsd','repaired_b'],required=True);p.add_argument('--output',required=True)
    a=p.parse_args(); output=Path(a.output);output.mkdir(exist_ok=False)
    os.environ.update(COST80_CASE=a.case,COST80_OUTPUT=str(output))
    from omegaconf import OmegaConf
    import ray
    config=OmegaConf.load(a.config)
    config.data.train_batch_size=16;config.data.shuffle_train_dataloader=False
    config.trainer.n_gpus_per_node=4;config.trainer.total_training_steps=1
    config.trainer.training_state_enabled=False;config.trainer.val_before_train=False
    config.trainer.default_local_dir=str(output/'checkpoint')
    config.actor_rollout_ref.actor.training_state_enabled=False
    config.actor_rollout_ref.actor.fresh200_protocol=None
    config.actor_rollout_ref.actor.guard_continue_protocol=None
    config.actor_rollout_ref.actor.optim.total_training_steps=1
    config.baselines.training_protocol='user_cost80_diagnostic_v1'
    config.actor_rollout_ref.actor.ppo_micro_batch_size=16 if a.case=='repaired_b' else 4
    config.actor_rollout_ref.rollout.log_prob_micro_batch_size=32
    config.actor_rollout_ref.ref.log_prob_micro_batch_size=32
    config.actor_rollout_ref.actor.policy_safety.monitor_rows_per_rank=16
    if a.case=='old_opsd':
        config.baselines.method='opsd';config.actor_rollout_ref.actor.baseline_method='opsd'
        config.actor_rollout_ref.actor.use_kl_loss=False;config.actor_rollout_ref.actor.policy_safety.enabled=False
        config.actor_rollout_ref.actor.recursive_document_objective=False
        config.actor_rollout_ref.rollout.tree_raw_emission=False
        config.actor_rollout_ref.rollout.n_agent=5
        config.baselines.distillation.teacher_path=config.actor_rollout_ref.model.path
        config.baselines.distillation.privileged_answers=True
        config.baselines.distillation.teacher_device='cuda'
        config.baselines.distillation.component_clip=None
        config.actor_rollout_ref.actor.distillation=config.baselines.distillation
        config.baselines.tree.behavior_mu.enabled=False
        config.baselines.tree.token_preservation.enabled=False
        config.actor_rollout_ref.actor.policy_safety.sampling_probability_correction.enabled=False
        config.baselines.opsd_e1={'enabled':False}
    else:
        os.environ.update(RECURSIVE_B_CACHE_DETOKENIZER_LENGTH='1',RECURSIVE_B_OMIT_FULL_BETA='1',TREE_LIGHTWEIGHT_CANDIDATE_REWARD='1')
    text=OmegaConf.to_yaml(config);(output/'resolved.yaml').write_text(text)
    os.environ['SEARCH_R1_BASELINE_CONFIG_SHA256']=hashlib.sha256(text.encode()).hexdigest()
    env={k:v for k,v in os.environ.items() if k.startswith(('COST80_','SEARCH_R1_','RECURSIVE_B_','TREE_','SLURM_'))}
    ray.init(num_gpus=4,runtime_env={'env_vars':env},include_dashboard=False)
    try:
        ray.get(ray.remote(run_driver).remote(config))
    finally:
        ray.shutdown()

if __name__=='__main__':main()
