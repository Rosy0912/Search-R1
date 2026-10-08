"""Two complete source rollouts + teacher rounds + original actor backward.
Original TOPD advantages/rewards are used; resource scope needs independent audit.
"""
import json,os,time
from pathlib import Path
from contracts import require,release_and_ack
from lifecycle import publish

def run(trainer):
 import torch,numpy as np
 from verl import DataProto
 from search_r1.llm_agent.generation import GenerationConfig,LLMGenerationManager
 from search_r1.baselines.rollout import select_rows
 cfg=trainer.config
 gc=GenerationConfig(max_turns=cfg.max_turns,max_start_length=cfg.data.max_start_length,max_prompt_length=cfg.data.max_prompt_length,max_response_length=cfg.data.max_response_length,max_obs_length=cfg.data.max_obs_length,num_gpus=8,no_think_rl=cfg.algorithm.no_think_rl,search_url=cfg.retriever.url,topk=cfg.retriever.topk,force_final_answer=cfg.get('force_final_answer',True))
 manager=LLMGenerationManager(trainer.tokenizer,trainer.actor_rollout_wg,gc)
 binding=json.loads(Path(os.environ['TOPD32B_BINDING']).read_text());records=[]
 for step in (1,2):
  trainer.global_steps=step;batch=DataProto.from_single_dict(next(iter(trainer.train_dataloader))).repeat(repeat_times=5,interleave=True)
  require(len(batch)==2560,'Profile must retain512x5 rows')
  gen=batch.pop(batch_keys=['input_ids','attention_mask','position_ids']);metrics={};manager.timing_raw={}
  generated,sources=trainer.baseline_controller.generate(manager,gen,batch,metrics)
  batch=select_rows(batch,sources)
  for key in generated.batch:generated.batch[key]=generated.batch[key].long()
  generated=generated.union(trainer.actor_rollout_wg.compute_log_prob(generated));batch=batch.union(generated)
  batch.non_tensor_batch['uid']=batch.non_tensor_batch['index'].copy()
  trainer._balance_batch(batch,metrics);batch.meta_info.update(topd_outer_step=step,temperature=1.,global_token_num=batch.batch['attention_mask'].sum(-1).tolist())
  phases=trainer.actor_rollout_wg.observed_phase()
  require(all(all(k['cache_engine_is_none'] and k['gpu_cache_is_none'] for k in r['kv']) for r in phases),'Actual original KV must be freed before teacher stage')
  began=time.time()
  batch.batch['token_level_scores']=trainer.reward_fn(batch)
  batch.batch['token_level_rewards']=batch.batch['token_level_scores']
  batch=trainer.baseline_controller.compute_advantages(batch,metrics)
  scoring_completed=time.time()
  released,_=release_and_ack(binding['ipc_directory'],step,binding)
  actual_scoring=released['original_scorer_receipt']
  if cfg.do_search and cfg.actor_rollout_ref.actor.state_masking:
   batch,metrics=trainer._create_loss_mask(batch,metrics)
  trainer.baseline_controller.before_actor_update(batch,step)
  result=trainer.actor_rollout_wg.update_actor(batch)
  from verl.trainer.ppo.ray_trainer import reduce_metrics
  actual_metrics=reduce_metrics(result.meta_info['metrics'])
  trainer.baseline_controller.after_actor(batch,actual_metrics)
  mask=batch.batch['attention_mask'][:,-batch.batch['responses'].shape[-1]:]*batch.batch['info_mask'][:,-batch.batch['responses'].shape[-1]:]
  records.append(dict(step=step,teacher_and_adv_wall_seconds=scoring_completed-began,scoring_started_epoch=actual_scoring['started_epoch'],scoring_completed_epoch=actual_scoring['released_epoch'],rows=len(batch),tokens=mask.sum().item(),actor_metrics=result.meta_info['metrics'],kv_released_receipts=phases))
 trainer._save_checkpoint()
 require(len(records)==2,'Incomplete hardware rounds')
 publish(Path(os.environ['TOPD32B_RUN'])/'profile-driver-result.json',dict(status='PROFILE_MODEL_MUTATED_HARDWARE_COMPLETE_UNREVIEWED_NOT_ALGORITHM_TRAINING',records=records,numerical_long_context_source='INDEPENDENT_NATIVE32B_ORIGINAL_GPU_PARITY6144',actual_actor_backward_observed=True,teacher_scoring_kv_released=True,next_rollout_kv_rebuilt_observation_required=True,actor_global_microbatch=64,actor_ranks=8,teacher_fp32_parameter_sha256=binding['fp32_parameter_sha256'],advantages='ORIGINAL_TOPD_FUNCTION_NO_SURROGATE',algorithm_accepted=False))
 return records
