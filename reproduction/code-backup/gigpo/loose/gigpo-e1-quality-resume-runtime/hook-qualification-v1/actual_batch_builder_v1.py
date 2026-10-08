"""Concrete full512x5 source RPC pipeline for the future diagnostic adapter.

No Adam, added model forward, source monkey-patch, retokenization or advantage
rewrite occurs here. The original controller may be the baseline A controller
or the independently bound B metadata generate hook. Admission/owned bootstrap
must occur in the unlaunched enclosing adapter before a trainer exists.
"""
from pathlib import Path
import numpy as np
import torch
from qualification_contract_v1 import require
from verl import DataProto
from search_r1.baselines.rollout import select_rows
from search_r1.llm_agent.generation import GenerationConfig,LLMGenerationManager
from search_r1.baselines.trajectory_records import RolloutTrace,record_scored_batch
from search_r1.baselines.recovery import atomic_torch_save
from calibration_state_v1 import state_digest


def build_actual_outer(trainer,training_iterator,phase_directory,label,binding,generate):
    require(binding['status']=='REVIEWED_FOR_EXTERNAL_HOOK_ONLY' and label in ('A','B'),
            'Null/unreviewed diagnostic binding cannot fetch actual training data')
    require(trainer.recovery_sampler.cursor==12800 and trainer.global_steps==25,
            'Each actual A/B begins at complete25 data and logical boundary')
    config=trainer.config
    require(config.actor_rollout_ref.actor.ppo_mini_batch_size==256
            and config.actor_rollout_ref.actor.ppo_micro_batch_size==16
            and config.actor_rollout_ref.rollout.n_agent==5 and config.actor_rollout_ref.rollout.n==1
            and config.actor_rollout_ref.actor.policy_safety.mode=='calibration',
            'Diagnostic mini/micro/five-response/numeric-only scope differs')
    trainer.global_steps=26
    batch=DataProto.from_single_dict(next(training_iterator))
    require(len(batch)==512 and trainer.recovery_sampler.cursor==13312,'Incomplete actual512 fetch/cursor')
    batch=batch.repeat(5,interleave=True)
    # Exact frozen source52 fit assignment. Baseline A calls the same explicit
    # audit-only v3 normalization; B calls the v3 probability generate wrapper.
    # No DataProto reconstruction occurs before that compatibility entry.
    batch.non_tensor_batch['audit_trajectory_row']=np.arange(2560,dtype=np.int64)
    gen_batch=batch.pop(batch_keys=['input_ids','attention_mask','position_ids'])
    manager=LLMGenerationManager(trainer.tokenizer,trainer.actor_rollout_wg,
        GenerationConfig(max_turns=4,max_start_length=2048,max_prompt_length=4096,
            max_response_length=500,max_obs_length=500,num_gpus=8,
            no_think_rl=config.algorithm.no_think_rl,search_url=config.retriever.url,
            topk=3,force_final_answer=config.force_final_answer))
    directory=Path(phase_directory)
    manager.rollout_trace=RolloutTrace(directory/'trajectories',26,batch.non_tensor_batch['index'].tolist(),5)
    metrics={}
    generated,sources=generate(manager,gen_batch,batch,metrics)
    require(len(generated)==2560,'Incomplete actual2560 responses')
    batch=select_rows(batch,sources)
    for key in generated.batch.keys():generated.batch[key]=generated.batch[key].long()
    with torch.no_grad():generated=generated.union(trainer.actor_rollout_wg.compute_log_prob(generated))
    batch.non_tensor_batch['uid']=batch.non_tensor_batch['index'].copy()
    batch=batch.union(generated)
    trainer._balance_batch(batch,metrics=metrics)
    batch.meta_info['global_token_num']=torch.sum(batch.batch['attention_mask'],dim=-1).tolist()
    for key in batch.batch.keys():
        if key!='old_log_probs':batch.batch[key]=batch.batch[key].long()
    batch=batch.union(trainer.ref_policy_wg.compute_ref_log_prob(batch))
    batch.batch['token_level_scores']=trainer.reward_fn(batch)
    require(config.actor_rollout_ref.actor.use_kl_loss,'Fixed-reference loss missing')
    batch.batch['token_level_rewards']=batch.batch['token_level_scores']
    batch=trainer.baseline_controller.compute_advantages(batch,metrics)
    batch,metrics=trainer._create_loss_mask(batch,metrics)
    ids=[int(value) for value in batch.non_tensor_batch['audit_trajectory_row']]
    require(len(ids)==2560 and set(ids)==set(range(2560)),'Actual balanced trajectory provenance differs')
    by_source={original:index for index,original in enumerate(ids)}
    questions=[batch.non_tensor_batch['index'][by_source[source]] for source in range(2560)]
    require(len(set(questions))==512 and all(len(set(questions[group*5:group*5+5]))==1 for group in range(512)),
            'Actual512 question/five-response groups differ')
    require(batch.non_tensor_batch['uid'].tolist()==batch.non_tensor_batch['index'].tolist(),
            'Actual GiGPO question grouping identity differs')
    batch.meta_info['baseline_outer_step']=26
    record_scored_batch(directory/'trajectories',26,batch,trainer.tokenizer)
    witness=atomic_torch_save(dict(tensors={key:value.detach().cpu().clone() for key,value in batch.batch.items()},
        non_tensors=batch.non_tensor_batch,metadata=batch.meta_info,
        data_cursor=trainer.recovery_sampler.state_dict(),label=label,
        retained_goal_contribution=0),directory/'actual-full2560-scored.pt')
    # Do not substitute audit UUIDs. Keep all physical raw metadata while using
    # an explicit mathematical identity view for the A/B equality decision.
    # Nested records remain physical-only because they include fresh node UUIDs.
    math_non_tensors={key:value for key,value in batch.non_tensor_batch.items()
                      if key not in ('trajectory_uid','tree_uid','records')}
    mathematical_identity=state_digest(dict(tensors=dict(batch.batch.items()),non_tensors=math_non_tensors))
    return batch,dict(actual_questions=512,actual_trajectories=2560,data_cursor=13312,
        five_response_groups_verified=True,physical_witness=witness,mathematical_identity=mathematical_identity,
        audit_UUIDs_and_record_nodes_are_preserved_but_not_bitexact=True,
        UUID_excluded_from_model_and_GiGPO_loss=False, # Actual independent consumer proof still pending.
        retained_goal_contribution=0,metrics=metrics)
