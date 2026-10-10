#!/usr/bin/env python3
"""Generate the explicit new protocol; does not launch or overwrite an active run."""
import argparse,json,pathlib

def configure(c):
    c['mode']='topd200';c['expected_outer_steps']=200
    c['final_checkpoint']=c['final_checkpoint'].replace('global_step_300','global_step_200')
    changes={'trainer.n_gpus_per_node':'4','trainer.total_training_steps':'201',
             'actor_rollout_ref.actor.ppo_micro_batch_size':'16',
             'actor_rollout_ref.actor.use_kl_loss':'false',
             'actor_rollout_ref.actor.kl_loss_coef':'0','algorithm.kl_ctrl.kl_coef':'0',
             'baselines.bapo.stage_validation_num':'512',
             'baselines.topd.teacher_backend':'local_gpu_bf16_tp4_prompt_logprobs'}
    values=[];seen=set()
    for raw in c['training_overrides']:
        key,old=raw.split('=',1);plain=key.lstrip('+')
        if plain in seen:raise ValueError('Duplicate override '+plain)
        seen.add(plain);values.append(key+'='+changes.get(plain,old))
    if not set(changes)<=seen:raise ValueError('Required override absent: '+str(set(changes)-seen))
    c['training_overrides']=values
    c['required_protocol']={'gpus':4,'outer_iterations':200,'rows_per_rank':640,
       'mini_rows_per_rank':32,'actor_micro_per_rank':4,'old_ref_micro_per_rank':8,
       'gradient_accumulation':8,'optimizer_steps_per_rank_per_outer':20,
       'micro_forwards_per_rank_per_outer':160,'scheduler_steps_per_outer':1,
       'checkpoint_steps':[5,50,100,150,200],'fresh_initialization':True}
    c['preparation_status']='PROTOCOL_DRAFT_REQUIRES_FOUR_GPU_RUNTIME_GATES_AND_ACTUAL_VALIDATION'
    return c

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('output');a=p.parse_args()
    result=configure(json.loads(pathlib.Path(a.input).read_text()))
    with open(a.output,'x') as f:json.dump(result,f,indent=2);f.write('\n')
