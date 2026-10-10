"""Build explicit CoSearch overrides; input receipts must pass before launch."""
import json
from pathlib import Path
BASE='/data/user/jxu957/experiments/cosearch-continue300-x203-20261008'
TOPD='/data/user/jxu957/experiments/topd-20261008/runtime/assets'
RUN=BASE+'/runtime/cosearch-instruct200-20261010'

def configure(original):
    model=TOPD+'/qwen2.5-3b-instruct-aa8e725'
    values={
      'data.train_files':'['+RUN+'/data/train.parquet]',
      'data.val_files':'['+RUN+'/data/validation.parquet]',
      'data.train_batch_size':'512','data.val_batch_size':'512',
      'data.filter_overlong_prompts':'False',
      'data.dataloader_num_workers':'0',
      'trainer.nnodes':'1','trainer.n_gpus_per_node':'4',
      'trainer.total_training_steps':'200','trainer.total_epochs':'1',
      'trainer.resume_mode':'disable','trainer.default_local_dir':RUN+'/checkpoints',
      'trainer.save_freq':'50','trainer.test_freq':'50',
      'trainer.max_actor_ckpt_to_keep':'5','trainer.val_before_train':'False',
      'trainer.rollout_data_dir':RUN+'/rollouts',
      'trainer.validation_data_dir':RUN+'/validation',
      'trainer.experiment_name':'cosearch-instruct200-20261010',
      'algorithm.use_kl_in_reward':'False',
      'actor_rollout_ref.rollout.n':'5',
    }
    for role in ('actor_rollout_ref','reranker_actor_rollout_ref'):
        for key,value in {
          'model.path':model,'actor.ppo_mini_batch_size':'128',
          'actor.ppo_micro_batch_size_per_gpu':'4',
          'actor.ppo_epochs':'1','actor.ulysses_sequence_parallel_size':'1',
          'actor.optim.lr':'1e-6','actor.optim.lr_warmup_steps':'57',
          'actor.optim.lr_scheduler_type':'constant',
          'actor.use_kl_loss':'False','actor.kl_loss_coef':'0',
          'actor.entropy_coeff':'0','actor.clip_ratio':'0.2','actor.grad_clip':'1',
          'rollout.log_prob_micro_batch_size_per_gpu':'8',
          'ref.log_prob_micro_batch_size_per_gpu':'8',
          'actor.fsdp_config.param_offload':'True',
          'actor.fsdp_config.optimizer_offload':'True',
          'rollout.gpu_memory_utilization':'0.2',
          'actor.checkpoint.save_contents':'[model,optimizer,extra]',
          'actor.checkpoint.load_contents':'[model,optimizer,extra]',
        }.items():values[role+'.'+key]=value
    values['actor_rollout_ref.actor.mini_batch_counts_trajectories']='True'
    # Reranker preserves its original filtering and rollout count; no fixed update count.
    seen=set();result=[]
    for raw in original:
        key,value=raw.split('=',1);plain=key.lstrip('+')
        if plain in seen:raise ValueError('Duplicate '+plain)
        seen.add(plain)
        value = values.pop(plain,value)
        value = value.replace('/dataset/shaoshitong/search-r1-rl/cosearch/author-native/repo', RUN+'/repo')
        value = value.replace('/dataset/shaoshitong/search-r1-rl/cosearch/author-native/tools', RUN+'/tools')
        result.append(key+'='+value)
    result.extend('++'+key+'='+value for key,value in sorted(values.items()))
    return result

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('original');p.add_argument('output');a=p.parse_args()
    overrides=configure(json.loads(Path(a.original).read_text()))
    with open(a.output,'x') as f:json.dump(overrides,f,indent=2);f.write('\n')
