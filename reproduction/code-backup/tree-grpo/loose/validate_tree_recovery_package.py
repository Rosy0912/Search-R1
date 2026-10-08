import json,os,sys,time
from pathlib import Path
root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
tools=root/'instruct300/candidates/recovery-502e9fa-f91c4f17'
os.environ.update(CUDA_VISIBLE_DEVICES='',TREE_INSTRUCT_MODEL=str(root/'instruct300/assets/qwen2.5-3b-instruct-aa8e725'),
                 TREE_COMMON_DATA=str(root/'instruct300/common-data'),SEARCH_R1_ASSETS=str(root/'assets-root'))
run=root/'instruct300/validation-recovery-f91c4f17'
run.mkdir(exist_ok=False)
os.environ['SEARCH_R1_BASELINE_OUTPUT']=str(run/'output')
sys.path[:0]=[str(tools),str(root/'tools'),str(tools/'source/repo')]
import supervisor
config,_=supervisor.configuration(run,'recovery300')
assert config.data.train_batch_size*config.baselines.tree.roots*config.baselines.tree.leaves==3840
assert 3840//config.actor_rollout_ref.actor.ppo_mini_batch_size==10
assert config.trainer.acceptance_step==105
print(json.dumps(dict(status='ACTUAL_H1009_CPU_FROZEN_CONFIG_SPLIT_PASS_NOT_GPU',epoch=time.time(),
    train_questions=config.data.train_batch_size,mini=config.actor_rollout_ref.actor.ppo_mini_batch_size,
    micro=config.actor_rollout_ref.actor.ppo_micro_batch_size,expected_calls=10,reference=config.actor_rollout_ref.ref.model_path,
    acceptance_step=config.trainer.acceptance_step,eval_freq=config.trainer.test_freq,save_freq=config.trainer.save_freq)))
