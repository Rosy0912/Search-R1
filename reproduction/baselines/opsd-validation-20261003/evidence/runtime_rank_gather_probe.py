import json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
from verl import DataProto
from tensordict import TensorDict
from search_r1.baselines.actor import DistillationActor
HERE=Path(__file__).resolve().parent

def worker(rank):
 torch.set_num_threads(1)
 dist.init_process_group('gloo',init_method='file://'+str(HERE/'runtime-rank-rendezvous'),rank=rank,world_size=2)
 try:
  actor=DistillationActor.__new__(DistillationActor)
  actor.actor_module=torch.nn.Linear(1,1)
  actor.actor_optimizer=torch.optim.AdamW(actor.actor_module.parameters())
  actor.config=SimpleNamespace(distillation=SimpleNamespace(),ppo_epochs=1,ppo_mini_batch_size=1,ppo_micro_batch_size=1)
  actor.teacher=SimpleNamespace(parameter_count=10+rank,scored_tokens=20+rank,cost_metrics=lambda:{'teacher/cpu_seconds_cumulative_per_rank':1.+rank},audit_metrics=lambda optimizer:{'teacher/trainable_parameters_per_rank':0,'teacher/gradient_tensors_per_rank':0,'teacher/student_optimizer_membership_per_rank':0})
  actor._optimizer_step=lambda: (_ for _ in ()).throw(AssertionError('Empty mini must skip optimizer'))
  before=[p.detach().clone() for p in actor.actor_module.parameters()]
  data=DataProto(batch=TensorDict({'responses':torch.zeros(1,1,dtype=torch.long),'loss_mask':torch.zeros(1,1)},batch_size=[1]),non_tensor_batch={'reward_model':np.array([{}],dtype=object)})
  metrics=actor.update_policy(data)
  assert metrics['teacher/cpu_seconds_cumulative_per_rank/rank_0']==1.
  assert metrics['teacher/cpu_seconds_cumulative_per_rank/rank_1']==2.
  assert metrics['teacher/parameters_per_rank/rank_0']==10 and metrics['teacher/parameters_per_rank/rank_1']==11
  for a,b in zip(before,actor.actor_module.parameters()):assert torch.equal(a,b)
  (HERE/f'runtime_rank_{rank}.json').write_text(json.dumps(metrics,indent=2)+'\n')
 finally:dist.destroy_process_group()

if __name__=='__main__':
 mp.spawn(worker,nprocs=2,join=True)
 metrics=[json.loads((HERE/f'runtime_rank_{i}.json').read_text()) for i in range(2)]
 result=DataProto.concat([DataProto(meta_info={'metrics':v}) for v in metrics]).meta_info['metrics']
 assert result['teacher/cpu_seconds_cumulative_per_rank/rank_1']==2.
 print(json.dumps({'real_cpu_gloo_world':2,'complete_rank_keys_survive_first_metadata_concat':True,'rank0_legacy_cost':result['teacher/cpu_seconds_cumulative_per_rank'],'rank1_observation':result['teacher/cpu_seconds_cumulative_per_rank/rank_1'],'loss_parameters_unchanged_empty_minis':True,'gpu_tested':False},indent=2))
