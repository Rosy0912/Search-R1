"""Execution layout contract. No model, optimizer, CUDA or scheduler imports."""
from pathlib import Path
import copy,hashlib,json,os
WORLD=16
NODES=2
GPUS=8
JOB='zhangjunming-cgrts7-main16-v1'
ENV='CGRTS16_PARALLEL_PLAN'
PARENT='92942a32161071b85cc70906050143ed58244eb1'
BASE=Path('/plm-dhw/zhangjunming/Workspace/Recursive-Tree-GRPO-CGRTS7B-Main16-20261020-v1')
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def read_plan(path=None):
 p=Path(path or os.environ[ENV]);v=json.loads(p.read_text())
 assert v['job']==JOB and v['root']==str(BASE) and v['world_size']==WORLD and v['nodes']==NODES
 assert v['gpus_per_node']==GPUS and v['parent_science_source']==PARENT and v['fresh_initialization'] and not v['resume']
 assert v['allocation']==dict(nodes=2,gpus_per_node=8,master_cpus=64,master_memory_gib=900,worker_cpus=48,worker_memory_gib=512)
 assert v['actual_update_limit']==200 and v['scheduler_scale']==200 and v['warmup_steps']==57
 return v
def active():return bool(os.environ.get(ENV))
def projected_config(config):
 from omegaconf import OmegaConf
 actual=OmegaConf.to_container(config,resolve=True)
 assert actual['trainer']['n_gpus_per_node']==8 and actual['trainer']['nnodes']==2
 a=actual['actor_rollout_ref']['actor'];assert a['ppo_mini_batch_size']==128 and a['ppo_micro_batch_size']==16 and a['ppo_epochs']==1
 assert actual['data']['train_batch_size']==512 and a['ulysses_sequence_parallel_size']==1
 assert (a['optim']['lr'],a['optim']['lr_warmup_steps'],a['optim']['total_training_steps'])==(1e-6,57,200)
 out=copy.deepcopy(actual);out['trainer']['nnodes']=1;out['actor_rollout_ref']['actor']['ppo_micro_batch_size']=16
 return OmegaConf.create(out)
def reshard_original_eight_rows(rows):
 """Preserve the exact global128-row group of each of the20 optimizer calls."""
 assert rows==2560
 order=[]
 for new_rank in range(16):
  old_rank,half=divmod(new_rank,2)
  for opportunity in range(20):
   begin=old_rank*320+opportunity*16+half*8
   order.extend(range(begin,begin+8))
 assert sorted(order)==list(range(rows))
 return order
def validate_tasks(tasks):
 assert len(tasks)==2
 assert [(t['role'],t['replicas']) for t in tasks]==[('PYTORCH_MASTER',1),('PYTORCH_WORKER',1)]
 for i,t in enumerate(tasks):
  r=t['resource_spec'];assert r['accelerate_device_count']==8 and r['cpu_count']==(64 if i==0 else 48) and r['memory_gib']==(900 if i==0 else 512)
  assert r['machine_types']==['n2ls.ie.i60b'] and r['rdma_name']=='ib-cluster-01'
def validate_gate(plan,sub,gate):
 assert sub['job']==gate['job']==plan['job'] and sub['uid']==gate['uid']
 assert sub['plan_sha256']==gate['plan_sha256'] and gate['state']=='RUNNING'
 validate_tasks(gate['tasks'])
def node_coordinates(environ):
 n=int(environ['NODE_RANK']);assert int(environ['NODE_COUNT'])==2 and 0<=n<2
 assert int(environ['RANK'])==n and int(environ['WORLD_SIZE'])==2
 return n,environ['MASTER_ADDR']
