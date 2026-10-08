import ast,hashlib,importlib.util,json,os,random,subprocess,sys,tempfile
from pathlib import Path
from types import MethodType
from unittest.mock import patch
import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tensordict import TensorDict
from omegaconf import OmegaConf
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'work/tree-grpo')]
from verl.workers.actor.dp_actor import DataParallelPPOActor
from search_r1.baselines import tree_rank_audit
from search_r1.baselines.rollout import select_rows
spec=importlib.util.spec_from_file_location('distributed_fixture',ROOT/'work/tree-grpo/tests/baselines/test_distributed_paper_metrics.py');fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)

def run(data,micro=1,distributed=False,fail=None):
 model=torch.nn.Linear(1,1,bias=False)
 with torch.no_grad():model.weight.fill_(.15)
 if distributed:model=torch.nn.parallel.DistributedDataParallel(model)
 actor=DataParallelPPOActor.__new__(DataParallelPPOActor);actor.actor_module=model;actor.actor_optimizer=torch.optim.SGD(model.parameters(),lr=.02)
 actor.config=OmegaConf.create(dict(ppo_mini_batch_size=len(data),ppo_micro_batch_size=micro,ppo_epochs=1,state_masking=True,use_kl_loss=True,use_dynamic_bsz=False,baseline_method='tree_grpo',entropy_coeff=.03,clip_ratio=.2,grad_clip=100.,kl_loss_type='kl',kl_loss_coef=.2))
 def forward(self,micro_batch,temperature):
  logp=self.actor_module(micro_batch['input_ids'][:,:1].float()).expand(-1,3);return logp*.2+1.,logp
 actor._forward_micro_batch=MethodType(forward,actor)
 if fail:
  def bad():raise fail
  actor._optimizer_step=bad
 with patch.object(TensorDict,'cuda',lambda x,*a,**k:x):metrics=actor.update_policy(data)
 return metrics,next(model.parameters()).detach().clone()

def seed():random.seed(31);np.random.seed(31);torch.manual_seed(31)
def rng():return random.getstate(),np.random.get_state(),torch.random.get_rng_state().clone()
def equal_rng(a,b):
 assert a[0]==b[0] and a[1][0]==b[1][0] and np.array_equal(a[1][1],b[1][1]) and a[1][2:]==b[1][2:] and torch.equal(a[2],b[2])
def environment(path):return {'SEARCH_R1_TREE_RANK_AUDIT_DIR':str(path),'SEARCH_R1_BASELINE_RUN_ID':'independent-fixture','SEARCH_R1_BASELINE_COMMIT':'source-fixture','SEARCH_R1_BASELINE_CONFIG_SHA256':'config-fixture','SEARCH_R1_BASELINE_DATA_IDENTITY':'split-fixture','SLURM_JOB_ID':'123'}

def test_ast_removing_only_tree_guard_preserves_actor_source():
 original=subprocess.check_output(['git','show','87e4080:verl/workers/actor/dp_actor.py'],cwd=ROOT/'work/tree-grpo',text=True)
 tree=ast.parse((ROOT/'work/tree-grpo/verl/workers/actor/dp_actor.py').read_text());old=ast.parse(original)
 count=0
 for n in ast.walk(tree):
  if isinstance(n,ast.If) and n.orelse and isinstance(n.orelse[0],ast.If) and 'TreeRankAudit' in ast.unparse(n.orelse[0]):
   n.orelse=[];count+=1
 assert count==1 and ast.dump(tree,include_attributes=False)==ast.dump(old,include_attributes=False)

@pytest.mark.parametrize('case',['unequal','empty_leaf','empty_rank'])
@pytest.mark.parametrize('micro',[1,4])
def test_actual_actor_on_off_unchanged(case,micro,tmp_path):
 data=fixture.full_data(case);data.meta_info['baseline_outer_step']=1;saved={k:v.clone() for k,v in data.batch.items()}
 with patch.dict(os.environ,environment(tmp_path)):
  os.environ.pop('SEARCH_R1_TREE_RANK_AUDIT_DIR');seed();off,w0=run(data,micro);r0=rng()
  os.environ['SEARCH_R1_TREE_RANK_AUDIT_DIR']=str(tmp_path);seed();on,w1=run(data,micro);r1=rng()
 assert off==on and torch.equal(w0,w1);equal_rng(r0,r1)
 for k,v in data.batch.items():assert torch.equal(v,saved[k])
 report=json.loads((tmp_path/'step-000001-rank-00.json').read_text());assert report['successful_optimizer_calls']==1 and report['finite_grad_norms']==1 and report['local_shard_sample_layout_unchanged'] and report['local_shard_sample_delta']['nonzero']>0

 def_not_used=0

def test_global_empty_skip(tmp_path):
 data=fixture.full_data('unequal');data.meta_info['baseline_outer_step']=2;data.batch['loss_mask'].zero_();data.batch['policy_weights'].zero_()
 with patch.dict(os.environ,environment(tmp_path)):metrics,w=run(data)
 report=json.loads(next(tmp_path.glob('*.json')).read_text());assert metrics['actor/optimizer_steps']==0 and report['successful_optimizer_calls']==0 and report['skipped_empty_minibatches']==1 and report['existing_grad_norms']==[] and w.item()==pytest.approx(.15)

def test_optimizer_original_exception_no_completion(tmp_path):
 data=fixture.full_data('unequal');data.meta_info['baseline_outer_step']=1;failure=RuntimeError('original optimizer failure')
 with patch.dict(os.environ,environment(tmp_path)),pytest.raises(RuntimeError) as caught:run(data,fail=failure)
 assert caught.value is failure and not list(tmp_path.glob('*.json'))

def test_publication_failure_duplicate_and_count_contamination(tmp_path):
 data=fixture.full_data('unequal');data.meta_info['baseline_outer_step']=1
 with patch.dict(os.environ,environment(tmp_path)),patch.object(tree_rank_audit.os,'link',side_effect=OSError('disk error')),pytest.raises(OSError):run(data)
 assert not list(tmp_path.glob('*.json'))
 for p in tmp_path.iterdir():p.unlink()
 with patch.dict(os.environ,environment(tmp_path)):run(data)
 original=next(tmp_path.glob('*.json')).read_bytes()
 with patch.dict(os.environ,environment(tmp_path)),pytest.raises(FileExistsError):run(data)
 assert next(tmp_path.glob('*.json')).read_bytes()==original
 a=tree_rank_audit.TreeRankAudit(torch.nn.Linear(1,1),data.batch,data.meta_info,OmegaConf.create({'ppo_mini_batch_size':4,'ppo_micro_batch_size':1,'ppo_epochs':1}),ROOT/'work/tree-grpo/verl/workers/actor/dp_actor.py')
 with pytest.raises(ValueError):a.gradient(torch.tensor(float('nan')))
 with pytest.raises(ValueError):a.finish(1,0)

def worker(rank,rendezvous,out):
 torch.set_num_threads(1);dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2)
 try:
  for case in ['unequal','empty_rank']:
   for micro in [1,2]:
    data=select_rows(fixture.full_data(case),[rank,rank+2]);step=1+['unequal','empty_rank'].index(case)*2+micro-1;data.meta_info['baseline_outer_step']=step
    env=environment(Path(out));seed()
    with patch.dict(os.environ,env):os.environ.pop('SEARCH_R1_TREE_RANK_AUDIT_DIR');off,w0=run(data,micro,True);r0=rng()
    seed()
    with patch.dict(os.environ,env):on,w1=run(data,micro,True);r1=rng()
    assert off==on and torch.equal(w0,w1);equal_rng(r0,r1)
 finally:dist.destroy_process_group()

def test_real_two_rank_gloo_actor_evidence(tmp_path):
 mp.spawn(worker,args=(str(tmp_path/'rendezvous'),str(tmp_path)),nprocs=2,join=True)
 reports=[json.loads(p.read_text()) for p in tmp_path.glob('*.json')];assert len(reports)==8
 for step in range(1,5):
  rows=[r for r in reports if r['outer_step']==step];assert {r['rank'] for r in rows}=={0,1} and all(r['world_size']==2 and r['successful_optimizer_calls']==1 and len(r['existing_grad_norms'])==1 for r in rows)
 # Explicit rank0-empty case remains a successful synchronized update, not skip.
 assert all(r['successful_optimizer_calls']==1 for r in reports)
