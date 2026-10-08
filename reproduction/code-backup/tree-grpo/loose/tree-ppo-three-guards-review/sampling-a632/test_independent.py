import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='4'
import json,pathlib,tempfile,datetime,ast
from types import SimpleNamespace
from unittest.mock import patch
import torch,torch.distributed as dist,torch.multiprocessing as mp
from tensordict import TensorDict
from search_r1.baselines.sampling_consistency import generation_preflight,record_raw_turn,reduce_pair_statistics,calibrate_actor
N=dict(temperature=1.,top_p=1.,top_k=-1,min_p=0.,presence_penalty=0.,frequency_penalty=0.,repetition_penalty=1.,n=1,logits_processors=[])
class Output:
 def __init__(self):
  self.batch={'responses':torch.ones(2,3,dtype=torch.long),'input_ids':torch.ones(2,5,dtype=torch.long),'attention_mask':torch.ones(2,5,dtype=torch.long),'position_ids':torch.arange(5).repeat(2,1),'old_log_probs':torch.zeros(2,3),'__sampling_log_probs':torch.zeros(2,3)}
  self.meta_info={'__sampling_distribution':dict(N)}
 def __len__(self):return len(self.batch['responses'])
def worker(rank,rendezvous,directory):
 torch.set_num_threads(1);dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2,timeout=datetime.timedelta(seconds=8))
 os.environ.update(SEARCH_R1_BASELINE_OUTPUT=directory,SEARCH_R1_BASELINE_COMMIT='a632-fixed',SEARCH_R1_BASELINE_CONFIG_SHA256='cpu-config',SEARCH_R1_BASELINE_DATA_IDENTITY='data',SEARCH_R1_MODEL_IDENTITY='model',SEARCH_R1_BASELINE_RUN_ID='cpu-run',SLURM_JOB_ID='123',TREE_PPO_CALIBRATION_ONLY='1',TREE_RECOVERY_PARENT_OPTIMIZER_STEPS='2500',SEARCH_R1_TREE_RANK_AUDIT_DIR=directory+'/audit')
 rollout=SimpleNamespace(temperature=1.,sampling_consistency=SimpleNamespace(enabled=True,enforce=False,max_abs_delta=.01))
 simple=SimpleNamespace(config=SimpleNamespace(rollout=rollout),_is_actor=True,actor=SimpleNamespace(use_ulysses_sp=False),actor_lr_scheduler=SimpleNamespace(last_epoch=110))
 results={}
 try:
  generation_preflight(simple,SimpleNamespace(meta_info={},__len__=lambda:2)) if False else None
  class Prompts:
   meta_info={}
   def __len__(self):return 2
  generation_preflight(simple,Prompts())
  rollout.sampling_consistency.enabled=(rank==0)
  try:generation_preflight(simple,Prompts());raise AssertionError('signature accepted')
  except RuntimeError as e:assert 'signature mismatch' in str(e);results['enabled_signature_refusal']=True
  rollout.sampling_consistency.enabled=True
  output=Output()
  if rank==1:output.batch.pop('__sampling_log_probs')
  try:record_raw_turn(simple,output);raise AssertionError('missingprivate accepted')
  except RuntimeError as e:assert 'preparation refusal' in str(e);results['missing_private_global_refusal']=True
  huge=reduce_pair_statistics(torch.zeros(2,dtype=torch.float64),torch.full((2,),-1e308,dtype=torch.float64),torch.ones(2),N,1.,.01)
  assert huge['mean_abs_delta'] is None and huge['sampled_k3_mean'] is None and not huge['allowed'];json.dumps(huge,allow_nan=False);results['global_sum_overflow_json_finite']=True
  output=Output();before={k:v.clone() for k,v in output.batch.items() if not k.startswith('__')};rng=torch.get_rng_state().clone()
  record_raw_turn(simple,output)
  assert '__sampling_log_probs' not in output.batch and '__sampling_distribution' not in output.meta_info
  for k,v in before.items():assert torch.equal(output.batch[k],v)
  assert torch.equal(rng,torch.get_rng_state());raw=json.loads(pathlib.Path(simple._sampling_consistency_records[0]).read_text());assert raw['saved_rows']==2
  results['raw_record_all_rows_old_tensor_rng_preserved']=True
  module=torch.nn.Linear(1,1,bias=False);opt=torch.optim.AdamW(module.parameters());param=next(module.parameters());opt.state[param]={'step':torch.tensor(2500.),'exp_avg':torch.ones_like(param)*.1,'exp_avg_sq':torch.ones_like(param)*.2}
  originalparam=param.detach().clone();originalstate={k:v.clone() for k,v in opt.state[param].items()};forbidden=lambda *a,**k:(_ for _ in ()).throw(AssertionError('optimizer/scheduler write attempted'))
  opt.step=forbidden
  policy=dict(enabled=True,dual_clip=3.,kl_log_bound=30.,monitor_rows_per_rank=4,target_old_policy_kl=.01,negative_tail_limit=.01,initial_old_logprob_max_abs_delta=.01)
  config=SimpleNamespace(ppo_micro_batch_size=2,ppo_mini_batch_size=4,ppo_epochs=1,clip_ratio=.2,grad_clip=1.,use_dynamic_bsz=False,state_masking=True)
  config.get=lambda k,d=None:{'baseline_method':'tree_grpo','recovery_max_grad_norm':1000.}.get(k,d)
  actor=SimpleNamespace(config=config,use_ulysses_sp=False,actor_module=module)
  def forward(data,temperature):
   # Deliberately consume CPU RNG. measure's fork must restore it.
   z=torch.rand_like(data['old_log_probs'])*0+param.sum()*1e-7
   return z,z
  actor._forward_micro_batch=forward
  w=SimpleNamespace(_is_actor=True,actor=actor,actor_optimizer=opt,actor_lr_scheduler=SimpleNamespace(last_epoch=110,step=forbidden),config=SimpleNamespace(actor=SimpleNamespace(policy_safety=policy)),_sampling_consistency_records=simple._sampling_consistency_records,_sampling_consistency_entries=simple._sampling_consistency_entries)
  batch=TensorDict(dict(input_ids=torch.ones(8,5,dtype=torch.long),responses=torch.ones(8,3,dtype=torch.long),attention_mask=torch.ones(8,5,dtype=torch.long),position_ids=torch.arange(5).repeat(8,1),loss_mask=torch.ones(8,3),policy_weights=torch.full((8,3),1/3),old_log_probs=torch.zeros(8,3),advantages=torch.ones(8,3)),batch_size=[8],device='cpu')
  data=SimpleNamespace(batch=batch,meta_info={'calibration_parent_outer':110,'baseline_outer_step':111,'temperature':1.})
  # Execute exact production worker endpoint AST with only device transfer/offload mocked.
  source=pathlib.Path(__file__).with_name('source')/'verl/workers/fsdp_workers.py'
  tree=ast.parse(source.read_text());klass=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='ActorRolloutRefWorker');method=next(x for x in klass.body if isinstance(x,ast.FunctionDef) and x.name=='calibrate_policy_safety');method.decorator_list=[]
  from verl import DataProto
  ns={'torch':torch,'DataProto':DataProto};exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),'fixed-worker-endpoint','exec'),ns)
  data.to=lambda device:data;w._is_offload_param=False
  rng=torch.get_rng_state().clone();mode=module.training
  with patch.object(torch.cuda,'empty_cache',lambda:None):returned=ns['calibrate_policy_safety'](w,data)
  assert returned.meta_info=={'calibration_complete':True}
  record=json.loads(pathlib.Path(directory,f'probability-calibration/rank-{rank:02d}.json').read_text())
  assert record['optimizer_committed']==0 and record['scheduler_advanced'] is False and record['restored_optimizer_steps']==[2500] and record['scheduler_step']==110
  assert torch.equal(rng,torch.get_rng_state()) and module.training==mode and torch.equal(param,originalparam)
  for k,v in originalstate.items():assert torch.equal(opt.state[param][k],v)
  assert param.grad is None and pathlib.Path(record['monitor_path']).exists();results['actual_calibrate_actor_zero_writes_rng_mode_params_adam_grad_unchanged']=True
  # Only one rank's raw publication fails; all must synchronously reject before returning.
  output=Output()
  import search_r1.baselines.sampling_consistency as M
  save=M.atomic_save
  def badsave(*args,**kwargs):
   if rank==1:raise OSError('independent IO fault')
   return save(*args,**kwargs)
  with patch.object(M,'atomic_save',badsave):
   try:record_raw_turn(simple,output);raise AssertionError('IO accepted')
   except RuntimeError as e:assert 'evidence-write failure' in str(e);results['one_rank_io_global_refusal']=True
 finally:
  pathlib.Path(directory,f'proof-{rank}.json').write_text(json.dumps(results));dist.destroy_process_group()
def test_real_two_rank_mock_worker_and_zero_write_actor():
 with tempfile.TemporaryDirectory() as directory:
  mp.spawn(worker,args=(directory+'/rdzv',directory),nprocs=2,join=True)
  proofs=[json.loads(pathlib.Path(directory,f'proof-{r}.json').read_text()) for r in range(2)];assert proofs[0]==proofs[1] and len(proofs[0])==6
 pathlib.Path(__file__).with_name('INDEPENDENT_GLOO_WORKER_PROOF.json').write_text(json.dumps({'actual_cpu_gloo_ranks':2,'checks_per_rank':proofs,'not_cuda_fsdp':True},indent=2)+'\n')
def test_wrong_shape_and_extreme_single_reductions():
 for sampled,current in [(torch.zeros(1),torch.zeros(2,3)),(torch.zeros(2,2),torch.zeros(2,3))]:
  try:reduce_pair_statistics(sampled,current,torch.ones_like(current),N,1.,.01);raise AssertionError('broadcast accepted')
  except RuntimeError as e:assert 'preparation refusal' in str(e)
 for value in [1e200,-1e200,1e308,-1e308]:
  decision=reduce_pair_statistics(torch.zeros(1,dtype=torch.float64),torch.tensor([value],dtype=torch.float64),torch.ones(1),N,1.,.01);json.dumps(decision,allow_nan=False);assert not decision['allowed']
