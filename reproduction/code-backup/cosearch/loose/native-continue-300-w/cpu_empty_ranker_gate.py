"""Actual native worker/manager/transformed fit branches. CPU only, no PPO updates."""
import ast,asyncio,copy,hashlib,inspect,json,random,textwrap,types,tempfile
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from verl import DataProto
from verl.experimental.agent_loop.agent_loop import _InternalAgentLoopOutput,AgentLoopMetrics,CoSearchAgentLoopManager
from verl.experimental.agent_loop.search_r1_reranker_agent_loop_worker import CoSearchAgentLoopWorker
import verl.experimental.agent_loop.search_r1_reranker_agent_loop_worker as wm
from verl.trainer.ppo.search_r1_reranker_ray_trainer import CoSearchRayTrainer
import empty_ranker as fix
from bounded_sampling import generate_supported,validate_result,input_digest

def assert_nonempty(value):
 assert len(value)>0

records=[];raw=[]
record=lambda k,v:records.append((k,v))
def plain(x):
 if torch.is_tensor(x):return (str(x.dtype),list(x.shape),x.tolist())
 if hasattr(x,'model_dump'):return plain(x.model_dump())
 if isinstance(x,DataProto):return dict(tensors=plain(dict(x.batch)),non_tensors=plain(x.non_tensor_batch),meta=plain(x.meta_info))
 if isinstance(x,np.ndarray):return (str(x.dtype),x.tolist())
 if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [plain(v) for v in x]
 return x
cfg=OmegaConf.create({'actor_rollout_ref':{'rollout':{'temperature':1.,'top_p':1.,'calculate_log_probs':False,'free_cache_engine':True,'val_kwargs':{'temperature':0.,'top_p':1.},'agent':{'default_agent_loop':'co_search'}}},'reranker_actor_rollout_ref':{'rollout':{'prompt_length':2,'response_length':2,'temperature':1.,'top_p':1.,'val_kwargs':{'temperature':0.,'top_p':1.}}},'trainer':{'reranker_sampling_val_start_step':10000,'balance_batch':True,'rollout_data_dir':None}})
def internal(i):
 return _InternalAgentLoopOutput(prompt_ids=torch.tensor([[1,i]]),response_ids=torch.tensor([[2,3]]),input_ids=torch.tensor([[1,i,2,3]]),position_ids=torch.arange(4).reshape(1,4),attention_mask=torch.ones(1,4,dtype=torch.long),response_mask=torch.ones(1,2,dtype=torch.long),reward_score=float(i%2),metrics=AgentLoopMetrics(generate_sequences=1.,tool_calls=.5),extra_fields={'uid':str(i)})
def batch(n=512,validate=False):
 return DataProto.from_dict(tensors={'input_ids':torch.ones(n,2,dtype=torch.long)},non_tensors={'tools_kwargs':np.array([{} for _ in range(n)],dtype=object)},meta_info={'global_steps':13,'validate':validate})
worker=CoSearchAgentLoopWorker.__ray_metadata__.modified_class
obj=worker.__new__(worker);obj.config=cfg
original_generate=worker.generate_sequences
original_post=worker._postprocess
async def info(*a):return [{} for _ in a[1]]
wm.get_trajectory_info=info
wm.RolloutTraceConfig=types.SimpleNamespace(get_instance=lambda:types.SimpleNamespace(max_samples_per_step_per_worker=None))
rr=[]
async def runloop(*a,**kw):return internal(7),[]
async def filterloop(*a):return rr,{'post_filter/num_groups':0}
obj._run_agent_loop=runloop;obj._process_reranker_outputs=filterloop
fix.install_worker(worker,record,lambda k,v:raw.append((k,v)))
out=asyncio.run(obj.generate_sequences(batch()));assert len(out[0])==512 and fix.is_empty(out[1])
assert out[1].meta_info['worker_reranker_metrics']=={'post_filter/num_groups':0}
rr[:]=[internal(1),internal(2),internal(3),internal(4)]
a=asyncio.run(original_generate(obj,batch()));b=asyncio.run(obj.generate_sequences(batch()));assert plain(a)==plain(b)
assert plain(asyncio.run(original_generate(obj,batch(validate=True))))==plain(asyncio.run(obj.generate_sequences(batch(validate=True))))
try:obj._postprocess([])
except AssertionError:pass
else:raise AssertionError('empty outside generation accepted')
# Empty main is explicitly rejected at first postprocess inside real generation scope.
oldloop=obj._run_agent_loop
async def badloop(*a,**kw):return None,[]
obj._run_agent_loop=badloop
try:asyncio.run(obj.generate_sequences(batch()))
except (AttributeError,AssertionError):pass
else:raise AssertionError('malformed main accepted')
obj._run_agent_loop=oldloop

# Capture the exact submitted source transformations while installing actual native methods.
captured={};compile_original=fix._compile
def capture(original,replacements,extra):
 source=textwrap.dedent(inspect.getsource(original))
 for before,after in replacements:
  if '\n' in after:
   indent=source[:source.index(before)].rsplit('\n',1)[-1];after=after.replace('\n','\n'+indent)
  source=source.replace(before,after)
 captured[original.__name__]=(source,extra,dict(original.__globals__))
 return compile_original(original,replacements,extra)
fix._compile=capture
manager=CoSearchAgentLoopManager.__new__(CoSearchAgentLoopManager);manager.config=cfg;manager.reward_model_manager=None
wake=[];manager.wake_up=lambda:wake.append('wake');manager.sleep=lambda:wake.append('sleep')
metrics_seen=[];manager._aggregate_reranker_metrics=lambda ms:metrics_seen.append(copy.deepcopy(ms)) or {'post_filter/num_groups':sum(m.get('post_filter/num_groups',0) for m in ms)}
fix.install_controller(manager,CoSearchRayTrainer,record);fix._compile=compile_original
original_manager=CoSearchAgentLoopManager.generate_sequences
class Remote:
 def __init__(self,value):self.value=value
 def remote(self,*a):return self.value
outputs=[]
import verl.experimental.agent_loop.agent_loop as am
oldget=am.ray.get;am.ray.get=lambda x:x
try:
 for empty_indices in (set(),{0},{3},{7},{0,3,7},set(range(8))):
  outputs=[]
  for i in range(8):
   main=original_post(obj,[internal(i) for _ in range(512)])
   rank=fix.empty_batch(2,2) if i in empty_indices else original_post(obj,[internal(i) for _ in range(4)])
   rank.meta_info['worker_reranker_metrics']={'post_filter/num_groups':0 if i in empty_indices else 1}
   outputs.append((main,rank))
  manager.agent_loop_workers=[types.SimpleNamespace(generate_sequences=Remote(v)) for v in outputs]
  expected_main=DataProto.concat([copy.deepcopy(v[0]) for v in outputs])
  expected_rows=[copy.deepcopy(v[1]) for v in outputs if len(v[1])]
  expected_rank=DataProto.concat(expected_rows) if expected_rows else None
  baseline=None
  if not empty_indices:
   saved=copy.deepcopy(outputs)
   baseline=original_manager(manager,batch(4096))
   manager.agent_loop_workers=[types.SimpleNamespace(generate_sequences=Remote(v)) for v in saved]
  before_wake=len(wake);result=manager.generate_sequences(batch(4096));assert len(wake)-before_wake==2
  assert len(result[0])==4096 and plain(dict(result[0].batch))==plain(dict(expected_main.batch))
  assert len(metrics_seen[-1])==8 and result[1].meta_info['reranker_rollout_metrics']['post_filter/num_groups']==8-len(empty_indices)
  if expected_rank is None:assert fix.is_empty(result[1]) and result[1].meta_info['timing']=={}
  else:
   assert plain(dict(result[1].batch))==plain(dict(expected_rank.batch))
   assert plain(result[1].non_tensor_batch)==plain(expected_rank.non_tensor_batch)
  if baseline is not None:assert plain(result)==plain(baseline)

finally:am.ray.get=oldget

# Actual transformed native fit branch: after generation through both RPC/results/stats.
source,extra,globals_fit=captured['fit'];tree=ast.parse(source)
step=next(n for n in ast.walk(tree) if isinstance(n,ast.With) and any(isinstance(c.context_expr,ast.Call) and c.context_expr.args and isinstance(c.context_expr.args[0],ast.Constant) and c.context_expr.args[0].value=='step' for c in n.items))
branch=step.body[1:] # generation handled above by actual manager; all subsequent original step statements.
fit_calls=[]
class RPC:
 def remote(self,**kwargs):
  fit_calls.append(kwargs['agent_name']);return (kwargs['batch'],{}, {}, {})
for empty in (True,False,True,False):
 m=original_post(obj,[internal(1),internal(2),internal(3),internal(4)])
 r=fix.empty_batch(2,2) if empty else original_post(obj,[internal(1),internal(2),internal(3),internal(4)])
 trainer=types.SimpleNamespace(config=cfg,is_reranker_trainable=True,global_steps=13,use_reference_policy=False,use_critic=False,critic_wg=None,actor_rollout_wg=object(),reranker_actor_rollout_wg=object(),tokenizer=None,reranker_tokenizer=None,kl_ctrl_in_reward=None,ref_in_actor=False,
  _balance_batch=lambda b,**kw:assert_nonempty(b),_compute_rollout_stats=lambda b,*a,**kw:assert_nonempty(b) or {},_log_rollout_data=lambda *a:None)
 before=len(fit_calls)
 namespace=dict(globals_fit,**extra);namespace.update(self=trainer,main_batch=m,reranker_batch=r,reranker_train_enabled=True,metrics={},timing_raw={},process_single_agent_ppo_step=RPC(),ray=types.SimpleNamespace(get=lambda x:x))
 exec(compile(ast.Module(body=branch,type_ignores=[]),'<actual-transformed-native-fit-step>','exec'),namespace)
 assert fit_calls[before:]==(['main'] if empty else ['main','reranker'])
 assert trainer.is_reranker_trainable is True
 assert namespace['reranker_train_enabled']==(not empty)
 # Call the actual original save method after the actual transformed step branch.
 with tempfile.TemporaryDirectory() as target:
  save_calls=[]
  trainer.config=copy.deepcopy(cfg)
  trainer.config.trainer.default_local_dir=target
  trainer.config.trainer.default_hdfs_dir=None
  trainer.config.trainer.max_actor_ckpt_to_keep=2
  trainer.actor_rollout_wg=types.SimpleNamespace(save_checkpoint=lambda *a,**k:save_calls.append(('main',a,k)))
  trainer.reranker_actor_rollout_wg=types.SimpleNamespace(save_checkpoint=lambda *a,**k:save_calls.append(('reranker',a,k)))
  trainer.train_dataloader=types.SimpleNamespace(state_dict=lambda:{'CPU_fixture':13})
  CoSearchRayTrainer._save_checkpoint(trainer)
  assert [x[0] for x in save_calls]==['main','reranker']
  assert all(x[2]['max_ckpt_to_keep']==2 for x in save_calls)
  assert (Path(target)/'global_step_13/data.pt').is_file()

def expected_failure(value):
 try:validate_result((original_post(obj,[internal(1)]),value),1,False)
 except (AssertionError,KeyError,TypeError,AttributeError):return
 raise AssertionError('malformed marked empty accepted')
e=fix.empty_batch(2,2);validate_result((original_post(obj,[internal(1)]),e),1,False)
for key in ('responses','attention_mask','response_mask','rm_scores','input_ids','position_ids','prompts'):
 malformed=fix.empty_batch(2,2);del malformed.batch[key];expected_failure(malformed)
malformed=fix.empty_batch(2,2);malformed.batch['rm_scores']=torch.empty(0,3);expected_failure(malformed)

support=[];calls=[]
def original(b):
 calls.append(input_digest(b));support[:]=[0]
 return original_post(obj,[internal(1)]),fix.empty_batch(2,2)
result=generate_supported(original,batch(1),support,lambda x:None,record,lambda *a:None,lambda *a:None,13,False)
assert len(calls)==2 and calls[0]==calls[1] and fix.is_empty(result[1])
assert not torch.cuda.is_initialized()
output={'passed':True,'actual_modified_worker':True,'actual_manager_mixed_cases':6,'actual_transformed_fit_main_and_ranker_RPC_cases':4,'actual_original_save_both_roles_cases':4,'canonical_schema_rejections':8,'bounded_terminal_empty_two_calls':True,'CUDA_initialized':False,'PPO_updates':0,'fix_sha256':hashlib.sha256(Path(fix.__file__).read_bytes()).hexdigest()}
Path(__file__).with_name('cpu-empty-ranker-gate.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(output),flush=True)
