"""Actual original CPU call chain; fake retrieval and generate backend, zero RL."""
import asyncio,copy,hashlib,inspect,json,random,ast,textwrap,types
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from omegaconf import OmegaConf
import phase_sampling as fix
from verl.experimental.agent_loop.co_search_agent_loop import CoSearchAgentLoop,AgentData
from verl.experimental.agent_loop.tool_parser import FunctionCall
from verl.tools.co_search_tool import CoSearchTool
from verl.experimental.agent_loop.search_r1_reranker_agent_loop_worker import CoSearchAgentLoopWorker
from verl.experimental.agent_loop.agent_loop import CoSearchAgentLoopManager
ROOT=Path('/dataset/shaoshitong/search-r1-rl/cosearch/author-native')
tok=AutoTokenizer.from_pretrained('/dataset/shaoshitong/search-r1-rl/cosearch/assets-root/assets/qwen2.5-3b',local_files_only=True)
original=CoSearchAgentLoop._handle_processing_tools_state
corrected,metadata=fix.corrected_sampling_method(original)
# Execute exact original phase selection statements, not a handwritten replacement.
worker=CoSearchAgentLoopWorker.__ray_metadata__.modified_class
src=textwrap.dedent(inspect.getsource(worker.generate_sequences))
tree=ast.parse(src);body=tree.body[0].body
start=next(i for i,n in enumerate(body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='reranker_config' for t in n.targets))
end=next(i for i in range(start,len(body)) if isinstance(body[i],ast.If) and isinstance(body[i].test,ast.Name) and body[i].test.id=='use_val_params_for_reranker')+1
phase_ast=ast.Module(body=body[start:end],type_ignores=[])
def phase(validate,step):
 cfg=OmegaConf.create(dict(reranker_actor_rollout_ref=dict(rollout=dict(temperature=1.,top_p=1.,val_kwargs=dict(temperature=0.,top_p=1.))),trainer=dict(reranker_sampling_val_start_step=10000)))
 scope=dict(self=types.SimpleNamespace(config=cfg),is_validate=validate,batch=types.SimpleNamespace(meta_info=dict(global_steps=step)))
 exec(compile(phase_ast,'original_phase_selection','exec'),scope)
 return scope['reranker_sampling_params']
DOCS=[dict(id='cpu-doc-'+str(i),contents='passage '+str(i)+' contains CPU-answer') for i in range(1,51)]
output='<reason>Order by relevance.</reason><rerank>[1] > [2] > [3] > [4] > [5]</rerank>'
async def one(method,sampling):
 calls=[]
 class Backend:
  async def generate(self,**kwargs):
   calls.append(copy.deepcopy(kwargs))
   return types.SimpleNamespace(token_ids=tok.encode(output,add_special_tokens=False),log_probs=None)
 tool=CoSearchTool(dict(retrieval_service_url='http://unused',default_top_n=50,default_top_m=5))
 async def retrieve(self,query,top_n):
  assert query=='CPU question' and top_n==50
  return copy.deepcopy(DOCS)
 tool._call_retrieval_api=types.MethodType(retrieve,tool)
 obj=CoSearchAgentLoop.__new__(CoSearchAgentLoop)
 for k,v in dict(tools={'search':tool},reranker_tokenizer=tok,reranker_server_manager=Backend(),max_parallel_calls=1,processor=None,tool_parser_name='qwen',loop=asyncio.get_running_loop(),tokenizer=tok,system_prompt=[],response_length=100000,max_tool_response_length=100000,tool_response_truncate_side='left',collect_reranker_outputs=True).items():setattr(obj,k,v)
 kwargs={'search':{'create_kwargs':{'keep':'unchanged'}}}
 if sampling is not None:kwargs['search']['create_kwargs']['reranker_sampling_params']=copy.deepcopy(sampling)
 before=copy.deepcopy(kwargs)
 data=AgentData(messages=[],image_data=None,metrics={},request_id='cpu-request',tools_kwargs=kwargs,initial_query='CPU question',answers=['CPU-answer']);data.extra_fields={'uid':'cpu-main'}
 data.tool_calls=[FunctionCall(name='search',arguments=json.dumps({'query':'CPU question'}))]
 state=await method(obj,data,{'temperature':1.,'top_p':1.})
 assert kwargs==before and not tool._instance_kwargs and len(calls)==1
 rr=data.reranker_output_list[0]
 signature=dict(state=state.value,messages=data.messages,prompt_ids=data.prompt_ids,response_mask=data.response_mask,tool_rewards=data.tool_rewards,ranker_prompt_ids=rr.prompt_ids,ranker_response_ids=rr.response_ids,ranker_mask=rr.response_mask,ranker_extra=rr.extra_fields)
 return calls[0],signature
async def run():
 cases=[('train',phase(False,1)),('validation',phase(True,1)),('late',phase(False,10000)),('fixed_missing_key',None)]
 rows=[]
 for label,params in cases:
  cpu_rng=torch.get_rng_state().clone();np_rng=np.random.get_state();py_rng=random.getstate()
  a,sa=await one(original,params);b,sb=await one(corrected,params)
  assert a['sampling_params']=={'temperature':0.,'top_p':1.}
  assert b['sampling_params']==(params if params is not None else {'temperature':0.,'top_p':1.})
  assert {k:v for k,v in a.items() if k!='sampling_params'}=={k:v for k,v in b.items() if k!='sampling_params'} and sa==sb
  assert torch.equal(cpu_rng,torch.get_rng_state()) and random.getstate()==py_rng
  curr=np.random.get_state();assert np_rng[0]==curr[0] and np.array_equal(np_rng[1],curr[1]) and np_rng[2:]==curr[2:]
  rows.append(dict(case=label,original=a['sampling_params'],corrected=b['sampling_params'],other_backend_args_and_returns_exact=True,rng_equal=True))
 return rows
rows=asyncio.run(run())
metrics=[{'post_filter/num_groups':0,'post_filter/avg_group_size':0.,'post_filter/total_outputs':0,'fallback/kept_outputs':4},{'post_filter/num_groups':0,'post_filter/avg_group_size':0.,'post_filter/total_outputs':0,'fallback/kept_outputs':4}]
records=[];m=CoSearchAgentLoopManager.__new__(CoSearchAgentLoopManager);aggregate=CoSearchAgentLoopManager._aggregate_reranker_metrics
before=copy.deepcopy(metrics)
try:aggregate(m,metrics)
except ZeroDivisionError:pass
else:raise AssertionError('Original zero-support behavior changed')
result=fix.aggregate_supported(aggregate,m,metrics,lambda k,v:records.append((k,v)))
assert metrics==before and 'post_filter/avg_group_size' not in result and result['post_filter/num_groups']==0
assert records[-1][1]['qualified_groups']==0 and records[-1][1]['undefined_means']['post_filter/avg_group_size']['support_count']==0
nonzero=[{'post_filter/num_groups':2,'post_filter/avg_group_size':4.,'post_filter/total_outputs':8},{'post_filter/num_groups':1,'post_filter/avg_group_size':5.,'post_filter/total_outputs':5}]
assert fix.aggregate_supported(aggregate,m,nonzero,lambda *a:None)==aggregate(m,nonzero)
for support in [[],[0]]:
 try:fix.require_training_support(support)
 except AssertionError:pass
 else:raise AssertionError('No-qualified-group eligibility admitted fallback')
fix.require_training_support([1])
# Actual controller writer, NumPy scalar metrics, mixed zero/nonzero support.
import tempfile,os
from launch_observed import write
with tempfile.TemporaryDirectory(prefix='phase-m-cpu-write-') as tmp:
 os.environ['COSEARCH_NATIVE_EVIDENCE_ROOT']=tmp
 typed=[{k:np.int64(v) if isinstance(v,int) else np.float64(v) for k,v in m.items()} for m in metrics]
 typed_before=copy.deepcopy(typed)
 result=fix.aggregate_supported(aggregate,m,typed,write)
 assert typed==typed_before and result['post_filter/num_groups']==0
 mixed=[typed[0],{'post_filter/num_groups':np.int64(2),'post_filter/avg_group_size':np.float64(4.),'post_filter/total_outputs':np.int64(8)}]
 assert fix.aggregate_supported(aggregate,m,mixed,write)==aggregate(m,mixed)
 original_bound,support=fix.install_aggregation(m,write)
 try:
  m._aggregate_reranker_metrics(typed);assert support==[0]
  try:fix.require_training_support(support)
  except AssertionError:pass
  else:raise AssertionError('Fallback acquired training eligibility')
  support.clear();m._aggregate_reranker_metrics(mixed);assert support==[2]
  fix.require_training_support(support)
 finally:m._aggregate_reranker_metrics=original_bound
 assert m._aggregate_reranker_metrics(mixed)==aggregate(m,mixed)
 saved=[json.loads(x) for x in next(Path(tmp).glob('controller-*.jsonl')).read_text().splitlines()]
 assert len(saved)==4 and saved[0]['undefined_means']['post_filter/avg_group_size']['support_count']==0
 assert saved[1]['undefined_means']=={} and saved[1]['qualified_groups']==2
assert not torch.cuda.is_initialized()
report=dict(passed=True,real_RL_updates=0,synthetic_backend=True,cases=rows,zero_support_undefined_and_rejected=True,nonzero_aggregate_exact=True,actual_writer_numpy_scalar_mixed_zero_support_passed=True,manager_installed_support_clear_and_restore_passed=True,correction_metadata=metadata,sha256={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in ['phase_sampling.py','observe_worker.py','launch_observed.py','cpu_phase_gate.py']})
Path(__file__).with_name('phase-cpu-gate.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
