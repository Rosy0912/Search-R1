"""Ray's actual cloudpickle on full modified native class; fresh CPU interpreter only."""
import asyncio,hashlib,importlib.util,json,os,subprocess,sys,tempfile,types,inspect
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from verl import DataProto
import ray
import ray.cloudpickle as cloudpickle
from verl.experimental.agent_loop.agent_loop import SearchR1RerankerAgentLoopWorkerBase,_InternalAgentLoopOutput,AgentLoopMetrics
from verl.experimental.agent_loop.search_r1_reranker_agent_loop_worker import CoSearchAgentLoopWorker
import observe_worker,fallback_alignment,empty_ranker


def run_restored(cls,root):
 os.environ['COSEARCH_NATIVE_EVIDENCE_ROOT']=str(root)
 obj=cls.__new__(cls)
 obj.config=OmegaConf.create({'actor_rollout_ref':{'rollout':{'temperature':1.,'top_p':1.,'calculate_log_probs':False,'val_kwargs':{'temperature':0.,'top_p':1.},'agent':{'default_agent_loop':'co_search'}}},'reranker_actor_rollout_ref':{'rollout':{'prompt_length':2,'response_length':2,'temperature':1.,'top_p':1.,'val_kwargs':{'temperature':0.,'top_p':1.}}},'trainer':{'reranker_sampling_val_start_step':10000}})
 def row(i):
  return _InternalAgentLoopOutput(prompt_ids=torch.tensor([[1,i]]),response_ids=torch.tensor([[2,3]]),input_ids=torch.tensor([[1,i,2,3]]),position_ids=torch.arange(4).reshape(1,4),attention_mask=torch.ones(1,4,dtype=torch.long),response_mask=torch.ones(1,2,dtype=torch.long),reward_score=float(i%2),metrics=AgentLoopMetrics(),extra_fields={'uid':str(i)})
 async def info(*args):return [{} for _ in args[1]]
 # Patch only the restored native original generation's fake tracing backend.
 # Traverse real closure functions; do not replace the production wrappers.
 visited=set()
 def adapt(fn):
  if not inspect.isfunction(fn) or id(fn) in visited:return
  visited.add(id(fn))
  if 'get_trajectory_info' in fn.__globals__:
   fn.__globals__['get_trajectory_info']=info
   fn.__globals__['RolloutTraceConfig']=types.SimpleNamespace(get_instance=lambda:types.SimpleNamespace(max_samples_per_step_per_worker=None))
  for cell in fn.__closure__ or ():
   try:value=cell.cell_contents
   except ValueError:continue
   if inspect.isfunction(value):adapt(value)
 adapt(cls.generate_sequences)
 rr=[]
 async def loop(*a,**k):return row(7),[]
 async def filtered(*a):return rr,{'post_filter/num_groups':0}
 obj._run_agent_loop=loop;obj._process_reranker_outputs=filtered
 def batch(validate=False):
  return DataProto.from_dict(tensors={'input_ids':torch.ones(512,2,dtype=torch.long)},non_tensors={'tools_kwargs':np.array([{} for _ in range(512)],dtype=object)},meta_info={'global_steps':13,'validate':validate})
 a=asyncio.run(obj.generate_sequences(batch()));assert len(a[0])==512 and empty_ranker.is_empty(a[1])
 assert empty_ranker._generation.get() is None
 rr[:]=[row(i) for i in range(4)]
 b=asyncio.run(obj.generate_sequences(batch()));assert len(b[0])==512 and len(b[1])==4
 c=asyncio.run(obj.generate_sequences(batch(True)));assert len(c)==512
 assert empty_ranker._generation.get() is None
 # Actual restored U fallback method, including real raw evidence callbacks.
 d=obj._fallback_keep_outputs_to_align({'q':[row(1),row(2)]},4);assert len(d)==4
 try:obj._postprocess([])
 except AssertionError:pass
 else:raise AssertionError('unscoped empty accepted')
 assert not torch.cuda.is_initialized() and not ray.is_initialized()
 assert list((root/'raw-rollouts').glob('*.pt'))
 return {'empty_main_rows':512,'empty_ranker_rows':0,'nonempty_ranker_rows':4,'validation_rows':512,'underfull_rows':4,'context_reset':True,'actual_original_generate_after_restore':True,'CUDA_initialized':False,'Ray_initialized':False,'PPO_updates':0}

if len(sys.argv)>1 and sys.argv[1]=='--child':
 payload=Path(sys.argv[2]);cls=cloudpickle.loads(payload.read_bytes())
 result=run_restored(cls,payload.parent/'child-evidence')
 (payload.parent/'child-result.json').write_text(json.dumps(result))
 print(json.dumps(result),flush=True)
else:
 with tempfile.TemporaryDirectory() as directory:
  root=Path(directory);os.environ['COSEARCH_NATIVE_EVIDENCE_ROOT']=str(root/'parent-evidence')
  actual=CoSearchAgentLoopWorker.__ray_metadata__.modified_class
  observe_worker.install_raw_rollout_observations(SearchR1RerankerAgentLoopWorkerBase,actual)
  fallback_alignment.install(actual,observe_worker.write_record,observe_worker.preserve_raw_payload)
  # Reproduce V's original exact ContextVar serialization failure on a derived
  # actual native modified class, then test W on the actual class itself.
  legacy_path=Path('/dataset/shaoshitong/search-r1-rl/cosearch/author-native/runs/author-native-continue-300-v/tools/empty_ranker.py')
  assert hashlib.sha256(legacy_path.read_bytes()).hexdigest()=='5c2e5043df14d784374d49f158970b2523408a6e6285584143406c55336ebd73'
  spec=importlib.util.spec_from_file_location('_frozen_v_empty_cpu',legacy_path);legacy=importlib.util.module_from_spec(spec);sys.modules[spec.name]=legacy;spec.loader.exec_module(legacy)
  class OldActual(actual):pass
  legacy.install_worker(OldActual,observe_worker.write_record,observe_worker.preserve_raw_payload)
  try:cloudpickle.dumps(OldActual)
  except (TypeError,RuntimeError) as e:assert 'ContextVar' in str(e)
  else:raise AssertionError('legacy ContextVar failure not reproduced')
  empty_ranker.install_worker(actual,observe_worker.write_record,observe_worker.preserve_raw_payload)
  payload=cloudpickle.dumps(actual);path=root/'actual-modified-class.pkl';path.write_bytes(payload)
  restored=cloudpickle.loads(payload)
  parent=run_restored(restored,root/'roundtrip-evidence')
  # Serialize once BEFORE fake backend adaptation and execute that payload in
  # a genuinely fresh interpreter with the immutable same W import path.
  completed=subprocess.run([sys.executable,__file__,'--child',str(path)],capture_output=True,text=True,check=False,timeout=120)
  if completed.returncode:raise AssertionError(completed.stderr[-12000:])
  child=json.loads((root/'child-result.json').read_text());assert child==parent
  assert not torch.cuda.is_initialized() and not ray.is_initialized()
  output={'passed':True,'actual_ray_modified_class_full_hooks':True,'legacy_ContextVar_failure_reproduced':True,'actual_ray_cloudpickle_roundtrip':True,'fresh_interpreter_original_generate_passed':True,'cases':parent,'serialized_bytes':len(payload),'fix_sha256':hashlib.sha256(Path(empty_ranker.__file__).read_bytes()).hexdigest(),'observer_sha256':hashlib.sha256(Path(observe_worker.__file__).read_bytes()).hexdigest(),'CUDA_initialized':False,'Ray_initialized':False,'PPO_updates':0}
  Path(__file__).with_name('cpu-worker-serialization-gate.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output),flush=True)
