"""External observation wrappers; original actor/math functions called unchanged."""
import json,os,time
from pathlib import Path
from contracts import require,release_and_ack
from lifecycle import publish
from verl.workers.fsdp_workers import ActorRolloutRefWorker as OriginalWorker
from verl.single_controller.base.decorator import register,Dispatch


def kv_workers(engine):
 executor=engine.llm_engine.model_executor
 workers=[]
 for name in ('worker','driver_worker','workers'):
  value=getattr(executor,name,None)
  if value is not None:workers.extend(value if isinstance(value,list) else [value])
 require(workers,'Actual local vLLM workers not discoverable')
 return workers

def kv_state(engine):
 rows=[]
 for w in kv_workers(engine):
  # Pinned vLLM WorkerWrap exposes worker; failclosed if actual attributes differ.
  w=getattr(w,'worker',w)
  require(hasattr(w,'cache_engine') and hasattr(w,'gpu_cache'),'Actual vLLM KV fields missing')
  rows.append(dict(cache_engine_is_none=w.cache_engine is None,gpu_cache_is_none=w.gpu_cache is None))
 return rows

class ObservedWorker(OriginalWorker):
 def _record(self,event,outer_step=None):
  import torch
  from topd32b_shared_pipeline.protocol import process_identity
  identity=process_identity(os.getpid())
  device=torch.cuda.current_device()
  # UUID determined from actual current device properties, never logical rank assumption.
  import subprocess
  apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
  seen=sorted({line.split(',')[0].strip() for line in apps.splitlines() if int(line.split(',')[1])==os.getpid()})
  require(len(seen)==1,'Actor PID must occupy exactly its actual leased UUID')
  engine=self.rollout.inference_engine;model=kv_workers(engine)[0];model=getattr(model,'worker',model)
  parameter_count=sum(p.numel() for p in model.model_runner.model.parameters())
  row=dict(event=event,outer_step=outer_step,rank=int(torch.distributed.get_rank()),pid=os.getpid(),identity=identity,
   job=os.environ['SLURM_JOB_ID'],gpu_uuid=seen[0],actual_gpu_uuids=seen,initialized=True,
   vllm_model_parameter_count=parameter_count,allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved(),
   kv=kv_state(engine),epoch=time.time())
  directory=Path(os.environ['TOPD32B_RUN'])/'workers';directory.mkdir(exist_ok=True)
  with (directory/(str(row['rank'])+'.jsonl')).open('a') as f:f.write(json.dumps(row,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
  return row
 @register(dispatch_mode=Dispatch.ONE_TO_ALL)
 def observed_ready(self):return self._record('INITIALIZED_ACTOR_VLLM_WEIGHTS')
 @register(dispatch_mode=Dispatch.ONE_TO_ALL)
 def observed_phase(self):return self._record('ACTUAL_PHASE_ALLOCATOR_KV')
 @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
 def update_actor(self,data):
  binding=json.loads(Path(os.environ['TOPD32B_BINDING']).read_text())
  release_and_ack(binding['ipc_directory'],int(data.meta_info['topd_outer_step']),binding)
  self._record('ALL8_RELEASED_ACKED_BEFORE_ACTOR_UPDATE',int(data.meta_info['topd_outer_step']))
  result=super().update_actor(data)
  self._record('ORIGINAL_ACTOR_UPDATE_RETURNED',int(data.meta_info['topd_outer_step']));return result
 @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
 def generate_sequences(self,prompts):
  self._record('BEFORE_ORIGINAL_ROLLOUT')
  engine=self.rollout.inference_engine;original=engine.generate
  def observe(*args,**kwargs):
   self._record('ACTUAL_REBUILT_KV_BEFORE_VLLM_GENERATE');return original(*args,**kwargs)
  engine.generate=observe
  try:return super().generate_sequences(prompts)
  finally:
   engine.generate=original;self._record('AFTER_ORIGINAL_ROLLOUT_KV_RELEASE')


def install(mode,config):
 import verl.workers.fsdp_workers as workers
 import verl.trainer.ppo.ray_trainer as rt
 workers.ActorRolloutRefWorker=ObservedWorker
 original_fit=rt.RayPPOTrainer.fit
 def fit(trainer):
  if mode=='formal500':require(len(trainer.train_dataloader)*trainer.config.trainer.total_epochs>=500,'Actual epoch/dataloader budget cannot produce500 outer updates')
  receipts=trainer.actor_rollout_wg.observed_ready()
  publish(Path(os.environ['TOPD32B_RUN'])/'global-worker-ready.json',dict(receipts=receipts,epoch=time.time()))
  from lifecycle import wait
  wait(Path(os.environ['TOPD32B_RUN'])/'global-worker-authorized.json',[],[],300)
  if mode=='profile':
   from hardware_profile import run
   return run(trainer)
  return original_fit(trainer)
 rt.RayPPOTrainer.fit=fit
 # Fixed common-stage64 diagnostics use source validator; training reward untouched.
 from search_r1.baselines.controller import Controller
 original_stage=Controller.stage_validation
 def stage(controller,step):
  if controller.method!='topd':return original_stage(controller,step)
  trainer=controller.trainer
  import pandas as pd
  from search_r1.baselines.controller import phase_validation_frame
  from verl.utils.dataset.rl_dataset import RLHFDataset,collate_fn
  from torch.utils.data import DataLoader
  old=trainer.val_dataloader
  dataset=RLHFDataset(parquet_files=config.baselines.bapo.stage_validation_files,tokenizer=trainer.tokenizer,prompt_key=config.data.prompt_key,max_prompt_length=config.data.max_prompt_length,filter_prompts=True,return_raw_chat=config.data.get('return_raw_chat',False),truncation='error')
  dataset.dataframe=phase_validation_frame(dataset.dataframe,64,config.baselines.seed)
  trainer.val_dataloader=DataLoader(dataset,batch_size=64,shuffle=False,drop_last=False,collate_fn=collate_fn)
  try:
   metrics=trainer._validate();require(int(metrics['val/qa_count/all'])==64,'Stage diagnostics must actually evaluate64')
   with (Path(os.environ['TOPD32B_RUN'])/'stage64.jsonl').open('a') as f:f.write(json.dumps(dict(step=step,metrics=metrics))+'\n')
   return {'stage/'+k:v for k,v in metrics.items()}
  finally:trainer.val_dataloader=old
 Controller.stage_validation=stage
