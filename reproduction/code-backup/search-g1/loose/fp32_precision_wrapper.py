"""Isolated complete-FP32 candidate; original parity tolerance unchanged."""
import os,json
from pathlib import Path
import torch
from omegaconf import open_dict
import reproduction.common8.model_smoke as smoke
original_resolved=smoke.resolved


def resolved(base):
 run,c,args=original_resolved(base)
 with open_dict(c):
  c.actor_rollout_ref.actor.fsdp_config.model_dtype='fp32'
  c.actor_rollout_ref.actor.fsdp_config.mixed_precision=dict(param_dtype='fp32',reduce_dtype='fp32',buffer_dtype='fp32')
  c.actor_rollout_ref.ref.fsdp_config.model_dtype='fp32'
  c.actor_rollout_ref.rollout.dtype='float32'
 return run,c,args
smoke.resolved=resolved
original_autocast=torch.autocast


def autocast(*args,**kwargs):
 device=args[0] if args else kwargs.get('device_type')
 if device=='cuda':kwargs['enabled']=False
 return original_autocast(*args,**kwargs)
torch.autocast=autocast
# TF32/reduced-mantissa math is not part of this full-FP32 candidate.
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
from verl.workers.fsdp_workers import ActorRolloutRefWorker
original_init=ActorRolloutRefWorker.init_model


def initialize(self):
 result=original_init(self)
 role='actor' if self._is_actor else 'reference'
 module=self.actor_module_fsdp if self._is_actor else self.ref_module_fsdp
 p=Path(os.environ['SEARCH_G1_BASE'])/'logs'/('precision-rank-'+os.environ['RANK']+'.jsonl')
 row=dict(kind='actual_fp32_initialization',rank=int(os.environ['RANK']),role=role,
  parameter_dtypes=sorted({str(v.dtype) for v in module.parameters()}),
  buffer_dtypes=sorted({str(v.dtype) for v in module.buffers()}),
  fsdp_mixed_param_dtype=str(module.mixed_precision.param_dtype),
  autocast_cuda_enabled=False,allow_tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
  requested_rollout_dtype=self.config.rollout.dtype,actual_grpo_updates=0,
  semantics='Isolated precision candidate; official same model/data/recipe budgets, actual resolved dtype overrides')
 if self._is_rollout:
  model=self.rollout.inference_engine.llm_engine.model_executor.worker.model_runner.model
  row['actual_rollout_parameter_dtypes']=sorted({str(v.dtype) for v in model.parameters()})
 with p.open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
 return result
ActorRolloutRefWorker.init_model=initialize
original_compare=smoke.compare


def compare(*args,**kwargs):
 old,ref,mask,tokens,raw,responses=args;valid=mask.cpu().bool();raw_valid=valid[:,:raw.shape[1]]
 row=dict(kind='actual_fp32_parity',rank=int(os.environ['RANK']),actor_reference_max_abs=float((old.cpu()[valid].float()-ref.cpu()[valid].float()).abs().max()),
  actor_vllm_max_abs=float((old.cpu()[:,:raw.shape[1]][raw_valid].float()-raw.cpu()[raw_valid].float()).abs().max()),tolerance=smoke.LOGPROB_ATOL,
  parameter_writes=0,precision_candidate_only=True)
 p=Path(os.environ['SEARCH_G1_BASE'])/'logs'/('precision-rank-'+os.environ['RANK']+'.jsonl')
 with p.open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
 return original_compare(*args,**kwargs)
smoke.compare=compare
smoke.main(os.environ['SEARCH_G1_BASE'],'initial')
