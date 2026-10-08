import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='4'
import importlib.util,sys,types,pathlib,json,tempfile
import torch
torch.set_num_threads(4)
from unittest.mock import patch
from types import SimpleNamespace
base=pathlib.Path('work/tree-ppo-three-guards-review/sampling-interface')
# Load exact frozen review bytes; original dependencies only for CPU serialization.
import search_r1.baselines.training_state
spec=importlib.util.spec_from_file_location('search_r1.baselines.review_sampling',base/'reviewed-snapshot/search_r1/baselines/sampling_consistency.py');M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
neutral=dict(temperature=1.,top_p=1.,top_k=-1,min_p=0.,presence_penalty=0.,frequency_penalty=0.,repetition_penalty=1.,n=1,logits_processors=[])
proof={}
# Wrong sampled shape silently broadcasts, so the comparable-token identity is unproved.
x=M.reduce_pair_statistics(torch.zeros(1),torch.zeros(2,3),torch.ones(2,3),neutral,1.,.01)
assert x['allowed'] and x['valid_tokens']==6;proof['wrong_shape_silently_allowed']=x
x=M.reduce_pair_statistics(torch.tensor([-1e200],dtype=torch.float64),torch.zeros(1,dtype=torch.float64),torch.ones(1),neutral,1.,.01)
# positive safe delta excluded k3 but square overflow still leaks Inf.
try:json.dumps(x,allow_nan=False);raise AssertionError('expected nonfinite diagnostic')
except ValueError:proof['nonfinite_diagnostic_from_finite_inputs']=repr(x)
class Batch:
 def __init__(self):self.batch={'token_level_rewards':torch.ones(8,3)};self.meta_info={}
with tempfile.TemporaryDirectory() as directory,patch.dict(os.environ,{'SEARCH_R1_BASELINE_OUTPUT':directory,'SEARCH_R1_BASELINE_COMMIT':'sourceA','SEARCH_R1_BASELINE_CONFIG_SHA256':'configA','SEARCH_R1_BASELINE_DATA_IDENTITY':'dataA','SEARCH_R1_MODEL_IDENTITY':'modelA','SLURM_JOB_ID':'123'}):
 root=pathlib.Path(directory)/'probability-calibration';root.mkdir()
 for rank in range(8):
  (root/f'rank-{rank:02d}.json').write_text(json.dumps(dict(rank=rank,world_size=8,optimizer_committed=0,scheduler_advanced=False,scheduler_step=110,diagnostic_records=['nonexistent-unhashed.json'],code_commit='wrong-source',config_sha256='wrong-config',parent_completed_outer=100,observed_outer=105,restored_optimizer_steps=[0])))
 config=SimpleNamespace(actor_rollout_ref=SimpleNamespace(actor=SimpleNamespace(policy_safety=SimpleNamespace(enabled=True)),rollout=SimpleNamespace(temperature=1.,sampling_consistency=SimpleNamespace(enabled=True,enforce=False))))
 trainer=SimpleNamespace(config=config,actor_rollout_wg=SimpleNamespace(calibrate_policy_safety=lambda _:None),global_steps=111,actual_outer_updates=110,checkpoint_sampler=SimpleNamespace(cursor=58240),recovery_provenance={'parent_completed_step':110,'parent_cursor':57600})
 x=M.finalize_calibration(trainer,Batch());assert x['status']=='CALIBRATION_OBSERVED_ZERO_WRITES';proof['wrong_worker_identity_and_missing_diagnostic_accepted']=True
(base/'INDEPENDENT_COUNTEREXAMPLES.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof,indent=2))
