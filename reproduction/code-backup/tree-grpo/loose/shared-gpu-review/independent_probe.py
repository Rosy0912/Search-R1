import ast,hashlib,importlib.util,json,os,sys,tempfile
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'work'),str(ROOT/'work/tree-grpo')]
import tree_shared_supervisor as s
import tree_shared_owned_runtime as runtime
from omegaconf import OmegaConf
from search_r1.baselines import holdout,runtime_audit
checks=[]
with tempfile.TemporaryDirectory() as temp:
 root=Path(temp);(root/'repo').symlink_to(ROOT/'work/tree-grpo',target_is_directory=True)
 def split(v):
  v.data.train_files=str(root/'split/train.parquet');v.baselines.bapo.stage_validation_files=str(root/'split/stage-validation.parquet')
  return {'content_identity':{'sha256':s.CONTENT},'holdout_rows':3393}
 with patch.object(s,'ROOT',root),patch.object(holdout,'prepare_common_holdout',split):
  for stage,steps in [('gate',0),('two',2),('five',5),('formal',500)]:
   run=root/stage;run.mkdir();v,n=s.configuration(run,stage)
   assert n==steps and v.trainer.total_training_steps==steps+1
   assert v.actor_rollout_ref.actor.ppo_micro_batch_size==64 and v.data.val_batch_size==64
   raw=OmegaConf.load(run/'input-config.yaml');actual=OmegaConf.load(run/'resolved.yaml')
   split(raw);assert OmegaConf.to_container(raw,resolve=True)==OmegaConf.to_container(actual,resolve=True)
   sha=hashlib.sha256((run/'resolved.yaml').read_bytes()).hexdigest()
   env={'SEARCH_R1_BASELINE_AUDIT_DIR':str(run/'entry'),'SEARCH_R1_BASELINE_CONFIG_SHA256':sha,'SEARCH_R1_BASELINE_DATA_IDENTITY':s.CONTENT}
   with patch.dict(os.environ,env):runtime_audit.config_boundary(raw,split(raw))
   assert (run/'entry/entry-config.yaml').read_bytes()==(run/'resolved.yaml').read_bytes()
   raw.actor_rollout_ref.actor.ppo_micro_batch_size=8
   with patch.dict(os.environ,env):
    try:runtime_audit.config_boundary(raw,split(raw));raise AssertionError('contamination accepted')
    except ValueError:pass
 checks.append('four real Hydra stage configurations: post-holdout exact config bytes; micro64/val64/budget; micro8 contamination rejected')
 # Execute ACTUAL finally AST with write-failure: cleanup still runs and state fails.
 tree=ast.parse((ROOT/'work/tree_shared_supervisor.py').read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main');final=next(n for n in main.body if isinstance(n,ast.Try) and n.finalbody).finalbody
 calls=[]
 class Own:
  def cleanup(self):calls.append('cleanup')
 class Signal:
  SIGTERM=15;SIGINT=2;SIG_IGN=1
  @staticmethod
  def signal(*a):pass
 class Run:
  def __truediv__(self,name):return File(name)
 class File:
  def __init__(self,name):self.name=name
  def write_text(self,text):
   if self.name=='termination-cost.json':raise OSError('intentional disk failure')
   calls.append(self.name)
 class Alias:
  def unlink(self,**kwargs):calls.append('alias-cleanup')
 import time,types
 state={'status':'REAL_LOCAL_GPU_RETRIEVAL_GATE_READY_FOR_REVIEW'}
 namespace={'signal':Signal,'run':Run(),'job':'123','a':types.SimpleNamespace(stage='gate'),'start':time.time(),'time':time,'json':json,'owned':Own(),'result':state,'alias':Alias()}
 try:exec(compile(ast.Module(body=final,type_ignores=[]),'<actual-finally>','exec'),namespace);raise AssertionError('disk failure hidden')
 except OSError:pass
 assert calls==['cleanup','alias-cleanup','run-result.json','cost.json'] and state['status']=='FAILED'
 checks.append('actual supervisor finally disk-write failure: cleanup+alias always execute; FAILED status persists')
 # Actual guard exception handler (not NVML/GPU): logging failure cannot skip own cancellation.
 guard=ast.parse((ROOT/'work/resource_guard.py').read_text());handler=next(n for n in guard.body if isinstance(n,ast.Try)).handlers[0]
 class Sub:
  @staticmethod
  def run(args,**kwargs):calls.append(args)
 def fail(record):raise OSError('intentional log failure')
 import traceback
 ns={'append':fail,'dict':dict,'time':time,'job':'123','traceback':traceback,'subprocess':Sub}
 try:exec(compile(ast.Module(body=handler.body,type_ignores=[]),'<guard-except>','exec'),ns)
 except OSError:pass
 assert calls[-1]==['scancel','123'];checks.append('actual guard error handler: failed log still cancels ONLY own job')
 # Linux pidfd cleanup mocked, no real process signal. Baseline and neighboring scope protected.
 scope=root/'cg';scope.mkdir();(scope/'cgroup.procs').write_text('100\n')
 own=runtime.OwnedProcesses(scope,'/slurm/job_123/step_0');signals=[];current=[100,200,300]
 original=Path.read_text
 def read(p,*a,**k):
  if str(p)=='/proc/200/cgroup':return '0::/slurm/job_123/step_0/child\n'
  if str(p)=='/proc/300/cgroup':return '0::/slurm/job_123/step_01\n'
  return original(p,*a,**k)
 def send(fd,sig):signals.append((fd,sig));current.remove(fd)
 def sleep(_):
  if 300 in current:current.remove(300) # exited foreign snapshot race
 with patch.object(runtime,'cgroup_pids',lambda scope:set(current)),patch.object(Path,'read_text',read),patch.object(runtime.os,'pidfd_open',lambda pid:pid,create=True),patch.object(runtime.os,'close',lambda fd:None),patch.object(runtime.signal,'pidfd_send_signal',send,create=True),patch.object(runtime.time,'sleep',sleep):own.cleanup()
 assert signals==[(200,runtime.signal.SIGTERM)] and current==[100]
 checks.append('actual cleanup mocked pidfd: signals new own nested child only; baseline/neighbor step prefix protected')
# Exact copied standalone fixture AST identity for real proof.
original=ast.parse((ROOT/'work/test_validation_cost_hooks.py').read_text());copied=ast.parse((ROOT/'work/validation_cost_ray_fixture.py').read_text())
def defs(tree):return {n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
o,c=defs(original),defs(copied)
for name in c:assert name in o and c[name]==o[name],name
proof=json.loads((ROOT/'outputs/evidence/validation-cost-target-ray-2902-proof.json').read_text())
assert proof['ray_version']=='2.40.0' and proof['off']['result']==proof['on']['result'] and proof['off']['rng_torch_sha256']==proof['on']['rng_torch_sha256'] and proof['entry_pid']!=proof['on']['pid']
assert next(iter(proof['source_sha256'].values()))==hashlib.sha256((ROOT/'work/probe_validation_cost_ray.py').read_bytes()).hexdigest()
checks.append('standalone copied fixture AST exact; target Ray2.40 proof current sourceSHA/result/RNG/remotePID verified; not production loader/full validation proof')
files=['tree_shared_supervisor.py','tree_shared_owned_runtime.py','resource_guard.py','probe_tree_shared_retrieval.py','start_tree_shared.sh','validation_cost_ray_fixture.py','probe_validation_cost_ray.py']
print(json.dumps({'status':'PASS','scope':'Independent local CPU fixtures only; no Slurm/GPU/network service or real signal','checks':checks,'source_sha256':{f:hashlib.sha256((ROOT/'work'/f).read_bytes()).hexdigest() for f in files}},indent=2))
