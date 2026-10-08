"""CPU constructor/entropy provenance probe; teacher loading is explicitly mocked.
No training, generation, GPU, import-checker policy edit, receipt, or deployment.
Legacy compile uses backend=eager to exercise real Dynamo graph-break transformation.
"""
import argparse,hashlib,importlib.util,json,os,sys,types
from pathlib import Path
SOURCE='89beac273f783d8e48d7d3e34da040f753514d90'
ACTOR_SHA='a9e5005f9d5e48f2cba652ee88b2ff2f3b0bb671dde8ce70ea7653eeb82c5ed9'
def require(ok,message):
 if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def run(repo,candidate,leg):
 require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Explicit CPU visibility required')
 sys.path.insert(0,str(repo));import torch
 from omegaconf import OmegaConf
 import verl.utils.torch_functional as f
 from verl.workers.actor.dp_actor import DataParallelPPOActor
 require(not torch.cuda.is_initialized(),'CUDA already initialized')
 import subprocess
 require(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==SOURCE,'Wrong actual frozen source')
 checker=repo/'reproduction/baselines/e1_import_bootstrap.py';spec=importlib.util.spec_from_file_location('probe_source_checker',checker);ck=importlib.util.module_from_spec(spec);spec.loader.exec_module(ck)
 allowed={str(checker.resolve()):ck.compiled_codes(checker)}
 for name in ck.LIBRARY_MODULES:
  path=ck.source_path(repo,name);allowed[str(path)]=ck.compiled_codes(path)
 path=Path(f.__file__).resolve();allowed[str(path)]=ck.compiled_codes(path)
 before=ck.check_live_code(f,allowed);names_before=set(vars(f));real_compile=torch.compile;compile_calls=[]
 # Same Dynamo frontend as the inherited constructor. Backend=eager avoids
 # CPU Inductor toolchain requirements and does not represent a GPU benchmark.
 def compile_cpu(fn,*args,**kw):
  compile_calls.append({'original_entropy':fn is f.entropy_from_logits,'dynamic':kw.get('dynamic')});return real_compile(fn,*args,backend='eager',**kw)
 torch.compile=compile_cpu
 if leg=='candidate':
  require(sha(candidate)==ACTOR_SHA,'Candidate bytes differ');s=importlib.util.spec_from_file_location('search_r1.baselines.entropy_eager_candidate',candidate);module=importlib.util.module_from_spec(s);s.loader.exec_module(module)
 else:
  import search_r1.baselines.actor as module
 class TeacherFixture:
  def __init__(self,*args):pass
 module.FrozenTeacher=TeacherFixture
 config=OmegaConf.create({'use_remove_padding':True,'ulysses_sequence_parallel_size':1,'use_kl_loss':False,'entropy_coeff':0.,'distillation':{'teacher_path':'EXPLICIT_CPU_TEACHER_FIXTURE','teacher_device':'cpu'}})
 actor=module.DistillationActor(config,torch.nn.Linear(2,2),None,None);torch.compile=real_compile
 require(compile_calls==[{'original_entropy':True,'dynamic':True}],'Actual inherited constructor selection differs')
 selected=actor.compute_entropy_from_logits
 inputs=torch.arange(52,dtype=torch.float32).reshape(4,13)/7.;x=inputs.clone().requires_grad_();y=inputs.clone().requires_grad_();a=selected(x);b=f.entropy_from_logits(y);a.sum().backward();b.sum().backward()
 require(torch.equal(a,b) and torch.equal(x.grad,y.grad),'Entropy formula/gradient changed')
 rejected=[]
 for bad in (float('nan'),float('inf')):
  t=inputs.clone();t[0,0]=bad
  try:selected(t)
  except ValueError:rejected.append(str(bad))
 require(len(rejected)==2,'Nonfinite logits accepted')
 generated={k:{'filename':v.__code__.co_filename,'name':v.__code__.co_name} for k,v in vars(f).items() if isinstance(v,types.FunctionType) and ('torch_dynamo_resume' in v.__qualname__ or k.startswith('__resume_at_'))}
 try:after=ck.check_live_code(f,allowed);error=None
 except ValueError as e:after=None;error=str(e)
 if leg=='candidate':
  require(selected is f.entropy_from_logits and not generated and set(vars(f))==names_before and error is None,'Eager OPSD still injected runtime code or fails strict checker')
  # An actually tampered source function must still fail the unchanged checker.
  original=f.entropy_from_logits;ns={'torch':torch};exec(compile('def entropy_from_logits(logits):\n return logits.sum(-1)\n',str(path),'exec'),ns);f.entropy_from_logits=ns['entropy_from_logits']
  try:ck.check_live_code(f,allowed)
  except ValueError:tamper_rejected=True
  else:tamper_rejected=False
  finally:f.entropy_from_logits=original
  require(tamper_rejected,'Source tamper bypassed strict checker')
 else:
  tamper_rejected=None;require(error is not None and 'torch_dynamo_resume' in error and bool(generated),'Legacy Dynamo source failure did not reproduce')
 require(not torch.cuda.is_initialized(),'Probe initialized CUDA')
 return {'scope':'CPU_ACTUAL_CONSTRUCTOR_WITH_MOCK_TEACHER_NOT_MODEL_OR_TRAINING_ACCEPTANCE','source_commit':SOURCE,'leg':leg,'torch_version':torch.__version__,'candidate_actor_sha256':ACTOR_SHA if leg=='candidate' else None,'compile_calls':compile_calls,'eager_selection':selected is f.entropy_from_logits,'entropy_values_exact':True,'entropy_gradients_exact':True,'nonfinite_rejected':rejected,'before_function_count':before,'after_function_count':after,'generated_functions':generated,'strict_guard_error':error,'tamper_rejected':tamper_rejected,'cuda_is_initialized':False}
def main():
 p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--candidate',required=True);p.add_argument('--leg',choices=['legacy','candidate'],required=True);p.add_argument('--output',required=True);a=p.parse_args();result=run(Path(a.repo).resolve(),Path(a.candidate).resolve(),a.leg)
 with Path(a.output).open('x') as f:json.dump(result,f,sort_keys=True,indent=2)
 print(json.dumps(result,sort_keys=True))
if __name__=='__main__':main()
