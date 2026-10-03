"""Explicit launch-only replacement window for post-start native lease validation.
No model/data/generation/optimizer function is replaced. Main config is untouched.
"""
import argparse,hashlib,inspect,json,marshal,os,runpy,sys,types
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE/'strict-tools'))
from training_evidence import digest
from runtime_binding import load_runtime
from owned_runtime import reassert_owned

def compiled_function_code(path,name):
 module=compile(Path(path).read_text(),str(Path(path).resolve()),'exec',dont_inherit=True,optimize=sys.flags.optimize)
 def find(code):
  for value in code.co_consts:
   if isinstance(value,types.CodeType):
    if value.co_name==name:return value
    nested=find(value)
    if nested is not None:return nested
  return None
 result=find(module)
 if result is None:raise ValueError('Expected frozen function code missing')
 return hashlib.sha256(marshal.dumps(result)).hexdigest()

def actual_code_sha(function):return hashlib.sha256(marshal.dumps(function.__code__)).hexdigest()

class MainLeaseWindow:
 def __init__(self,preflight,record_path,expected_source_sha,binding):
  self.preflight=preflight;self.path=Path(record_path);self.original=preflight.validate_inherited_gpu_leases;self.calls=[];self.binding=binding
  source=Path(inspect.getsourcefile(self.original)).resolve()
  if source!=Path(preflight.__file__).resolve() or digest(source)!=expected_source_sha:raise ValueError('Actual original native lease function source differs')
  code=self.original.__code__
  if Path(code.co_filename).resolve()!=source or self.original.__qualname__!='validate_inherited_gpu_leases':raise ValueError('Actual original native function/code identity differs')
  if actual_code_sha(self.original)!=compiled_function_code(source,'validate_inherited_gpu_leases'):raise ValueError('Actual original function code differs from compiled frozen source')
  self.original_source={'code_sha256':actual_code_sha(self.original),'path':str(source),'sha256':digest(source),'qualname':self.original.__qualname__,'code_filename':str(Path(code.co_filename).resolve()),'function_source_sha256':hashlib.sha256(inspect.getsource(self.original).encode()).hexdigest()}
 def replacement(self,expected_count=None):
  result=reassert_owned(self.preflight,expected_count)
  self.calls.append(result)
 def __enter__(self):
  self.preflight.validate_inherited_gpu_leases=self.replacement
  return self
 def __exit__(self,kind,error,trace):
  intact=self.preflight.validate_inherited_gpu_leases==self.replacement
  self.preflight.validate_inherited_gpu_leases=self.original
  replacement={'code_sha256':actual_code_sha(self.replacement),'path':str(Path(self.replacement.__code__.co_filename).resolve()),'sha256':digest(Path(__file__)),'qualname':self.replacement.__qualname__,'code_filename':str(Path(self.replacement.__code__.co_filename).resolve()),'function_source_sha256':hashlib.sha256(inspect.getsource(self.replacement).encode()).hexdigest()}
  record={'schema':'actual-corrected-main-lease-adapter-v1','original':self.original_source,'replacement':replacement,'runtime_binding':self.binding,'calls':self.calls,'restored_original':self.preflight.validate_inherited_gpu_leases is self.original,'replacement_intact':intact,'main_exception':None if error is None else type(error).__name__}
  with self.path.open('x') as handle:json.dump(record,handle,sort_keys=True)
  if not intact:raise RuntimeError('Native lease adapter replaced during its window')
  if error is None and not self.calls:raise RuntimeError('Native main did not invoke actual lease adapter')
  return False

def run(config_path,record_path):
 bundle,binding=load_runtime(__file__,config_path)
 from search_r1.baselines import corrected_evaluation as ce,preflight
 from omegaconf import OmegaConf
 cfg=OmegaConf.load(config_path);stage=ce.load_stage(cfg.baselines.corrected_evaluation)
 from eval_supervisor import validate_config
 validate_config(cfg,stage)
 repo=Path(ce.__file__).resolve().parents[2]
 if Path(preflight.__file__).resolve()!=repo/'search_r1/baselines/preflight.py':raise ValueError('Actual preflight import source outside pinned corrected repo')
 old_argv=list(sys.argv)
 try:
  sys.argv=['search_r1.baselines.main','--config-path',str(Path(config_path).resolve().parent),'--config-name',Path(config_path).stem]
  with MainLeaseWindow(preflight,record_path,stage['source_files']['search_r1/baselines/preflight.py'],binding):
   runpy.run_module('search_r1.baselines.main',run_name='__main__')
 finally:sys.argv=old_argv

if __name__=='__main__':
 print('BLOCKED launch-only lease adapter preview',file=sys.stderr)
 raise SystemExit(78)
 parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--record',required=True)
 args=parser.parse_args();run(args.config,args.record)
