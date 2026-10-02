import ast,json,os,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
source=(ROOT.parent/'runtime-20261003/resource_sample.py').read_text()
tree=ast.parse(source);ns={'re':re,'os':os,'job':'12'}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),'runtime_sampler_functions','exec'),ns)
assert ns['owned_cgroup']('0::/slurm/job_12/step_0')
assert not ns['owned_cgroup']('0::/slurm/job_123/step_0')
class FakePath:
 def __init__(self,path):self.path=str(path)
 def read_text(self):
  if self.path.endswith('status'):return 'Name:\tpython\nUid:\t1000 1000 1000 1000\nVmRSS:\t2048 kB\nVmHWM:\t4096 kB\n'
  if self.path.endswith('cgroup'):return '0::/slurm/job_12/step_0'
  fields=['0']*52;fields[11]='7';fields[12]='9';fields[19]='99'
  return '99 (cpu retriever) '+ ' '.join(fields)
ns['Path']=FakePath;actual=ns['usage'](99)
assert actual['cpu_user_ticks']==7 and actual['cpu_system_ticks']==9 and actual['process_start_ticks']=='99'
class GonePath(FakePath):
 def read_text(self):raise FileNotFoundError(self.path)
ns['Path']=GonePath;missing=ns['usage'](99)
assert missing=={'pid':99,'unavailable':True}
print(json.dumps({'cpu_process_fixture':actual,'exact_job_component_matching':True,'process_disappearance_does_not_abort_sampler':missing,'scope':'AST fixture only; no Linux/proc/GPU runtime sampling'},indent=2))
