"""Same Hydra baseline entry, with an observer in its actual Ray main task."""
import hashlib,importlib.abc,importlib.machinery,os,sys
from validation_cost_hooks_fixed_short import measured_remote
TARGET='verl.trainer.main_ppo'
EXPECTED='95e630c73fc0701738c28f48481ee338bd08aff771b513490260b61c412051b9'

class Loader(importlib.abc.Loader):
 def __init__(self,original):self.original=original
 def create_module(self,spec):return self.original.create_module(spec)
 def exec_module(self,module):
  assert hashlib.sha256(open(module.__file__,'rb').read()).hexdigest()==EXPECTED
  self.original.exec_module(module)
  original=module.main_task
  # The fixed original source uses @ray.remote with no overridden resources.
  assert original._default_options=={}
  module.main_task=measured_remote(original._function)

class Finder(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname!=TARGET:return None
  spec=importlib.machinery.PathFinder.find_spec(fullname,path)
  assert spec is not None and spec.loader is not None
  spec.loader=Loader(spec.loader)
  sys.meta_path.remove(self)
  return spec

if __name__=='__main__':
 assert TARGET not in sys.modules,'Refuse late installation after task import'
 sys.meta_path.insert(0,Finder())
 from search_r1.baselines.main import main
 main()
