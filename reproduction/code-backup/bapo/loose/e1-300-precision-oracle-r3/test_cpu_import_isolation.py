"""Real competing installed-old/new package trees; no production/target proof."""
import importlib.util,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
LAUNCHER=Path(__file__).parent/'tools/cpu-tests.py'
CRITICAL=('verl','search_r1','verl.protocol','verl.workers.actor.dp_actor','verl.trainer.ppo.core_algos','search_r1.baselines.distillation','search_r1.baselines.objectives','search_r1.baselines.full_state','search_r1.baselines.checkpoint_gate','search_r1.baselines.gradient_norm')
class Isolation(unittest.TestCase):
 def packages(self,root,marker):
  for name in CRITICAL:
   parts=name.split('.');p=root
   for part in parts[:-1]:p=p/part;p.mkdir(exist_ok=True);(p/'__init__.py').write_text('TREE='+repr(marker))
   if name in ('verl','search_r1'):p=root/name;p.mkdir(exist_ok=True);(p/'__init__.py').write_text('TREE='+repr(marker))
   else:(p/(parts[-1]+'.py')).write_text('TREE='+repr(marker))
 def run_case(self,kind):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d).resolve();old=root/'installed-old';new=root/'new-repo';old.mkdir();new.mkdir();self.packages(old,'OLD');self.packages(new,'NEW')
   code="import sys,importlib.util,json,types;from pathlib import Path;sys.path.insert(0,"+repr(str(old))+");spec=importlib.util.spec_from_file_location('launcher',"+repr(str(LAUNCHER.resolve()))+");m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);repo=Path("+repr(str(new))+")\n"
   if kind=='preloaded':code+='import verl\n'
   code+='m.isolate_repo(repo)\nbefore=m.namespace_origins(repo)\n'
   if kind=='after':code+="fake=types.ModuleType('verl.trainer.ppo.core_algos');fake.__file__="+repr(str(old/'verl/trainer/ppo/core_algos.py'))+";sys.modules[fake.__name__]=fake\n"
   code+='after=m.namespace_origins(repo)\nprint(json.dumps(dict(before=before,after=after)))\n'
   p=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
   if kind=='good':
    self.assertEqual(p.returncode,0,p.stderr);v=json.loads(p.stdout);self.assertEqual(set(v['before']),set(v['after']))
    for record in v['before'].values():self.assertIn(str(new),record['path']);self.assertEqual(len(record['sha256']),64)
   else:self.assertNotEqual(p.returncode,0);self.assertIn('Outside repository import',p.stderr)
 def test_old_installed_path_shadowed_by_newrepo_priority(self):self.run_case('good')
 def test_old_preloaded_namespace_refused(self):self.run_case('preloaded')
 def test_after_test_origin_contamination_refused(self):self.run_case('after')
 def test_semantic_code_digest_stable_after_execution_and_detects_same_path_replacement(self):
  spec=importlib.util.spec_from_file_location('launcher',LAUNCHER);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  ns={};exec(compile('def f(x): return x + 1','/owned/current.py','exec'),ns);fn=ns['f'];before=m.code_digest(fn.__code__)
  for i in range(1000):fn(i)
  self.assertEqual(before,m.code_digest(fn.__code__))
  ns={};exec(compile('def f(x): return x + 2','/owned/current.py','exec'),ns);self.assertNotEqual(before,m.code_digest(ns['f'].__code__))
 def test_cached_old_function_alias_rejected_with_current_module_path(self):
  repo=(LAUNCHER.parent.parent.parent/'bapo-e1-300-devicefix').resolve()
  code="import importlib.util;from pathlib import Path;spec=importlib.util.spec_from_file_location('launcher',"+repr(str(LAUNCHER.resolve()))+");m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);repo=Path("+repr(str(repo))+");m.isolate_repo(repo);m.namespace_origins(repo);m.function_origins(repo)\n"
  code+="import verl.workers.actor.dp_actor as actor;ns={};exec(compile('def old(self): return None','/old/repo/verl/workers/actor/dp_actor.py','exec'),ns);ns['old'].__module__=actor.__name__;actor.DataParallelPPOActor.update_policy=ns['old'];m.function_origins(repo)\n"
  p=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
  self.assertNotEqual(p.returncode,0);self.assertIn('Outside function code',p.stderr)
if __name__=='__main__':unittest.main()
