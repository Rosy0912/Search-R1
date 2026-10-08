"""CPU only new protocol/root and unchanged resource-core counterexamples."""
import ast,hashlib,importlib.util,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
TOOLS=Path(__file__).resolve().parents[1]/'tools';sys.path.insert(0,str(TOOLS))
import diagnostic_contract as contract
s=importlib.util.spec_from_file_location('diagwatch',TOOLS/'resource_watch.py');w=importlib.util.module_from_spec(s);s.loader.exec_module(w)
class DiagnosticAnchor(unittest.TestCase):
 def test_formal_receipt_cannot_authorize_diag(self):
  with self.assertRaisesRegex(ValueError,'Production approvals'):w.validate_fresh200_approval({'protocol':'BAPO_FRESH200_LAUNCH_V1'},w.SOURCE,w.DIAGNOSTIC_ROOT)
 def test_exact_hf_command_role(self):
  expected=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-behavior-20261006/numeric-diagnostic-3310-20261006')
  self.assertEqual(contract.ROOT,expected);self.assertEqual(w.DIAGNOSTIC_ROOT,expected)
  self.assertEqual(contract.SOURCE,'90755a561abd239fbb23882939410e643bb913ae');self.assertEqual(w.SOURCE,contract.SOURCE)
  self.assertEqual(contract.PROTOCOL,'BAPO_ZERO_ADAM_NUMERIC_DIAGNOSTIC_V1');self.assertEqual(contract.PHASE,'numeric-zero-adam-replay')
  import launch_transaction as transaction
  self.assertEqual(transaction.ROOT,expected);self.assertEqual(transaction.PHASES,(contract.PHASE,))
  parent=TOOLS.parent
  for name in ['stage.sh','phase.sh','submit.sh','numeric-zero-adam-replay.sbatch']:
   text=(TOOLS/name).read_text();self.assertIn(str(expected),text);self.assertNotIn('probability-diagnostic-3307-v4-20261006',text)
  for name in ['capture-cpu-runtime.py','run-target-tests.py','verify-diagnostic.py']:
   text=(TOOLS/name).read_text();self.assertIn('from diagnostic_contract import',text);self.assertNotIn('probability-diagnostic-3307-v4-20261006',text)
  def values(path):
   scope={'Path':Path}
   for node in ast.parse(path.read_text()).body:
    if isinstance(node,ast.Assign)and all(isinstance(t,ast.Name)for t in node.targets)and any(t.id in ('ROOT','SOURCE_ROOT','S','R','SOURCE','PHASE','PROTOCOL')for t in node.targets):
     try:value=eval(compile(ast.Expression(node.value),str(path),'eval'),{'__builtins__':{}},scope)
     except NameError:continue
     for target in node.targets:scope[target.id]=value
   return scope
  for name in ['deploy-head-metadata.CANDIDATE.py','mirror-head-approval.CANDIDATE.py']:
   header=values(parent/name);self.assertEqual(header['ROOT'],expected);self.assertEqual(header['SOURCE'],contract.SOURCE)
  local=parent/'deploy-target-cpu.py';header=values(local if local.exists()else parent/'test-reference/deploy-target-header.py')
  self.assertEqual(header['R'],expected);self.assertEqual(header['S'],contract.SOURCE_ROOT)
  self.assertIn('from diagnostic_contract import',(parent/'publish-diagnostic-approval.CANDIDATE.py').read_text())
  inventory=json.loads((parent/'tools-manifest.json'if (parent/'tools-manifest.json').exists()else parent/'logs/tools-manifest.json').read_text())['files']
  self.assertEqual(len(inventory),13);self.assertEqual(set(inventory),{p.name for p in TOOLS.iterdir()if p.is_file()})
  for name,digest in inventory.items():self.assertEqual(hashlib.sha256((TOOLS/name).read_bytes()).hexdigest(),digest)
  run=w.DIAGNOSTIC_ROOT/'runs/example'
  command=['python','-B',str(w.DIAGNOSTIC_ROOT/'tools/numeric-probe.py'),'--execute-gpu']
  self.assertTrue(w.is_trainer_command(command,run));self.assertFalse(w.is_trainer_command(['bash']+command[1:],run));self.assertFalse(w.is_trainer_command(['python','-m','search_r1.baselines.main'],run))
  command[2]='/foreign/numeric-probe.py';self.assertFalse(w.is_trainer_command(command,run))
 def test_wrong_run_root_refused(self):
  with patch.object(contract,'verify',return_value={}):
   with self.assertRaisesRegex(ValueError,'independent diagnostic'):w.load_runtime_anchor('/foreign/runs/test')
 def test_validated_anchor_runtime_binding(self):
  with tempfile.TemporaryDirectory()as t:
   root=Path(t);runtime=root/'runtime.json';exe=Path(sys.executable).resolve();stat=exe.stat()
   record={'executable':{'resolved_path':str(exe),'device':stat.st_dev,'inode':stat.st_ino},'sys_prefix':sys.prefix};runtime.write_text(json.dumps(record))
   b=dict(path=str(runtime),bytes=runtime.stat().st_size,sha256=hashlib.sha256(runtime.read_bytes()).hexdigest())
   with patch.object(w,'DIAGNOSTIC_ROOT',root),patch.object(contract,'verify',return_value={'runtime_executable':b}),patch.object(w,'executable_identity',return_value=record['executable']):
    anchor=w.load_runtime_anchor(root/'runs/test',verify_current=True);self.assertEqual(anchor['sha256'],b['sha256'])
 def test_bool_probe_count_refused(self):
  value=dict(optimizer_writes=0,formal_outer_updates=0,scheduler_updates=0,native_acceptance=False,production_release=False,allocation_gpu_count=8,probe_gpu_count=True)
  with self.assertRaises(ValueError):contract.validate_scope(value)
 def test_bool_zero_optimizer_refused(self):
  value=dict(optimizer_writes=False,formal_outer_updates=0,scheduler_updates=0,native_acceptance=False,production_release=False,allocation_gpu_count=8,probe_gpu_count=1)
  with self.assertRaises(ValueError):contract.validate_scope(value)
  stage=(TOOLS/'stage.sh').read_text()
  self.assertIn('export TRITON_LIBCUDA_PATH="$SOURCE_ROOT/lib/driver" LIBRARY_PATH="$SOURCE_ROOT/lib/driver"',stage)
  with patch.object(contract,'driver_library_identity',return_value={'directory':'/own/lib/driver'}):
   with patch.dict(contract.os.environ,TRITON_LIBCUDA_PATH='/own/lib/driver',LIBRARY_PATH='/own/lib/driver'):
    contract.verify_driver_library({'directory':'/own/lib/driver'},require_env=True)
   with patch.dict(contract.os.environ,TRITON_LIBCUDA_PATH='/foreign',LIBRARY_PATH='/own/lib/driver'):
    with self.assertRaisesRegex(ValueError,'Job-local'):contract.verify_driver_library({'directory':'/own/lib/driver'},require_env=True)
   with self.assertRaisesRegex(ValueError,'bytes changed'):contract.verify_driver_library({'directory':'/tampered'})
  with tempfile.TemporaryDirectory()as t:
   base=Path(t).resolve();real=base/'observed/driver';real.mkdir(parents=True);(real/'libcuda.so').write_bytes(b'actual-driver-fixture')
   alias=base/'own-root';alias.symlink_to(base/'observed',target_is_directory=True)
   record=contract.driver_library_identity(alias/'driver',real)
   self.assertEqual(record['directory'],str(alias/'driver'));self.assertEqual(record['resolved_directory'],str(real));self.assertEqual(record['resolved_path'],str(real/'libcuda.so'));self.assertEqual(record['sha256'],hashlib.sha256(b'actual-driver-fixture').hexdigest())
   with self.assertRaisesRegex(ValueError,'Exact observed'):contract.driver_library_identity(alias/'driver',base/'wrong')
 def test_invalid_anchor_never_starts_phase(self):
  with tempfile.TemporaryDirectory()as t,patch.dict(w.os.environ,SLURM_JOB_ID='999',BAPO_RUN_ID=t),patch.object(w.sys,'argv',['watch','--run',t,'--','python','noop']),patch.object(w,'load_runtime_anchor',side_effect=ValueError('unbound diagnostic')),patch.object(w.subprocess,'Popen')as child:
   with self.assertRaisesRegex(ValueError,'unbound diagnostic'):w.main()
   child.assert_not_called()
 def test_monitor_core_exactly_preserved(self):
  old=Path(__file__).resolve().parents[1]/'test-reference/resource_watch.py'
  self.assertEqual(hashlib.sha256(old.read_bytes()).hexdigest(),'92311badc137b4d5c19820bd5d4bd86ae73e84496f9d065c1f3f5c86ba491b3f')
  def funcs(p):
   text=p.read_text();return {n.name:ast.dump(n,include_attributes=False)for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
  a,b=funcs(old),funcs(TOOLS/'resource_watch.py')
  changed={'is_trainer_command','validate_fresh200_approval','load_runtime_anchor','main'}
  self.assertEqual(set(a),set(b))
  for name in set(a)-changed:self.assertEqual(a[name],b[name],name)
if __name__=='__main__':unittest.main()
