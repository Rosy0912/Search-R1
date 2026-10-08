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
  run=w.DIAGNOSTIC_ROOT/'runs/example'
  command=['python','-B',str(w.DIAGNOSTIC_ROOT/'tools/hf-probe.py'),'--execute-gpu']
  self.assertTrue(w.is_trainer_command(command,run));self.assertFalse(w.is_trainer_command(['bash']+command[1:],run));self.assertFalse(w.is_trainer_command(['python','-m','search_r1.baselines.main'],run))
  command[2]='/foreign/hf-probe.py';self.assertFalse(w.is_trainer_command(command,run))
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
