import importlib.util,json,pathlib,tempfile,unittest
D=pathlib.Path(__file__).parent;spec=importlib.util.spec_from_file_location('copy_tool',D/'create-cpu-log-path-compatibility.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Copy(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();m.ROOT=pathlib.Path(self.tmp.name).resolve();self.logs=m.ROOT/'logs';self.logs.mkdir();self.run=m.ROOT/'runs/bapo-instruct300-hardware-999';self.run.mkdir(parents=True);A=pathlib.Path('work/evidence/ray-envfix-hardware3287-raw-evidence');raw=(A/'logs/cpu-gate-durable.log').read_bytes();exitraw=(A/'logs/cpu-gate-durable-exit.json').read_bytes();(self.logs/'cpu-gate-durable.log').write_bytes(raw);(self.logs/'cpu-gate-durable-exit.json').write_bytes(exitraw);(m.ROOT/'source-expected.txt').write_text(m.SOURCE+'\n');manifest=(A/'logs/tools-manifest.json').read_bytes();(self.logs/'tools-manifest.json').write_bytes(manifest);approval=json.loads((A/'logs/launch-approval.json').read_bytes());approval['root']=str(m.ROOT)
  for key,name in [('cpu_gate_log','cpu-gate-durable.log'),('cpu_gate_exit_record','cpu-gate-durable-exit.json')]:approval[key]['path']=str(self.logs/name)
  ar=json.dumps(approval).encode();m.APPROVAL_SHA=m.sha(ar);(self.logs/'launch-approval.json').write_bytes(ar);terminal=b'JobId=999 JobState=COMPLETED ExitCode=0:0 DerivedExitCode=0:0 Restarts=0 NodeList=h100-5';(self.run/'slurm-terminal.txt').write_bytes(terminal);ev=[dict(path=str(p),bytes=len(r),sha256=m.sha(r))for p,r in [(self.logs/'cpu-gate-durable.log',raw),(self.logs/'cpu-gate-durable-exit.json',exitraw),(self.logs/'launch-approval.json',ar),(self.run/'slurm-terminal.txt',terminal)]];(self.run/'actual-fullhash-inventory.json').write_text(json.dumps(dict(job_id=999,source_commit=m.SOURCE,run_root=str(self.run),evidence=ev)))
 def tearDown(self):self.tmp.cleanup()
 def test_exact_copy_and_truthful_provenance(self):
  r=m.create(999);self.assertEqual((self.logs/'cpu-gate.log').read_bytes(),(self.logs/'cpu-gate-durable.log').read_bytes());p=json.loads((self.logs/'cpu-gate-log-compatibility-copy.json').read_text());self.assertFalse(p['cpu_rerun']);self.assertFalse(p['gpu_execution']);self.assertEqual(p['source_binding']['sha256'],p['target_binding']['sha256']);self.assertEqual(p['primary_log_path_in_approval'],str(self.logs/'cpu-gate-durable.log'))
 def test_existing_never_overwrite(self):
  p=self.logs/'cpu-gate.log';p.write_text('existing')
  with self.assertRaisesRegex(ValueError,'Exclusive'):m.create(999)
  self.assertEqual(p.read_text(),'existing')
 def test_dangling_symlink_reject(self):
  (self.logs/'cpu-gate.log').symlink_to(self.logs/'missing')
  with self.assertRaisesRegex(ValueError,'Exclusive'):m.create(999)
 def test_tampered_original_reject(self):
  (self.logs/'cpu-gate-durable.log').write_text('bad')
  with self.assertRaisesRegex(ValueError,'originals differ'):m.create(999)
 def test_incomplete_terminal_reject(self):
  (self.run/'slurm-terminal.txt').write_text('JobId=999 JobState=RUNNING')
  with self.assertRaises(ValueError):m.create(999)
 def test_wrong_job_reject(self):
  with self.assertRaises(ValueError):m.create(True)
 def test_existing_provenance_reject(self):
  (self.logs/'cpu-gate-log-compatibility-copy.json').write_text('existing')
  with self.assertRaisesRegex(ValueError,'Exclusive'):m.create(999)
if __name__=='__main__':unittest.main()
