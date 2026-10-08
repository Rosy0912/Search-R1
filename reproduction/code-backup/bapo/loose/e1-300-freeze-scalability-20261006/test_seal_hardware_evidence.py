import base64,hashlib,importlib.util,json,tempfile,unittest
from pathlib import Path
B=Path(__file__).resolve().parent;s=importlib.util.spec_from_file_location('seal',B/'seal-hardware-evidence.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class SealTests(unittest.TestCase):
 def test_terminal_actual_cost_and_strict_success(self):
  raw=b'JobId=3263 JobName=bapo-instruct300-hardware JobState=COMPLETED ExitCode=0:0 DerivedExitCode=0:0 Restarts=0 NodeList=h100-5 RunTime=00:21:01 NumNodes=1 TRES=cpu=192,mem=512G,node=1,billing=192 JOB_GRES=gpu:h100:8 TresPerNode=gres:gpu:h100:8\r\n'
  self.assertEqual(m.terminal_cost(raw,3263),1261)
  for bad in (raw.replace(b'h100-5',b'h100-5,h100-6'),raw.replace(b'COMPLETED',b'COMPLETEDxxx'),raw.replace(b'0:0 Derived',b'1:0 Derived'),raw+b'Restarts=0',raw.replace(b'gpu:h100:8',b'gpu:h100:7')):
   with self.assertRaises(ValueError):m.terminal_cost(bad,3263)
 def test_actual_prior_terminal_formatter_not_new_authorization(self):
  raw=(B.parent/'evidence/recovery3231-hardware3239-terminal-original.txt').read_bytes()
  self.assertEqual(m.terminal_cost(raw,3239),1236)
  with self.assertRaises(ValueError):m.terminal_cost(raw,3263)
 def test_actual3263_allocation_schema_and_eight_visible_physical_bindings(self):
  a=json.loads((B.parent/'evidence/freeze-scalability-hardware3263-allocation-files.json').read_text())
  text=a['slurm_live']['text'];self.assertNotIn('AllocTRES=',text)
  self.assertEqual(m.gpu_terminal_allocation(text)['gpu_count'],8)
  # Actual RUNNING raw cannot satisfy terminal success, despite valid allocation schema.
  with self.assertRaises(ValueError):m.terminal_cost(text.encode(),3263)
  allocation=a['files']['slurm-allocation.txt']['text'];gpus=a['files']['gpus-before.csv']['text']
  uuids=sorted(row['uuid'].strip()for row in m.csv.DictReader(m.io.StringIO(gpus),skipinitialspace=True))
  proof=m.allocated_cost_binding(allocation,gpus,{'allocated_uuids':uuids},{'allocated_uuids':uuids},3263)
  self.assertEqual(len(proof['actual_allocated_uuids']),8)
  with self.assertRaises(ValueError):m.allocated_cost_binding(allocation,gpus,{'allocated_uuids':uuids[:-1]},{'allocated_uuids':uuids},3263)
 def test_allocation_conflicts_and_duplicates_refused(self):
  a=json.loads((B.parent/'evidence/freeze-scalability-hardware3263-allocation-files.json').read_text());text=a['slurm_live']['text']
  for bad in (text+'\nTresPerNode=gres:gpu:h100:8',text+'\nAllocTRES=node=1,gres/gpu=7',text+'\nAllocTRES=gres/gpu=8 AllocTRES=gres/gpu=8',text.replace('NumNodes=1','NumNodes=2'),text.replace('JOB_GRES=gpu:h100:8','JOB_GRES=gpu:h100:7'),text.replace('node=1,billing','node=1,node=1,billing')):
   with self.assertRaises(ValueError):m.gpu_terminal_allocation(bad)
 def test_fullstream_hash_and_original_no_overwrite(self):
  with tempfile.TemporaryDirectory()as d:
   p=Path(d)/'native.pt';raw=b'original'*400000;p.write_bytes(raw)
   self.assertEqual(m.digest(p)['sha256'],hashlib.sha256(raw).hexdigest())
   m.create_or_compare(p,raw)
   with self.assertRaises(ValueError):m.create_or_compare(p,b'changed')
   self.assertEqual(p.read_bytes(),raw)
 def test_directories_not_files_and_fixture_all_raw(self):
  with tempfile.TemporaryDirectory()as d:
   r=Path(d);(r/'__pycache__').mkdir();(r/'__pycache__/tool.pyc').write_bytes(b'cache')
   (r/'freeze-scalability-gpu/tmp').mkdir(parents=True);f=r/'freeze-scalability-gpu/tmp/rank.json';f.write_text('{}')
   (r/'normal.json').write_text('{}');self.assertEqual(set(m.run_paths(r)),{f,r/'normal.json'})
 def test_run_symlink_rejected(self):
  with tempfile.TemporaryDirectory()as d:
   r=Path(d);(r/'raw').write_bytes(b'raw');(r/'link').symlink_to(r/'raw')
   with self.assertRaises(ValueError):m.run_paths(r)
 def test_real_head_byte_binding_and_terminal_replay_refusal(self):
  old=m.ROOT
  with tempfile.TemporaryDirectory()as d:
   m.ROOT=Path(d);self.addCleanup(setattr,m,'ROOT',old);(m.ROOT/'tools').mkdir()
   for n in ('submit.sh','hardware.sbatch'):(m.ROOT/'tools'/n).write_text(n)
   approved={n:m.digest(m.ROOT/'tools'/n)['sha256']for n in ('submit.sh','hardware.sbatch')}
   raws={'hardware-job-id.txt':b'3263\n','hardware-submit-intent.txt':b'hardware\n','submit.sh':b'submit.sh','hardware.sbatch':b'hardware.sbatch','approved-launch-hashes.json':json.dumps(approved).encode(),'hardware-3263-terminal.txt':b'actualterminal'}
   proof={'files':{n:{'path':str(m.ROOT/'scheduler-head'/n),'base64':base64.b64encode(v).decode(),'sha256':hashlib.sha256(v).hexdigest(),'bytes':len(v)}for n,v in raws.items()}}
   m.verify_head(proof,b'actualterminal',3263)
   with self.assertRaisesRegex(ValueError,'watcher terminal'):m.verify_head(proof,b'anotherterminal',3263)
   proof['files']['hardware-job-id.txt']['sha256']='0'*64
   with self.assertRaisesRegex(ValueError,'byte hash'):m.verify_head(proof,b'actualterminal',3263)
if __name__=='__main__':unittest.main()
