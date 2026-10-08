import importlib.util,json,tempfile,unittest
from pathlib import Path
B=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('allocation_capture',B/'capture-hardware-allocation.py');c=importlib.util.module_from_spec(s);s.loader.exec_module(c)
class CaptureTest(unittest.TestCase):
 def test_job_argument_strict(self):
  self.assertEqual(c.job_id('9001'),9001)
  for v in ('0','-1','9001;command','1/../../x','1.0','01',' 1'):
   with self.assertRaises(ValueError):c.job_id(v)
 def test_live_slurm_exact_and_duplicates(self):
  raw='JobId=9001 JobName=bapo-instruct300-hardware JobState=RUNNING Restarts=0 NodeList=h100-5\r\n'
  c.validate_slurm(raw,9001)
  for changed in (raw.replace('h100-5','h100-5,h100-6'),raw.replace('RUNNING','RUNNING_suffix'),raw.replace('Restarts=0','Restarts=0foo'),raw+'NodeList=h100-5\n',raw.replace('JobId=9001','JobId=90010')):
   with self.assertRaises(ValueError):c.validate_slurm(changed,9001)
 def test_original_bytes_and_symlink(self):
  with tempfile.TemporaryDirectory()as d:
   p=Path(d)/'raw';p.write_bytes(b'\xff\x00original\n');x=c.read_raw(p)
   self.assertEqual(c.base64.b64decode(x['base64']),p.read_bytes());self.assertNotIn('text',x)
   link=Path(d)/'link';link.symlink_to(p)
   with self.assertRaises(ValueError):c.read_raw(link)
 def test_prepermission_wait_and_input_capture(self):
  old=c.ROOT
  with tempfile.TemporaryDirectory()as d:
   c.ROOT=Path(d).resolve();self.addCleanup(setattr,c,'ROOT',old)
   r=c.ROOT/'runs/bapo-instruct300-hardware-9001';r.mkdir(parents=True)
   for n in c.REQUIRED:(r/n).write_text('raw\n')
   for n in ('source-commit.txt','cpu-gate-source.txt'):(r/n).write_text(c.SOURCE+'\n')
   (r/'slurm-allocation.txt').write_text('SLURM_JOB_ID=9001\n')
   p=c.ROOT/'config.json';p.write_text('{}\n');proof=c.read_raw(p);binding={k:proof[k]for k in ('path','sha256','bytes')}
   m={'target_job_id':9001,'target_source_commit':c.SOURCE,'target_run_root':str(r),'step':25,'next_step':26,'world_size':8,
      'resolved_config':binding,'historical_first5_receipt':binding,'train_prefix':binding,'shared_inputs':[binding]}
   (r/'recovery-manifest.json').write_text(json.dumps(m))
   raw='JobId=9001 JobName=bapo-instruct300-hardware JobState=RUNNING Restarts=0 NodeList=h100-5'
   x=c.capture(9001,raw);self.assertFalse(x['permission_signed']);self.assertFalse(x['actual_gpu']);self.assertEqual(len(x['files']),12)
   (r/'recovery-permission.json').write_text('{}')
   with self.assertRaisesRegex(ValueError,'Permission already'):c.capture(9001,raw)
if __name__=='__main__':unittest.main()
