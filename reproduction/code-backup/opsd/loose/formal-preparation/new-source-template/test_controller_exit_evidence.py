import copy,hashlib,json,sys,tempfile,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent)]
from controller_exit_evidence import verify_controller_exit
from verify_formal_prerequisites import digest,loads,own_file,verify_slurm
ROOT=HERE.parents[1]
COMMIT='8e60976d364ab7d7ae51714415813c1359dafe9a'
class Tests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.base=Path(self.temp.name)
  self.origin=loads((ROOT/'opsd/reproduction/baselines/opsd-validation-20261003/evidence/job2994_controller_reviewer_raw.json').read_text())
  self.data=dict(proof_kind='controller_normal_exit',source_commit=COMMIT,job_id='2994',node='h100-10',cpus=192,controller_host='h100-16',source_log_path='/var/log/slurm/slurmctld.log')
  self.put('controller_origin',json.dumps(self.origin))
  self.put('start_snapshot',(ROOT/'job2994-slurm-start.txt').read_text())
  # Synthetic config fixture only; not actual acceptance evidence.
  self.put('head_config','MinJobAge = 300\nAccountingStorageType = accounting_storage/none\nJobCompType = jobcomp/none\nSlurmctldLogFile = /var/log/slurm/slurmctld.log\nSlurmctldHost = h100-16\n')
  self.put('source_commit_evidence',COMMIT+'\n')
 def tearDown(self):self.temp.cleanup()
 def put(self,key,text):
  p=self.base/key;p.write_text(text);self.data[key]={'path':str(p),'sha256':digest(p)}
 def check(self):return verify_controller_exit(self.data,'2994',COMMIT,self.base,own_file,digest,loads)
 def lines(self,change):
  change(self.origin['files']['/var/log/slurm/slurmctld.log']['matched_lines'])
  src=self.origin['files']['/var/log/slurm/slurmctld.log'];src['matched_lines_sha256']=hashlib.sha256(('\n'.join(src['matched_lines'])+'\n').encode()).hexdigest()
  self.put('controller_origin',json.dumps(self.origin))
 def test_valid_explicit_proof(self):self.assertFalse(self.check()['scontrol_terminal_record_present'])
 def test_missing_done(self):
  self.lines(lambda a:a.pop())
  with self.assertRaises(AssertionError):self.check()
 def test_nonzero_exit(self):
  self.lines(lambda a:a.__setitem__(2,a[2].replace('WEXITSTATUS 0','WEXITSTATUS 1')))
  with self.assertRaises(AssertionError):self.check()
 def test_wrong_job(self):
  self.lines(lambda a:a.__setitem__(2,a[2].replace('2994','2995')))
  with self.assertRaises(AssertionError):self.check()
 def test_wrong_node(self):
  self.lines(lambda a:a.__setitem__(1,a[1].replace('h100-10','h100-11')))
  with self.assertRaises(AssertionError):self.check()
 def test_reversed_time(self):
  self.lines(lambda a:a.__setitem__(2,a[2].replace('17:45:17.975','15:45:17.975')))
  with self.assertRaises(AssertionError):self.check()
 def test_conflicting_terminal(self):
  self.lines(lambda a:a.append('[2026-10-03T17:45:18.000] _job_complete: JobId=2994 WEXITSTATUS 1'))
  with self.assertRaises(AssertionError):self.check()
 def test_duplicate_allocation(self):
  self.lines(lambda a:a.append(a[1]))
  with self.assertRaises(AssertionError):self.check()
 def test_bad_sha(self):
  self.data['controller_origin']['sha256']='0'*64
  with self.assertRaises(AssertionError):self.check()
 def test_old_source(self):
  self.data['source_commit']='a'*40
  with self.assertRaises(AssertionError):self.check()
 def test_source_file_bypass(self):
  self.put('source_commit_evidence','a'*40+'\n')
  with self.assertRaises(AssertionError):self.check()
 def test_fake_completed_start(self):
  self.put('start_snapshot',(ROOT/'job2994-slurm-start.txt').read_text().replace('JobState=RUNNING','JobState=COMPLETED'))
  with self.assertRaises(AssertionError):self.check()
 def test_missing_start(self):
  self.data.pop('start_snapshot')
  with self.assertRaises(AssertionError):self.check()
 def test_wrong_gpu_count(self):
  self.put('start_snapshot',(ROOT/'job2994-slurm-start.txt').read_text().replace('gres:gpu:h100:8','gres:gpu:h100:1'))
  with self.assertRaises(AssertionError):self.check()
 def test_ordinary_running_still_rejected(self):
  with self.assertRaises(AssertionError):verify_slurm((ROOT/'job2994-slurm-start.txt').read_text(),'2994')
 def test_ordinary_completed_still_supported(self):verify_slurm('JobId=2996 JobState=COMPLETED ExitCode=0:0','2996')
 def test_missing_hf_terminal_still_rejected(self):
  for text in ('JobId=2996 ExitCode=0:0','JobId=2996 JobState=RUNNING ExitCode=0:0','JobId=2996 JobState=FAILED ExitCode=1:0'):
   with self.subTest(text=text),self.assertRaises(AssertionError):verify_slurm(text,'2996')
if __name__=='__main__':unittest.main()
