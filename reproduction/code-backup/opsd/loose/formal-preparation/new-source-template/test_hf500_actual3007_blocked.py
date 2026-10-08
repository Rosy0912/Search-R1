"""Actual job ID binding, future completion remains absent; synthetic negatives only."""
import hashlib,importlib.util,json,sys,tempfile,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
HF=ROOT/'formal-preparation/hf500-formal3007-strict-two-raywarnings-preview'
sys.path.insert(0,str(HF))
spec=importlib.util.spec_from_file_location('hf500_actual3007',HF/'formal_completion_check.py');checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
class Tests(unittest.TestCase):
 def setUp(self):
  self.t=tempfile.TemporaryDirectory();self.base=Path(self.t.name);self.run=self.base/'runs/opsd-formal-3007';self.run.mkdir(parents=True);self.receipt=self.base/'receipt.json';self.evidence={}
  for role,text in {'raw_log':'','source_commit':checker.COMMIT,'slurm_terminal':'JobId=3007 JobState=RUNNING ExitCode=0:0 NodeList=h100-10 NumNodes=1 NumCPUs=192 TresPerNode=gres:gpu:h100:8','exit':'0','finished':'FORMAL_500_OUTERS_AND_FINAL_EVALUATION_EXITED','completeness':'{}','expected_final_counts':'{}','independent_review':'{}'}.items():self.put(role,text)
  self.seal()
 def put(self,key,text):
  p=self.run/key;p.write_text(text);self.evidence[key]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
 def seal(self,job='3007'):self.receipt.write_text(json.dumps({'source_commit':checker.COMMIT,'formal_job':job,'evidence':self.evidence}))
 def tearDown(self):self.t.cleanup()
 def test_missing_future_receipt(self):
  with self.assertRaises(AssertionError):checker.verify(self.base/'absent.json',self.base)
 def test_running3007_not_complete(self):
  with self.assertRaises(AssertionError):checker.verify(self.receipt,self.base)
 def test_previous2999_receipt_not_reused(self):
  self.seal('2999')
  with self.assertRaises(AssertionError):checker.verify(self.receipt,self.base)
 def test_previous2999_terminal_not_reused(self):
  self.put('slurm_terminal','JobId=2999 JobState=COMPLETED ExitCode=0:0');self.seal()
  with self.assertRaises(AssertionError):checker.verify(self.receipt,self.base)
 def test_completed_flag_without_actual500_rejected(self):
  self.put('slurm_terminal','JobId=3007 JobState=COMPLETED ExitCode=0:0 NodeList=h100-10 NumNodes=1 NumCPUs=192 TresPerNode=gres:gpu:h100:8');self.seal()
  with self.assertRaises(AssertionError):checker.verify(self.receipt,self.base)
 def test_actualjob_and_no_future_reload_id(self):
  m=json.loads((HF/'source-manifest.json').read_text());assert m['formal_job']=='3007' and m['future_reload_job_id'] is None and not m['formal_completion_evidence_present']
  assert m['formal_manifest_sha256']=='eb13b32b6dbcd608c237805bc950023a627ea4bdd4e98596a1afd5c9417d7051'
  shell=(HF/'reload_job.sh').read_text();assert 'exit 78' in shell and 'opsd-formal-3007/output/checkpoints/opsd-gpu-formal500/actor/global_step_500' in shell and "previous_training_job='3007'" in shell
  assert 'job_2999' not in shell and 'UNFILLED_ACTUAL_NEW_FORMAL_JOB_ID' not in shell
 def test_all_sources_fullsha(self):
  m=json.loads((HF/'source-manifest.json').read_text());assert len(m['file_sources'])==8
  for name,sha in m['file_sources'].items():assert hashlib.sha256((HF/name).read_bytes()).hexdigest()==sha
 def test_hf_numeric_and_periodic_checkers_preserved(self):
  old=ROOT/'formal-preparation/hf500-new-formal-job-UNFILLED-strict-two-raywarnings-preview'
  for name in ('checkpoint_reload.py','resource_sample.py','strict_metric_parser.py','periodic_validation.py'):assert (old/name).read_bytes()==(HF/name).read_bytes()
if __name__=='__main__':unittest.main()
