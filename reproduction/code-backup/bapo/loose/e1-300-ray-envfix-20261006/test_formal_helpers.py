import copy,hashlib,importlib.util,json,unittest
from pathlib import Path
from unittest.mock import patch
B=Path(__file__).resolve().parent;s=importlib.util.spec_from_file_location('formal_capture',B/'capture-formal-allocation.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class FormalCaptureTest(unittest.TestCase):
 def test_strict_actual_formal_job_and_suffix(self):
  raw='JobId=9002 JobName=bapo-instruct300-formal300 JobState=RUNNING Restarts=0 NodeList=h100-5\r\n';m.validate_slurm(raw,9002)
  for x in (raw.replace('formal300','hardware'),raw.replace('h100-5','h100-5,h100-6'),raw.replace('RUNNING','RUNNINGx')):
   with self.assertRaises(ValueError):m.validate_slurm(x,9002)
 def setUp(self):
  self.pin=patch.object(m,'HARDWARE_RECEIPT_SHA',hashlib.sha256((B/'hardware3287-final-receipt.json').read_bytes()).hexdigest());self.pin.start();self.addCleanup(self.pin.stop)
 def records(self):
  receipt=(B/'hardware3287-final-receipt.json').read_bytes();r=json.loads(receipt)
  report=(B/'HARDWARE3287-FINAL-INDEPENDENT-REVIEW.md').read_bytes()
  return r,{'path':str(m.ROOT/'logs/hardware-acceptance.json'),'sha256':hashlib.sha256(receipt).hexdigest(),'bytes':len(receipt),'text':receipt.decode()}, {'path':r['independent_review']['path'],'sha256':hashlib.sha256(report).hexdigest(),'bytes':len(report),'text':report.decode()}
 def test_actual373_binding_receipt_and_report_raw_identity(self):
  _,r,p=self.records()
  with patch.object(m,'read_raw',side_effect=[r,p]):result=m.hardware_binding()
  self.assertEqual(result['hardware-acceptance.json']['sha256'],m.HARDWARE_RECEIPT_SHA)
  self.assertEqual(result['hardware-independent-review'],p)
 def test_report_mutation_rejected_without_weakening_receipt_pin(self):
  _,r,p=self.records();p['sha256']='0'*64
  with patch.object(m,'read_raw',side_effect=[r,p]):
   with self.assertRaisesRegex(ValueError,'report hash'):m.hardware_binding()
 def test_receipt_changed_rejected_before_report_read(self):
  _,r,_=self.records();r['sha256']='0'*64
  with patch.object(m,'read_raw',return_value=r)as read:
   with self.assertRaisesRegex(ValueError,'receipt differs'):m.hardware_binding()
   self.assertEqual(read.call_count,1)
 def test_report_escape_rejected_even_when_candidate_sha_rebound(self):
  a,r,_=self.records();a['independent_review']['path']=str(m.ROOT/'repo/private.md');raw=json.dumps(a).encode();r.update(text=raw.decode(),sha256=hashlib.sha256(raw).hexdigest())
  with patch.object(m,'HARDWARE_RECEIPT_SHA',r['sha256']),patch.object(m,'read_raw',return_value=r)as read:
   with self.assertRaisesRegex(ValueError,'direct newROOT/logs'):m.hardware_binding()
   self.assertEqual(read.call_count,1)
 def test_explicit_final_pin_is_required_not_old_hold_receipt(self):
  _,r,_=self.records()
  with patch.object(m,'HARDWARE_RECEIPT_SHA',None),patch.object(m,'read_raw',return_value=r):
   with self.assertRaisesRegex(ValueError,'receipt differs'):m.hardware_binding()
 def test_positive_job_and_duplicate_slurm_fields(self):
  for bad in ('0','-1','1suffix','01','',True):
   with self.subTest(job=bad),self.assertRaises(ValueError):m.job_id(bad)
  raw='JobId=9002 JobName=bapo-instruct300-formal300 JobState=RUNNING Restarts=0 NodeList=h100-5 JobId=9002'
  with self.assertRaises(ValueError):m.validate_slurm(raw,9002)
 def test_unsigned_candidate_cannot_pass_publisher_status_guard(self):
  import ast
  tree=ast.parse((B/'publish-formal-permission.py').read_text());guard=next(n for n in tree.body if isinstance(n,ast.Assert)and "permit['status']"in ast.unparse(n.test))
  run=Path('/synthetic/run');job=9002;permit=dict(status='UNSIGNED_CANDIDATE_REQUIRES_ACTUAL_INDEPENDENT_ALLOCATION_REVIEW',target_job_id=job,target_run_root=str(run),target_source_commit=m.SOURCE)
  with self.assertRaises(AssertionError):exec(compile(ast.Module(body=[guard],type_ignores=[]),'publisher_guard','exec'),dict(permit=permit,job=job,run=run,source=m.SOURCE))
if __name__=='__main__':unittest.main()
