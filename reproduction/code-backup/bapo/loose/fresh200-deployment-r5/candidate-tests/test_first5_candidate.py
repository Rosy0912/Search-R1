"""Local CPU counterexamples. Fixtures never constitute native acceptance."""
import copy,importlib.util,json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import first5_helpers as h
spec=importlib.util.spec_from_file_location('first5_candidate',ROOT/'publish-first5-release.CANDIDATE.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
from search_r1.baselines.fresh200_gate import validate_first_five_attempts
from search_r1.baselines.fresh200_driver import Fresh200Progress
class CandidateTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
  self.paths=[self.root/x for x in ('collection.json','hf.json','manifest.json')]
  for p in self.paths:p.write_text('{}')
  self.review=dict(decision='PASS_INDEPENDENT_ACTUAL_FRESH200_FIRST5_REVIEW',reviewer_task_id=m.REVIEWER_TASK,producer_task_id='/root/fresh200_driver_contract',source_commit=m.SOURCE,run_root=str(m.RUN),job_id=m.JOB,
   bindings=dict(collection_sha256=h.digest(self.paths[0]),hf_sha256=h.digest(self.paths[1]),checkpoint_manifest_sha256=h.digest(self.paths[2])),whole_200_accepted=False,release_scope='CONTINUE_SAME_PROCESS_FROM_ACCEPTED_OUTER_6_TO_200',mustfix=[])
 def tearDown(self):self.tmp.cleanup()
 def test_unbound_cannot_publish(self):
  with self.assertRaisesRegex(ValueError,'UNBOUND'):m.build(*self.paths)
 def test_review_exact_current_binding(self):m.validate_review(self.review,*self.paths)
 def test_old_source_denied(self):
  r=copy.deepcopy(self.review);r['source_commit']='379b14c65e127cd3a71524f7783deaebafbbdb41'
  with self.assertRaises(ValueError):m.validate_review(r,*self.paths)
 def test_self_review_denied(self):
  r=copy.deepcopy(self.review);r['producer_task_id']=m.REVIEWER_TASK
  with self.assertRaises(ValueError):m.validate_review(r,*self.paths)
 def test_whole_200_claim_denied(self):
  r=copy.deepcopy(self.review);r['whole_200_accepted']=True
  with self.assertRaises(ValueError):m.validate_review(r,*self.paths)
 def test_changed_hf_denied(self):
  self.paths[1].write_text('{"tampered":true}')
  with self.assertRaises(ValueError):m.validate_review(self.review,*self.paths)
 def test_wrong_job_denied(self):
  r=copy.deepcopy(self.review);r['job_id']=3305
  with self.assertRaises(ValueError):m.validate_review(r,*self.paths)
 def test_atomic_evidence_no_overwrite(self):
  with self.assertRaises(FileExistsError):h.write_new(self.paths[0],dict(overwrite=True))
 def test_atomic_release_installs_complete_json_and_never_overwrites(self):
  p=self.root/'receipt.json';m.publish_atomic(p,{'complete':True})
  self.assertEqual(json.loads(p.read_text()),{'complete':True})
  with self.assertRaises(FileExistsError):m.publish_atomic(p,{'overwrite':True})
  self.assertEqual(json.loads(p.read_text()),{'complete':True})
 def test_partial_line_denied(self):
  src=self.root/'train.log';src.write_text('incomplete')
  with self.assertRaises(ValueError):h.immutable_prefix(src,self.root/'prefix')
 def test_variable_writes_not_100(self):
  progress=Fresh200Progress();attempts=[]
  for outer,status,writes,processed,abort in [(1,'ABORTED',3,3,True),(1,'COMPLETE_WITH_SKIPS',18,20,False),(2,'SKIPPED',0,20,False),(2,'COMPLETE_WITH_SKIPS',19,20,False),(3,'COMPLETE',20,20,False),(4,'COMPLETE',20,20,False),(5,'COMPLETE_WITH_SKIPS',17,20,False)]:
   aid=progress.begin_attempt(outer);complete=status in('COMPLETE','COMPLETE_WITH_SKIPS')
   ranks=[dict(rank=rank,actual_outer=outer,attempt_id=aid,status=status,optimizer_steps=writes,expected_optimizer_steps=20,optimizer_opportunities_total=20,processed_opportunities=processed,skipped_opportunities=processed-writes,unprocessed_opportunities=20-processed,update_complete=complete,attempt_complete=processed==20 and not abort,postabort=abort)for rank in range(8)]
   progress.record(ranks,outer);attempts.append(dict(attempt_id=aid,candidate_outer=outer,driver_pid=123,status=status,accepted_complete_outer=complete,rank_write_ledger=ranks,progress=progress.state_dict()))
  self.assertEqual(len(validate_first_five_attempts(attempts,progress.state_dict(),123)),5)
  self.assertEqual(progress.actual_adam_writes_per_rank,[97]*8)
  attempts[0]['rank_write_ledger'][0]['attempt_id']=True
  with self.assertRaises(ValueError):validate_first_five_attempts(attempts,progress.state_dict(),123)
if __name__=='__main__':unittest.main()
