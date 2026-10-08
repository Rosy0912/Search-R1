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
  self.oldrun,self.oldjob=m.RUN,m.JOB;m.RUN=h.ROOT/"runs/formal200-newfixture-9999";m.JOB=9999
  self.paths=[self.root/x for x in ('collection.json','hf.json','manifest.json')]
  for p in self.paths:p.write_text('{}')
  self.review=dict(decision='PASS_INDEPENDENT_ACTUAL_FRESH200_FIRST5_REVIEW',reviewer_task_id=m.REVIEWER_TASK,producer_task_id='/root/fresh200_actor_safety',source_commit=m.SOURCE,run_root=str(m.RUN),job_id=m.JOB,
   bindings=dict(collection_sha256=h.digest(self.paths[0]),hf_sha256=h.digest(self.paths[1]),checkpoint_manifest_sha256=h.digest(self.paths[2])),whole_200_accepted=False,release_scope='CONTINUE_SAME_PROCESS_FROM_ACCEPTED_OUTER_6_TO_200',mustfix=[])
 def tearDown(self):
  m.RUN,m.JOB=self.oldrun,self.oldjob;self.tmp.cleanup()
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
 def test_missing_actual_run_fail_closed(self):
  from unittest.mock import patch
  h.require_run_binding()
  with patch.object(h,'RUN',None),patch.object(h,'JOB',None):
   with self.assertRaisesRegex(ValueError,'UNBOUND'):h.require_run_binding()
 def test_v2_drift_not_identity_gate(self):
  p=dict(probability_contract='ACTUAL_NATIVE_BEHAVIOR_V2_HF_DIAGNOSTIC_SEPARATE',hf_native_drift_is_identity_gate=False,behavior_policy_role='actual_native_stochastic_sampling_policy',max_selected_logprob_abs_diff=2.3)
  self.assertIs(h.behavior_proof(p),p)
  for key,value in [('hf_native_drift_is_identity_gate',True),('probability_contract','V1'),('behavior_policy_role','hf_old_policy')]:
   bad=dict(p);bad[key]=value
   with self.assertRaises(ValueError):h.behavior_proof(bad)
 def test_frozen_source_and_package_binding(self):
  self.assertEqual(h.SOURCE,'90755a561abd239fbb23882939410e643bb913ae')
  self.assertEqual(h.RUN,h.ROOT/'runs/formal200-8c29851b54174f2083023a98cde9c19b-3312');self.assertEqual(h.JOB,3312)
  for name in ('first5_helpers.py','first5-extract.py','first5-hf.py','publish-first5-release.CANDIDATE.py'):
   text=(ROOT/name).read_text()
   self.assertNotIn('6a29642035ff40b2b7787de206f886c0',text)
   self.assertNotIn('aec5e057093b4c491bfb0a47abd7605d98981b56',text)
 def test_actual_behavior_alias_and_hf_separation(self):
  import torch
  from types import SimpleNamespace
  from unittest.mock import patch
  from search_r1.baselines.behavior_policy import bind_actual_behavior,validate_behavior_tensors
  old=torch.tensor([[-3.7,-2.0]]);sampled=torch.tensor([[-1.5,-1.9]])
  d=SimpleNamespace(batch=dict(old_log_probs=old,sampling_log_probs=sampled,sampling_mask=torch.ones_like(old,dtype=torch.bool)),meta_info={})
  with patch('search_r1.llm_agent.sampling_tokens.audit_batch',return_value={'probability_role':'stochastic_sampling_policy'}):
   bind_actual_behavior(d)
   self.assertTrue(torch.equal(d.batch['frozen_hf_log_probs'],old))
   self.assertTrue(torch.equal(d.batch['old_log_probs'],sampled))
   self.assertTrue(validate_behavior_tensors(d))
   with self.assertRaises(ValueError):bind_actual_behavior(d)
   d.batch['old_log_probs'][0,0]-=.01
   with self.assertRaises(ValueError):validate_behavior_tensors(d)
 def test_review_typed_job_and_foreign_run_denied(self):
  for key,value in [('job_id',True),('job_id',9999.0),('run_root','/old/foreign/run')]:
   bad=copy.deepcopy(self.review);bad[key]=value
   with self.assertRaises(ValueError):m.validate_review(bad,*self.paths)
if __name__=='__main__':unittest.main()
