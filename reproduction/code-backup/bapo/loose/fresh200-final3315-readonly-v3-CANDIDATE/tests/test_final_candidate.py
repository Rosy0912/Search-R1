import copy,importlib.util,json,sys,unittest
from pathlib import Path
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B))
s=importlib.util.spec_from_file_location('final_candidate',B/'final-evidence.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
from search_r1.baselines.fresh200_driver import Fresh200Progress
class Final(unittest.TestCase):
 def attempts(self,n=200):
  p=Fresh200Progress();out=[]
  def add(outer,status,writes,processed,abort):
   aid=p.begin_attempt(outer);total=20 if outer<50 else 10;complete=status.startswith('COMPLETE')
   ranks=[dict(rank=r,actual_outer=outer,attempt_id=aid,status=status,optimizer_steps=writes,expected_optimizer_steps=total,optimizer_opportunities_total=total,processed_opportunities=processed,skipped_opportunities=processed-writes,unprocessed_opportunities=total-processed,update_complete=complete,attempt_complete=processed==total and not abort,postabort=abort)for r in range(8)]
   p.record(ranks,outer);out.append(dict(source_commit=m.SOURCE,driver_pid=123,attempt_id=aid,candidate_outer=outer,status=status,accepted_complete_outer=complete,rank_write_ledger=ranks,progress=p.state_dict()))
  add(1,'ABORTED',3,3,True);add(1,'COMPLETE_WITH_SKIPS',18,20,False)
  for outer in range(2,n+1):
   total=20 if outer<50 else 10;add(outer,'COMPLETE',total,total,False)
  return out,p.state_dict()
 def test_200_accepts_not_fixed2490_actualwrites(self):
  a,p=self.attempts();r=m.replay_attempts(a,p,123)
  self.assertEqual(r['actual_writes_per_rank'],[2491]*8);self.assertEqual(len(a),201)
  self.assertEqual(r['opportunities_per_accepted_outer'][48:50],[20,10])
 def test_199_not_terminal_complete(self):
  a,p=self.attempts(199)
  with self.assertRaises(ValueError):m.replay_attempts(a,p,123)
 def test_wrong_rank_schedule_pid_source_rejected(self):
  a,p=self.attempts()
  for key,value in [('source_commit','old'),('driver_pid',124),('attempt_id',True)]:
   b=copy.deepcopy(a);b[0][key]=value
   with self.assertRaises(ValueError):m.replay_attempts(b,p,123)
  b=copy.deepcopy(a);b[50]['rank_write_ledger'][0]['expected_optimizer_steps']=20
  with self.assertRaises(ValueError):m.replay_attempts(b,p,123)
 def test_infrastructure_fault_never_normalaccepted(self):
  a,p=self.attempts();a[0]['rank_write_ledger'][0]['status']='INFRASTRUCTURE_FAULT'
  with self.assertRaises(ValueError):m.replay_attempts(a,p,123)
 def test_full_final_weighted_sources_stage_debugexcluded(self):
  x={'val/evaluated_count/nq':10000,'val/evaluated_count/rest':41713,'val/test_score/nq':.2,'val/test_score/rest':.5}
  r=m.final_metrics(x,51713);self.assertAlmostEqual(r['weighted_score'],(.2*10000+.5*41713)/51713)
  for size in [512,64,51712]:
   with self.assertRaises(ValueError):m.final_metrics(x,size)
 def test_final_nonfinite_duplicate_or_boolcounts_rejected(self):
  for raw in ['{"x":NaN}','{"x":1,"x":2}']:
   with self.assertRaises(ValueError):m.strict_json(raw)
  for count,score in [(True,.2),(51713,float('nan'))]:
   with self.assertRaises(ValueError):m.final_metrics({'val/evaluated_count/nq':count,'val/test_score/nq':score},51713)
 def test_running_proof_cannot_readheavy(self):
  p=dict(source_commit=m.SOURCE,run_root=str(m.RUN),job_id=m.JOB,scheduler_state='RUNNING',exit_code=0,native_acceptance=False)
  with self.assertRaisesRegex(ValueError,'Terminal COMPLETED'):m.terminal_gate(p)
if __name__=='__main__':unittest.main()
