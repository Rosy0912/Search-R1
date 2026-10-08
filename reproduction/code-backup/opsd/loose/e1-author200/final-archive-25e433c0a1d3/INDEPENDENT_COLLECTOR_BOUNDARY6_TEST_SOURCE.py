"""Independent local schema/closure counterexamples. Never runs production main."""
import unittest,types,json,hashlib,copy
from pathlib import Path
B=Path(__file__).resolve().parent
P=B/'final-archive-v2-candidate/collect_actual3261_final_originals.py'
raw=P.read_bytes();assert hashlib.sha256(raw).hexdigest()=='d83ef9dcbc950978a96b280789574917b0c8a81148c52d73ea1109a22f828406'
c=types.ModuleType('independent_archive_boundaries');c.__file__=str(P);exec(compile(raw,str(P),'exec'),c.__dict__)
class Boundaries(unittest.TestCase):
 def setUp(self):
  self.r={'stable_full_bytes':True,'initial_inode_matches_opened':True}
  self.t={'terminal':True,'state':'FAILED','exit_code':'1:0'}
  self.s={'visibility_complete':True,'original_processes':[{'status':'ABSENT_AT_OBSERVATION'}],'training_job_processes':[],'writers':{'x':[]}}
 def test_exit_disagreement_never_closed(self):
  t=dict(self.t,exit_code='0:0');self.assertFalse(c.closure_for(self.r,'x',self.s,self.s,self.t,t)['closed_writer_lifetime'])
 def test_missing_expected_inode_role_never_closed(self):
  s=copy.deepcopy(self.s);s['writers']={};self.assertFalse(c.closure_for(self.r,'x',self.s,s,self.t,self.t)['closed_writer_lifetime'])
 def test_reused_pid_does_not_hide_foreign_writer(self):
  s=copy.deepcopy(self.s);s['original_processes']=[{'status':'PID_REUSED_ORIGINAL_GONE'}];s['writers']['x']=[{'pid':12345,'uid':1000}]
  self.assertFalse(c.closure_for(self.r,'x',self.s,s,self.t,self.t)['closed_writer_lifetime'])
 def test_two_terminal_failure_observations_do_not_imply_training_success(self):
  self.assertTrue(c.closure_for(self.r,'x',self.s,self.s,self.t,self.t)['closed_writer_lifetime'])
  self.assertEqual(self.t['state'],'FAILED');self.assertNotIn('training_accepted',c.closure_for(self.r,'x',self.s,self.s,self.t,self.t))
 def test_empty_original_scan_cannot_close(self):
  s=copy.deepcopy(self.s);s['original_processes']=[];self.assertFalse(c.closure_for(self.r,'x',s,s,self.t,self.t)['closed_writer_lifetime'])
 def test_original_identity_unavailable_blocks_closed_scope(self):
  s=copy.deepcopy(self.s);s['original_processes']=[{'status':'UNAVAILABLE_IDENTITY_NOT_EXIT_PROOF'}];self.assertFalse(c.closure_for(self.r,'x',s,self.s,self.t,self.t)['closed_writer_lifetime'])
if __name__=='__main__':unittest.main(verbosity=2)
