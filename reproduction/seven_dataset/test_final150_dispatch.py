import copy,json,unittest
from unittest.mock import patch
from .final150_dispatch import validate_retirement,parse_submissions
class Final150Tests(unittest.TestCase):
 def setUp(self):
  self.old={'pid':13526,'groups':{'budget-5':{'phase':'WAITING_FINAL150'}}};self.r={'retired_pid':13526,'zero_mutation_verified':True,'confirmed_exited':True,'child_pids':[]}
 def test_retired_idle(self):
  with patch('os.kill',side_effect=ProcessLookupError):validate_retirement(self.old,self.r)
 def test_alive_rejected(self):
  with patch('os.kill'):
   with self.assertRaises(ValueError):validate_retirement(self.old,self.r)
 def test_active_phase_rejected(self):
  for phase in ['EXPORTING_ON_LUMIA','SUBMITTING_UNIQUE_THREE','FAILED']:
   o=copy.deepcopy(self.old);o['groups']['budget-5']['phase']=phase
   with self.assertRaises(ValueError):validate_retirement(o,self.r)
 def test_missing_evidence_rejected(self):
  for k,v in [('zero_mutation_verified',False),('confirmed_exited',False),('child_pids',[1]),('retired_pid',1)]:
   r=dict(self.r);r[k]=v
   with self.assertRaises(ValueError):validate_retirement(self.old,r)
 def rows(self):return [dict(job_id='j'+str(i),repeat_index=i,group='budget-5',outer=150) for i in (1,2,3)]
 def text(self,rows):return '\n'.join('SUBMISSION='+json.dumps(x) for x in rows)
 def test_three_exact(self):self.assertEqual(parse_submissions(self.text(self.rows())),self.rows())
 def test_partial_duplicate_rejected(self):
  rr=self.rows()
  for rows in [rr[:2],rr+[rr[0]],[rr[0],rr[0],rr[2]]]:
   with self.assertRaises(ValueError):parse_submissions(self.text(rows))
 def test_wrong_group_rejected(self):
  rr=self.rows();rr[0]['group']='budget-20'
  with self.assertRaises(ValueError):parse_submissions(self.text(rr))
if __name__=='__main__':unittest.main()
