import copy,unittest
from .contract import validate_recovery,validate_gate,job_name,ROOT,NBASE,SCIENCE_SOURCE,FAILURES,PLAN_SHA,ALLOCATION
from reproduction.trained_tree7b_v2 import contract as original
class ContractTests(unittest.TestCase):
 def pair(self,i=2):
  ref=dict(repeat=i,repeat_count=3,job=original.job_name(i),root=str(original.ROOT/('r'+str(i))),source_commit=SCIENCE_SOURCE,source_files={'science.py':'f'*64},science={'temperature':0,'seed':0,'tokens':500},model='same179',data='same51713')
  p=copy.deepcopy(ref);p.update(job=job_name(i),root=str(ROOT/('r'+str(i))))
  p['recovery']=dict(superseded_uid=FAILURES[i]['uid'],original_plan_sha256=PLAN_SHA[i],failed_receipt_sha256=FAILURES[i]['receipt_sha256'],entry_source_commit='a'*40,entry_source=str(NBASE/'sources'/('a'*40)),entry_source_files={'science.py':'f'*64},storage_admission_sha256='b'*64)
  return p,ref
 def test_only_remaining_two(self):
  for i in (2,3):
   p,r=self.pair(i);self.assertEqual(validate_recovery(p,r),ROOT/('r'+str(i)))
  for i in (0,1,4):
   with self.assertRaises(AssertionError):job_name(i)
 def test_science_mutations(self):
  for k,v in [('model','other'),('data','subset512'),('repeat_count',4),('source_commit','c'*40),('science',{'temperature':1})]:
   p,r=self.pair();p[k]=v
   with self.assertRaises(AssertionError):validate_recovery(p,r)
 def test_wrong_failed_uid(self):
  p,r=self.pair();p['recovery']['superseded_uid']=FAILURES[3]['uid']
  with self.assertRaises(AssertionError):validate_recovery(p,r)
 def test_wrong_receipt(self):
  p,r=self.pair();p['recovery']['failed_receipt_sha256']='f'*64
  with self.assertRaises(AssertionError):validate_recovery(p,r)
 def test_wrong_plan(self):
  p,r=self.pair();p['recovery']['original_plan_sha256']='f'*64
  with self.assertRaises(AssertionError):validate_recovery(p,r)
 def test_path_escape(self):
  p,r=self.pair();p['recovery']['entry_source_files']['../escape']='f'*64
  with self.assertRaises(AssertionError):validate_recovery(p,r)
 def test_old_scientific_source_sha(self):
  p,r=self.pair();p['recovery']['entry_source_files']['science.py']='a'*64
  with self.assertRaises(AssertionError):validate_recovery(p,r)
 def test_extra_provenance_field(self):
  p,r=self.pair();p['recovery']['auto_retry']=True
  with self.assertRaises(AssertionError):validate_recovery(p,r)
 def test_gate(self):
  for i in (2,3):
   p,r=self.pair(i);s=dict(job=p['job'],uid='new'+str(i),plan_sha256='c'*64,superseded_uid=FAILURES[i]['uid'],entry_source_commit='a'*40)
   g=dict(actual=dict(name=s['job'],uid=s['uid'],state='RUNNING',resources=ALLOCATION),plan_sha256=s['plan_sha256'],source=SCIENCE_SOURCE,superseded_uid=s['superseded_uid'],entry_source_commit='a'*40)
   self.assertEqual(validate_gate(p,s,g),g['actual']);g['actual']['uid']='wrong'
   with self.assertRaises(AssertionError):validate_gate(p,s,g)
if __name__=='__main__':unittest.main()
