import unittest
from search_r1.baselines.tree_fresh200 import validate_initial_worker_receipts as validate
class Initialization(unittest.TestCase):
 def receipts(self,n):return [dict(rank=i,optimizer_state_entries=0,scheduler_step=0) for i in range(n)]
 def test_actual_four_and_eight(self):
  for n in (4,8):validate(self.receipts(n),n)
 def test_missing_or_duplicate_rank(self):
  for r in (self.receipts(3),self.receipts(4)[:-1]+[self.receipts(4)[0]]):
   with self.assertRaises(RuntimeError):validate(r,4)
 def test_nonfresh_optimizer_or_scheduler(self):
  for k in ('optimizer_state_entries','scheduler_step'):
   r=self.receipts(4);r[2][k]=1
   with self.assertRaises(RuntimeError):validate(r,4)
 def test_wrong_world(self):
  with self.assertRaises(RuntimeError):validate(self.receipts(4),8)
  with self.assertRaises(RuntimeError):validate(self.receipts(4),2)
if __name__=='__main__':unittest.main()
