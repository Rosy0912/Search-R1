import importlib.util,pathlib,unittest
p=pathlib.Path(__file__).parents[2]/'search_r1/baselines/tree_guard_continue.py'
spec=importlib.util.spec_from_file_location('tree_guard_continue_count',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def ledger(world,writes=20):
 mini=128//world
 return [dict(index=i+1,rows=mini,local_row_indices=list(range(i*mini,(i+1)*mini)),
   actual_writes=int(i<writes),status='SUCCESS_ADAM_WRITE' if i<writes else 'SKIPPED_EMPTY') for i in range(20)]
class FourRankLedgerTests(unittest.TestCase):
 def test_four_and_eight_worlds_validate_same_global_opportunities(self):
  for world in (4,8):
   for writes in (1,11,20):
    x=ledger(world,writes);m.validate_ledger(x,writes,False,world_size=world)
    self.assertEqual(sum(e['rows'] for e in x)*world,2560)
 def test_wrong_world_and_nonconsecutive_row_are_rejected(self):
  with self.assertRaises(ValueError):m.validate_ledger(ledger(4),20,world_size=8)
  x=ledger(4);x[9]['local_row_indices'][0]=999
  with self.assertRaises(ValueError):m.validate_ledger(x,20,world_size=4)
 def test_zero_writes_are_partial_only(self):
  m.validate_ledger(ledger(4,0),0,True,world_size=4)
  with self.assertRaises(ValueError):m.validate_ledger(ledger(4,0),0,False,world_size=4)
if __name__=='__main__':unittest.main()
