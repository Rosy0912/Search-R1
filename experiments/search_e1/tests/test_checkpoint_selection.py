import unittest,tempfile,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'control'))
from select_grpo_checkpoint import select
class SelectionTest(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.initial=self.root/'initial';self.initial.mkdir();(self.initial/'config.json').write_text('{}');self.stage=self.root/'run'
  for step in [50,100,200]:
   p=self.stage/'actor'/f'global_step_{step}';p.mkdir(parents=True);(p/'config.json').write_text('{}')
  self.rows=[{'data_source':s} for s in ['nq','hotpotqa'] for _ in range(256)]
 def tearDown(self):self.temp.cleanup()
 def check(self,log):return select(log,self.rows,self.initial,self.stage,200)
 def test_collapse_selects_validated_earlier_model(self):
  r=self.check('step:0 - val/test_score/nq:0.391 - val/test_score/hotpotqa:0.391\nstep:50 - val/test_score/nq:0.500 - val/test_score/hotpotqa:0.500\nstep:200 - val/test_score/nq:0.250 - val/test_score/hotpotqa:0.250')
  self.assertEqual(r['selected']['step'],50);self.assertEqual(r['selected']['correct'],256)
 def test_numerical_rounding_equal_counts_prefer_baseline(self):
  r=self.check('step:0 - val/test_score/nq:0.391 - val/test_score/hotpotqa:0.391\nstep:50 - val/test_score/nq:0.395 - val/test_score/hotpotqa:0.387')
  self.assertEqual(r['selected']['step'],0);self.assertEqual(r['selected']['correct'],200)
 def test_final_validation_refers_to_final_saved_model(self):
  r=self.check('step:0 - val/test_score/nq:0.000 - val/test_score/hotpotqa:0.000\nstep:201 - val/test_score/nq:0.500 - val/test_score/hotpotqa:0.500')
  self.assertEqual(r['selected']['step'],200)
 def test_missing_baseline_refuses_selection(self):
  with self.assertRaisesRegex(ValueError,'baseline'):self.check('step:200 - val/test_score/nq:0.500 - val/test_score/hotpotqa:0.500')
if __name__=='__main__':unittest.main()
