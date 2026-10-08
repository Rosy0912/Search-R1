import json,sys,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'formal-preparation/hf500-formal2999-periodic64-seven-source-preview'))
from periodic_validation import verify_periodic_validations
class Tests(unittest.TestCase):
 def metrics(self,count=64):
  m={'val/qa_count/all':count}
  for source,n in [('nq',count-54)]+[(str(i),9) for i in range(6)]:m['val/qa_count/'+source]=n;m['val/evaluated_count/'+source]=n
  for source in ('all','nq')+tuple(str(i) for i in range(6)):
   for kind in ('f1','em','answer_format','format_rate'):m[f'val/{kind}/{source}']=.123456
  return m
 def pair(self,step,count=64):
  m=self.metrics(count);return ['BASELINE_EVALUATION_METRICS_JSON='+json.dumps(m),'step:'+str(step)+' - '+' - '.join(k+':'+format(v,'.3f') for k,v in m.items())]
 def text(self):return sum((self.pair(step) for step in range(0,501,50)),[])
 def check(self,lines):return verify_periodic_validations('\n'.join(lines),require_final=False)
 def test_all_eleven64_json_rounding(self):self.assertEqual(len(self.check(self.text())['periodic_steps']),11)
 def test_missing50(self):
  lines=self.text();del lines[2:4]
  with self.assertRaises(AssertionError):self.check(lines)
 def test_missing500(self):
  with self.assertRaises(AssertionError):self.check(self.text()[:-2])
 def test3393(self):
  lines=self.text();lines[2:4]=self.pair(50,3393)
  with self.assertRaises(AssertionError):self.check(lines)
 def test_duplicate_json(self):
  lines=self.text();lines.insert(2,lines[2])
  with self.assertRaises(AssertionError):self.check(lines)
 def test_duplicate_summary(self):
  lines=self.text();lines.insert(2,lines[1])
  with self.assertRaises(AssertionError):self.check(lines)
 def test_unexpected_step(self):
  lines=self.text();lines[3]=lines[3].replace('step:50','step:5')
  with self.assertRaises(AssertionError):self.check(lines)
 def test_json_summary_count_mismatch(self):
  lines=self.text();lines[3]=lines[3].replace('val/evaluated_count/nq:10.000','val/evaluated_count/nq:9.000')
  with self.assertRaises(AssertionError):self.check(lines)
 def test_json_summary_score_mismatch(self):
  lines=self.text();lines[3]=lines[3].replace('val/f1/all:0.123','val/f1/all:0.124')
  with self.assertRaises(AssertionError):self.check(lines)
 def test_nonfinite(self):
  lines=self.text();lines[3]=lines[3].replace('val/f1/all:0.123','val/f1/all:nan')
  with self.assertRaises(AssertionError):self.check(lines)
 def test_actual2994_raw_format(self):
  actual=json.loads((ROOT/'opsd/reproduction/baselines/opsd-validation-20261003/evidence/job2994_five_raw_reviewer.json').read_text())['files']['training.log']
  result=verify_periodic_validations(actual,expected_steps=(0,5),require_final=False)
  self.assertEqual(result['periodic_steps'],[0,5])
 def test_actual2999_initial_format(self):
  actual=json.loads((ROOT/'opsd/reproduction/baselines/opsd-validation-20261003/evidence/job2999_partial_outer_raw_reviewer.json').read_text())['text']
  result=verify_periodic_validations(actual,expected_steps=(0,),require_final=False)
  self.assertEqual(result['periodic_steps'],[0])
if __name__=='__main__':unittest.main()
