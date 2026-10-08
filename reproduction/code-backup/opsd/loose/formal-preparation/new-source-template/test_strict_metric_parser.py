"""Actual retained logs and fail-closed mutations; no training/GPU work."""
import importlib.util,json,re,sys,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from strict_metric_parser import parse_metric_summary,WARNING_BODY,WARNING_PATH
spec=importlib.util.spec_from_file_location('new_formal_verifier',ROOT/'formal-preparation/formal-2994-2996-strict-two-raywarnings-preview/verify_formal_prerequisites.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
RAW=json.loads((ROOT/'formal-preparation/strict-raywarning-actual-fixtures/job2999-actual-outer1-5-original.json').read_text())['text']
LINES=[x for x in RAW.splitlines() if re.search(r'\bstep:[1-5] - ',x)]
class Tests(unittest.TestCase):
 def test_real2999_allfive_exact_single_event(self):
  result=module.verify_outers(RAW,5)
  self.assertEqual([x['step'] for x in result['recognized_trailing_ray_warning_frames']],[4])
  self.assertEqual(result['recognized_trailing_ray_warning_frames'][0]['worker_pid'],3715367)
 def test_old_parser_failure_preserved(self):
  spec=importlib.util.spec_from_file_location('old_verifier',ROOT/'formal-preparation/verify_formal_prerequisites.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
  with self.assertRaises(ValueError):old.verify_outers(RAW,5)
 def test_real2994_no_normalization(self):
  raw=json.loads((ROOT/'opsd/reproduction/baselines/opsd-validation-20261003/evidence/job2994_five_raw_reviewer.json').read_text())['files']['training.log']
  self.assertEqual(module.verify_outers(raw,5)['recognized_trailing_ray_warning_frames'],[])
 def reject_line(self,line):
  with self.assertRaises(AssertionError):parse_metric_summary(line)
 def test_bad_numeric(self):self.reject_line(LINES[3].replace('timing_per_token_ms/gen:0.043','timing_per_token_ms/gen:broken'))
 def test_nan_numeric(self):self.reject_line(LINES[3].replace('actor/grad_norm_mean/rank_0:','actor/grad_norm_mean/rank_0:nan'))
 def test_overflow_numeric(self):self.reject_line(LINES[3].replace('timing_per_token_ms/gen:0.043','timing_per_token_ms/gen:1e999'))
 def test_real_nan_numeric(self):self.reject_line(LINES[3].replace('timing_per_token_ms/gen:0.043','timing_per_token_ms/gen:nan'))
 def test_fake_warning(self):self.reject_line(LINES[3].replace('FutureWarning:','OtherWarning:'))
 def test_warning_wrong_path(self):self.reject_line(LINES[3].replace('fully_sharded_data_parallel.py:689','fully_sharded_data_parallel.py:690'))
 def test_truncated_warning(self):self.reject_line(LINES[3][:-10])
 def test_warning_extra_trailing_content(self):self.reject_line(LINES[3]+' - actor/samples/rank_0:320.000')
 def test_warning_missing_worker_pid(self):self.reject_line(LINES[3].replace('pid=3715367','pid=notnumeric'))
 def test_warning_wrong_metric_attachment(self):self.reject_line(LINES[3].replace('timing_per_token_ms/gen:0.043','timing_per_token_ms/adv:0.043'))
 def test_warning_mid_summary(self):
  plain=re.sub(r'\x1b\[[0-9;]*m','',LINES[3]);frame=plain[plain.index('(WorkerDict pid='):]
  self.reject_line(LINES[0].replace(' - actor/lr:',frame+' - actor/lr:'))
 def test_duplicate_metric(self):self.reject_line(LINES[0]+' - actor/samples/rank_0:320.000')
 def test_extra_job_frame_rejected(self):self.reject_line(LINES[3]+' JobId=9999')
 def test_all_numeric_fields_unchanged_after_exact_frame_split(self):
  step,metrics,events=parse_metric_summary(LINES[3])
  clean=re.sub(r'\x1b\[[0-9;]*m','',LINES[3]).split('(WorkerDict pid=',1)[0]
  same_step,same_metrics,no_events=parse_metric_summary(clean)
  self.assertEqual((step,metrics),(same_step,same_metrics));self.assertEqual(len(events),1);self.assertEqual(no_events,[])
 def test_unknown_trailing_frame(self):self.reject_line(LINES[0]+'(WorkerDict pid=3715367) unknown warning')
 def test_malformed_segment_not_skipped(self):self.reject_line(LINES[0]+' - missing_value')
 def test_duplicate_step(self):
  with self.assertRaises(AssertionError):module.verify_outers(RAW+'\n'+LINES[4],5)
 def test_missing_required_rank_metric(self):
  raw=re.sub(r' - actor/samples/rank_7:320\.000','',RAW)
  with self.assertRaises((AssertionError,KeyError)):module.verify_outers(raw,5)
 def test_rank_budget_wrong(self):
  with self.assertRaises(AssertionError):module.verify_outers(RAW.replace('actor/samples/rank_7:320.000','actor/samples/rank_7:319.000'),5)
 def test_teacher_not_frozen(self):
  with self.assertRaises(AssertionError):module.verify_outers(RAW.replace('teacher/trainable_parameters_per_rank/rank_7:0.000','teacher/trainable_parameters_per_rank/rank_7:1.000'),5)
 def test_partial_five_not_fivehundred(self):
  with self.assertRaises(AssertionError):module.verify_outers(RAW,500)
if __name__=='__main__':unittest.main()
