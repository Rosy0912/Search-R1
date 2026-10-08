import hashlib,importlib.util,json,re,sys,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from strict_metric_parser import parse_metric_summary,CPU_AMP_PATH,CPU_AMP_BODY
PREVIEW=ROOT/'formal-preparation/formal-2994-2996-strict-two-raywarnings-preview'
spec=importlib.util.spec_from_file_location('latest_two_frame_verifier',PREVIEW/'verify_formal_prerequisites.py');verifier=importlib.util.module_from_spec(spec);spec.loader.exec_module(verifier)
RAWPATH=ROOT/'formal-preparation/strict-raywarning-actual-fixtures/job2999-complete-seven-actual-original.json'
RAW=json.loads(RAWPATH.read_text())['text']
LINES=[l for l in RAW.splitlines() if re.search(r'\bstep:\d+ - ',l) and 'actor/optimizer_steps:' in l]
CPU_LINE=next(l for l in LINES if 'checkpoint.py:1399: FutureWarning:' in l)
class Tests(unittest.TestCase):
 def reject(self,line):
  with self.assertRaises(AssertionError):parse_metric_summary(line)
 def test_actual_allseven_exact_two_events(self):
  self.assertEqual(len(LINES),7)
  result=verifier.verify_outers(RAW,7)
  self.assertEqual([(e['step'],e['worker_pid'],e['kind']) for e in result['recognized_trailing_ray_warning_frames']],[(4,3715367,'observed_complete_trailing_ray_fsdp_futurewarning'),(5,3715050,'observed_complete_trailing_ray_cpuamp_futurewarning')])
 def test_first_frame_only_original_failure_preserved(self):
  oldpath=ROOT/'formal-preparation/formal-2994-2996-strict-raywarning-preview/strict_metric_parser.py';spec=importlib.util.spec_from_file_location('old_one_frame_parser',oldpath);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
  with self.assertRaises(AssertionError):old.parse_metric_summary(CPU_LINE)
 def test_full_warning_only_does_not_change_any_numeric_field(self):
  clean=re.sub(r'\x1b\[[0-9;]*m','',CPU_LINE).split('(WorkerDict pid=',1)[0]
  step,metrics,events=parse_metric_summary(CPU_LINE);same_step,same_metrics,no_events=parse_metric_summary(clean)
  self.assertEqual((step,metrics),(same_step,same_metrics));self.assertEqual(no_events,[]);self.assertEqual(len(events),1)
 def test_bad_numeric(self):self.reject(CPU_LINE.replace('timing_per_token_ms/gen:0.046','timing_per_token_ms/gen:broken'))
 def test_nonfinite_numeric(self):
  for value in ('NaN','Inf','1e999'):
   with self.subTest(value=value):self.reject(CPU_LINE.replace('timing_per_token_ms/gen:0.046','timing_per_token_ms/gen:'+value))
 def test_fake_body(self):self.reject(CPU_LINE.replace('deprecated','removed'))
 def test_fake_source_path(self):self.reject(CPU_LINE.replace('checkpoint.py:1399','checkpoint.py:1400'))
 def test_truncated_frame(self):self.reject(CPU_LINE[:-1])
 def test_invalid_worker_pid(self):self.reject(CPU_LINE.replace('pid=3715050','pid=unknown'))
 def test_warning_wrong_metric_position(self):self.reject(CPU_LINE.replace('timing_per_token_ms/gen:0.046','timing_per_token_ms/adv:0.046'))
 def test_fake_extra_job_or_dedup(self):
  for suffix in (' JobId=7777',' [repeated 8x across cluster]',' - actor/samples/rank_0:320.000'):
   with self.subTest(suffix=suffix):self.reject(CPU_LINE+suffix)
 def test_missing_rank_metric_still_rejected(self):
  raw=re.sub(r' - actor/samples/rank_7:320\.000','',RAW)
  with self.assertRaises((AssertionError,KeyError)):verifier.verify_outers(raw,7)
 def test_allseven_not_fivehundred(self):
  with self.assertRaises(AssertionError):verifier.verify_outers(RAW,500)
if __name__=='__main__':unittest.main()
