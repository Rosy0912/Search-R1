import hashlib,importlib.util,json,sys,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
HF=ROOT/'formal-preparation/hf500-new-formal-job-UNFILLED-strict-two-raywarnings-preview'
sys.path[:0]=[str(HF),str(HERE)]
def module(name,file):
 spec=importlib.util.spec_from_file_location(name,HF/file);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
class Tests(unittest.TestCase):
 def test_missing_future_job_blocks_before_receipt(self):
  checker=module('new_future_checker','formal_completion_check.py')
  with self.assertRaisesRegex(AssertionError,'not filled'):checker.verify(Path('/not-a-real-future-receipt'))
 def test_current_unfilled_guard_and_no_oldtraining_job(self):
  text=(HF/'reload_job.sh').read_text();assert 'exit 78' in text and 'UNFILLED_ACTUAL_NEW_FORMAL_JOB_ID' in text
  assert 'opsd-formal-2999' not in text and 'job_2999' not in text
  assert json.loads((HF/'source-manifest.json').read_text())['formal_job'] is None
 def test_actual_partial2999_initial_periodic_and_warning_line(self):
  text=json.loads((ROOT/'formal-preparation/strict-raywarning-actual-fixtures/job2999-actual-outer1-5-original.json').read_text())['text']
  result=module('new_periodic_validator','periodic_validation.py').verify_periodic_validations(text,expected_steps=(0,),require_final=False)
  assert result['periodic_steps']==[0]
 def test_actual_partial_not_complete500_periodics(self):
  text=json.loads((ROOT/'formal-preparation/strict-raywarning-actual-fixtures/job2999-actual-outer1-5-original.json').read_text())['text']
  with self.assertRaises(AssertionError):module('new_periodic_validator_incomplete','periodic_validation.py').verify_periodic_validations(text)
 def test_manifest_all_sources_bound(self):
  m=json.loads((HF/'source-manifest.json').read_text());assert len(m['file_sources'])==8
  for n,v in m['file_sources'].items():assert hashlib.sha256((HF/n).read_bytes()).hexdigest()==v
 def test_original_hf_checker_bytes_unchanged(self):
  old=ROOT/'deployment-bundles/hf-runtime-8e60976d364a-pilot-2994-reviewed-47cab4b02661'
  for n in ('checkpoint_reload.py','resource_sample.py'):assert (HF/n).read_bytes()==(old/n).read_bytes()
if __name__=='__main__':unittest.main()
