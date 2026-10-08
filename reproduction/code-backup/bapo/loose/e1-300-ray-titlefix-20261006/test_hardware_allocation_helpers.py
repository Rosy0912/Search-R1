"""Actual-allocation helper negative contracts, no remote/job/GPU calls."""
import importlib.util,pathlib,tempfile,unittest,json
B=pathlib.Path(__file__).parent
spec=importlib.util.spec_from_file_location('collector',B/'capture-hardware-allocation.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
spec=importlib.util.spec_from_file_location('checker',B/'check-hardware-allocation.py');k=importlib.util.module_from_spec(spec);spec.loader.exec_module(k)
class Test(unittest.TestCase):
 def test_jobid_exact_positive(self):
  self.assertEqual(c.job_id('3279'),3279)
  for bad in ('0','-1','01',' 3279','True','3279;anything'):
   with self.subTest(bad=bad),self.assertRaises(ValueError):c.job_id(bad)
 def test_slurm_exact_current_and_duplicate_rejected(self):
  raw='JobId=3279 JobName=bapo-instruct300-hardware JobState=RUNNING Restarts=0 NodeList=h100-5';c.validate_slurm(raw,3279)
  for bad in (raw+' JobId=3279',raw.replace('RUNNING','PENDING'),raw.replace('Restarts=0','Restarts=1'),raw.replace('h100-5','h100-4')):
   with self.assertRaises(ValueError):c.validate_slurm(bad,3279)
 def test_raw_symlink_and_bound_size(self):
  with tempfile.TemporaryDirectory()as d:
   p=pathlib.Path(d)/'a';p.write_text('evidence');row=c.read_raw(p);self.assertEqual(row['bytes'],8);alias=pathlib.Path(d)/'link';alias.symlink_to(p)
   with self.assertRaises(ValueError):c.read_raw(alias)
   old=c.MAX_RAW_BYTES;c.MAX_RAW_BYTES=1
   try:
    with self.assertRaises(ValueError):c.read_raw(p)
   finally:c.MAX_RAW_BYTES=old
 def test_strict_json_no_duplicates_or_nonfinite(self):
  for text in ('{"a":1,"a":1}','{"a":NaN}'):
   with self.assertRaises(ValueError):k.json_strict(text)
 def test_legacy_job_or_root_cannot_be_new_authorization(self):
  with tempfile.TemporaryDirectory()as d:
   p=pathlib.Path(d)/'capture.json';p.write_text(json.dumps(dict(status='READ_ONLY_ACTUAL_ALLOCATED_INPUTS_NOT_PERMISSION_OR_GPU_ACCEPTANCE',job_id=3263,source_commit='b7119791b86951727cea36a1881e8a525c932494',root='/old',run_root='/old/runs/3263',actual_gpu=False,permission_signed=False)))
   with self.assertRaises(ValueError):k.validate(p,3279,'unused','unused')
 def test_collect_does_not_publish_permission_or_mutate_files(self):
  import ast
  tree=ast.parse((B/'capture-hardware-allocation.py').read_text());mutating={'write_text','write_bytes','mkdir','unlink','kill','pidfd_send_signal','rename','replace','chmod'}
  self.assertFalse(any(isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr in mutating for n in ast.walk(tree)))
if __name__=='__main__':unittest.main(verbosity=2)
