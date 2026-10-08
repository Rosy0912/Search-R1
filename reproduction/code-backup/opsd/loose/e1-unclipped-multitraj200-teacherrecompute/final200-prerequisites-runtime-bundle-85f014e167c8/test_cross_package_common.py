"""Local real physical source identity fixtures; no actual200, Slurm or main."""
import copy
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import types
import unittest
from unittest import mock

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'final200-prerequisites-candidate'
BUNDLE=HERE.parent/'final200-runtime-bundle-f11580b6a7bc'

def load(path,name,expected=None):
 raw=path.read_bytes()
 if expected is not None:assert hashlib.sha256(raw).hexdigest()==expected
 m=types.ModuleType(name);m.__file__=str(path)
 exec(compile(raw,str(path),'exec'),m.__dict__);return m

v=load(HERE/'verify_actual200_prerequisites.py','candidate_v2_prerequisite')
prior=load(OLD/'verify_actual200_prerequisites.py','original_reviewed_v1_prerequisite','98d4f9d2bc5c14a3b515630978db3fe76c14124e3e8a5c4375f8c65b09d36bd3')
c=v.load_common()

def report_source_portion(bundle=BUNDLE):
 # Only actual source records: never fabricate an actual200 report or receipt.
 common=c.read_file(bundle/'common.py',v.PINS['common.py'])[1]
 consumer=c.read_file(bundle/'consume_final200_logs_terminal.py',v.PINS['consume_final200_logs_terminal.py'])[1]
 return dict(common_source=common,consumer_source=consumer,input_records={common['physical_path']:common,consumer['physical_path']:consumer})

def code_identity(code):
 return (code.co_code,code.co_names,code.co_varnames,code.co_argcount,
         tuple(code_identity(x) if isinstance(x,types.CodeType) else x for x in code.co_consts))

class CrossPackageCommon(unittest.TestCase):
 def test_actual_same_bytes_original_identity_replayed(self):
  old=report_source_portion();before=c.read_file(c.__file__)[1]
  with mock.patch.object(v,'LOG_BUNDLE',BUNDLE):original,records=v.recorded_log_common(old,c)
  self.assertEqual(original.__file__,str(BUNDLE/'common.py'))
  self.assertEqual(original.read_file(original.__file__)[1],old['common_source'])
  self.assertEqual(c.read_file(c.__file__)[1],before)
  self.assertNotEqual(before['physical_path'],old['common_source']['physical_path'])
  self.assertEqual(before['sha256'],old['common_source']['sha256'])
  self.assertEqual(len(records),21)
  self.assertEqual(records[str(BUNDLE/'manifest.json')]['sha256'],v.LOG_BUNDLE_SHA)
 def test_real_old_vs_new_source_portion_still_rejected_not_normalized(self):
  old=report_source_portion();new=c.read_file(c.__file__)[1]
  with self.assertRaises(ValueError):v.same(new,old['common_source'],c,'Actual physical sources differ')
  with mock.patch.object(v,'LOG_BUNDLE',BUNDLE):original,_=v.recorded_log_common(old,c)
  v.same(original.read_file(original.__file__)[1],old['common_source'],c,'Preserved original physical source must match')
 def test_wrong_original_identity_missing_record_and_consumer_rejected(self):
  old=report_source_portion()
  cases=[]
  changed=copy.deepcopy(old);changed['common_source']=c.read_file(c.__file__)[1];cases.append(changed)
  changed=copy.deepcopy(old);changed['common_source']['inode']+=1;cases.append(changed)
  changed=copy.deepcopy(old);changed['common_source']['uid']=False;cases.append(changed)
  changed=copy.deepcopy(old);changed['input_records']={};cases.append(changed)
  changed=copy.deepcopy(old);changed['consumer_source']['physical_path']=str(HERE/'common.py');cases.append(changed)
  with mock.patch.object(v,'LOG_BUNDLE',BUNDLE):
   for value in cases:
    with self.subTest(value=value):
     with self.assertRaises(ValueError):v.recorded_log_common(value,c)
 def test_wrong_frozen_manifest_pin_rejected(self):
  with mock.patch.object(v,'LOG_BUNDLE',BUNDLE),mock.patch.object(v,'LOG_BUNDLE_SHA','0'*64):
   with self.assertRaises(ValueError):v.recorded_log_common(report_source_portion(),c)
 def test_unknown_frozen_member_rejected_only_in_owned_fixture(self):
  with tempfile.TemporaryDirectory() as directory:
   clone=Path(directory)/'synthetic-local-bundle-not-native';shutil.copytree(BUNDLE,clone)
   os.chmod(clone,0o700);(clone/'unknown.original').write_bytes(b'owned fixture extra');os.chmod(clone,0o555)
   try:
    with mock.patch.object(v,'LOG_BUNDLE',clone):
     with self.assertRaises(ValueError):v.recorded_log_common(report_source_portion(clone),c)
   finally:os.chmod(clone,0o700)
 def test_all_other_contract_validation_and_output_code_unchanged(self):
  for name in ['header','record_shape','merge','execution','validate_contents','qualify_review','qualify_original_bindings','collect_inputs','validate_wait_originals','load_common','invocation_identity','main']:
   with self.subTest(name=name):self.assertEqual(code_identity(getattr(v,name).__code__),code_identity(getattr(prior,name).__code__))
  self.assertEqual(v.BUDGET,prior.BUDGET);self.assertEqual(v.PINS,prior.PINS)
  self.assertEqual(v.REQUIRED_RAW,prior.REQUIRED_RAW);self.assertEqual(v.PROOF_KIND,prior.PROOF_KIND)
  self.assertEqual((HERE/'common.py').read_bytes(),(OLD/'common.py').read_bytes())

if __name__=='__main__':unittest.main(verbosity=2)
