"""Local CPU schema/publication counterexamples, never remote/GPU admission."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

PUBLISHER=Path(__file__).resolve().parents[1]/'publish-launch-approval.py'
spec=importlib.util.spec_from_file_location('fresh_launch_publisher',PUBLISHER)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

class LaunchPublisher(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name).resolve()
  self.root_patch=patch.object(m,'ROOT',self.root);self.root_patch.start()
  (self.root/'logs').mkdir();(self.root/'tests').mkdir()
 def tearDown(self):self.root_patch.stop();self.tmp.cleanup()
 def cpu(self,text='165 passed, 1 warning in 19.77s\n'):
  log=self.root/'logs/target-cpu-gate.raw.log';log.write_text(text)
  context=dict(status='ACTUAL_TARGET_CPU_ALL_SHARED_CONTEXT_BUDGET_OK_NOT_GPU_ACCEPTANCE',source_commit=m.SOURCE,source_sha256={'fixture':'cpu'},
   records={k:dict(rows=n,maximum_initial_tokens=1096,worst_before_fourth_request_tokens=4096) for k,n in [('train',169028),('validation',512),('stage',64),('final',51713)]},
   force_final_answer=False,max_turns=4,max_start_length=2048,max_rolling_context=4096,max_response_tokens=500,max_observation_tokens=500)
  path=self.root/'logs/context-budget-audit.json';path.write_text(json.dumps(context))
  record=dict(source_commit=m.SOURCE,exit_code=0,platform='linux',host='h100-5',scope='ACTUAL_CPU_IMPORT_TEST_CONTEXT_GATE_NOT_NATIVE_GPU',
   tools_manifest_sha256=m.TOOLS_SHA,critical_source_sha256={'fixture':'cpu'},log_path=str(log),log_sha256=digest(log),context_audit_sha256=digest(path))
  return record,context
 def validate_cpu(self,record,context):
  with patch.object(m,'CPU_RAW_SHA',digest(self.root/'logs/target-cpu-gate.raw.log')),patch.object(m,'CONTEXT_SHA',digest(self.root/'logs/context-budget-audit.json')):
   m.validate_cpu(record,context,{'fixture':'cpu'})
 def test_actual_warning_singular_plural_and_zero(self):
  for text in ['165 passed, 1 warning in 19.77s\n','165 passed, 2 warnings in 19.77s\n','165 passed in 19.77s\n']:
   record,context=self.cpu(text);self.validate_cpu(record,context)
 def test_cpu_false_exit_wrong_source_tool_or_context_rejected(self):
  record,context=self.cpu()
  for key,value in [('exit_code',False),('source_commit','8e'+'0'*38),('tools_manifest_sha256','0'*64),('platform','darwin'),('context_audit_sha256','0'*64)]:
   bad=copy.deepcopy(record);bad[key]=value
   with self.assertRaises(ValueError):self.validate_cpu(bad,context)
 def test_cpu_skip_or_incomplete_summary_rejected(self):
  for text in ['164 passed, 1 warning in 19.77s\n','165 passed, 1 skipped in 19.77s\n','165 passed, 1 warning in 19.77s\n1 failed\n']:
   record,context=self.cpu(text)
   with self.assertRaises(ValueError):self.validate_cpu(record,context)
 def test_context_counts_and_budget_rejected(self):
  record,context=self.cpu()
  for key,value in [('rows',169027),('worst_before_fourth_request_tokens',4097)]:
   bad=copy.deepcopy(context);bad['records']['train'][key]=value
   with self.assertRaises(ValueError):self.validate_cpu(record,bad)
 def resource(self,text='Ran 80 tests in 4.230s\n\nOK\n'):
  log=self.root/'logs/runtime-tests.actual.log';log.write_text(text);files={}
  for name in m.RESOURCE_TEST_HASHES:
   path=self.root/'tests'/name;path.write_text('CPU fixture '+name);files[name]=digest(path)
  record=dict(source_commit=m.SOURCE,host='h100-5',exit_code=0,scope='ACTUAL_LINUX_OWN_CHILD_RESOURCE_TESTS_NOT_GPU',files=files,raw_log_sha256=digest(log))
  return record,files
 def validate_resource(self,record,files):
  with patch.object(m,'RESOURCE_TEST_HASHES',files),patch.object(m,'RESOURCE_RAW_SHA',digest(self.root/'logs/runtime-tests.actual.log')):m.validate_resource(record)
 def test_resource_exact80_no_skips(self):
  record,files=self.resource();self.validate_resource(record,files)
 def test_resource_skipped_failed_or_incomplete(self):
  for text in ['Ran 80 tests in 4.230s\nOK (skipped=3)\n','Ran 79 tests in 4.230s\nOK\n','Ran 80 tests in 4.230s\nFAILED (errors=1)\n']:
   record,files=self.resource(text)
   with self.assertRaises(ValueError):self.validate_resource(record,files)
 def test_resource_raw_and_file_tamper(self):
  record,files=self.resource();(self.root/'tests'/next(iter(files))).write_text('changed')
  with self.assertRaises(ValueError):self.validate_resource(record,files)
 def test_strict_json_duplicate_and_nonfinite(self):
  for text in ['{"exit_code":0,"exit_code":1}','{"exit_code":NaN}']:
   with self.assertRaises(ValueError):m.parse_json(text)
 def test_permission_atomic_immutable_and_evidence_drift(self):
  evidence=self.root/'logs/evidence.json';evidence.write_text('{}')
  value=dict(protocol='CPU_FIXTURE_NOT_PERMISSION',scope='CPU_FIXTURE_NOT_PERMISSION',evidence=[m.binding(evidence)])
  with patch.object(m,'build_approval',return_value=value),patch('builtins.print'):
   m.publish('f'*64)
   with self.assertRaisesRegex(ValueError,'Existing approval'):m.publish('f'*64)
   evidence.write_text('drift')
   with self.assertRaisesRegex(ValueError,'Artifact bytes changed'):m.publish('f'*64)
 def test_symlink_evidence_rejected(self):
  path=self.root/'logs/real';path.write_text('{}');link=self.root/'logs/link';link.symlink_to(path)
  with self.assertRaises(ValueError):m.binding(link)
 def test_offload_overlay_replaces_only_reviewed_contract_and_adds_helper_test(self):
  names={'verl/utils/fsdp_utils.py','search_r1/baselines/fresh200_contract.py','tests/utils/test_fsdp_optimizer_transfer.py'}
  overlay=dict(status='PASS_CODE_ONLY_OFFLOAD_FIX_INDEPENDENT_REVIEW',base_source='379b14c65e127cd3a71524f7783deaebafbbdb41',must_fix=[],native_acceptance=False,files_sha256={n:'b'*64 for n in names})
  base={'search_r1/baselines/fresh200_contract.py':'a'*64,'verl/workers/actor/dp_actor.py':'c'*64}
  merged=m.merge_source_overlay(base,overlay,names)
  self.assertEqual(merged['search_r1/baselines/fresh200_contract.py'],'b'*64)
  self.assertEqual(merged['verl/workers/actor/dp_actor.py'],'c'*64)
  self.assertEqual(merged['verl/utils/fsdp_utils.py'],'b'*64)
  self.assertEqual(base['search_r1/baselines/fresh200_contract.py'],'a'*64)
 def test_offload_overlay_extra_missing_changed_source_or_native_claim_rejected(self):
  names={'verl/utils/fsdp_utils.py','search_r1/baselines/fresh200_contract.py','tests/utils/test_fsdp_optimizer_transfer.py'}
  overlay=dict(status='PASS_CODE_ONLY_OFFLOAD_FIX_INDEPENDENT_REVIEW',base_source='379b14c65e127cd3a71524f7783deaebafbbdb41',must_fix=[],native_acceptance=False,files_sha256={n:'b'*64 for n in names})
  for changed in [names|{'verl/workers/actor/dp_actor.py'},names-{'verl/utils/fsdp_utils.py'}]:
   with self.assertRaises(ValueError):m.merge_source_overlay({},overlay,changed)
  for key,value in [('native_acceptance',True),('base_source','8e'+'0'*38)]:
   bad=copy.deepcopy(overlay);bad[key]=value
   with self.assertRaises(ValueError):m.merge_source_overlay({},bad,names)
 def test_foreign_actual_environment_rejected_before_source_read(self):
  with patch.object(m,'CPU_RAW_SHA','f'*64),patch.object(m,'CONTEXT_SHA','f'*64):
   with self.assertRaisesRegex(ValueError,'Actual target ownROOT'):m.build_approval('f'*64)

if __name__=='__main__':unittest.main()
