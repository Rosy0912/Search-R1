"""Replay actual downloaded target bytes; only path access is mapped locally."""
import copy,hashlib,importlib.util,json,unittest
from pathlib import Path
from unittest.mock import patch
P=Path(__file__).resolve().parents[1]/'publish-launch-approval.CANDIDATE.py'
s=importlib.util.spec_from_file_location('actual_prep_publisher',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
ACTUAL=P.parents[1]/'evidence/fresh200-controller-initfix-actual-preparation-v2'
class Prep(unittest.TestCase):
 def raw(self,p):
  relative=Path(p).relative_to(m.ROOT)
  if relative.parts[0]=='logs':return (ACTUAL/relative.name).read_bytes()
  if relative.parts[0]=='tests':return (P.parent/relative).read_bytes()
  raise ValueError('No actual preparation mapping for '+str(p))
 def test_actual_192_and_all221317_context(self):
  cpu=json.loads((ACTUAL/'target-cpu-gate.json').read_text());context=json.loads((ACTUAL/'context-budget-audit.json').read_text())
  with patch.object(m,'raw',self.raw):m.validate_cpu(cpu,context,cpu['critical_source_sha256'])
 def test_actual_80_resource_and_new_test_hashes(self):
  resource=json.loads((ACTUAL/'runtime-tests.actual.json').read_text())
  with patch.object(m,'raw',self.raw):m.validate_resource(resource)
 def test_old_source_or186_cannot_substitute(self):
  cpu=json.loads((ACTUAL/'target-cpu-gate.json').read_text());context=json.loads((ACTUAL/'context-budget-audit.json').read_text());cpu['source_commit']=m.ANCESTOR_SOURCE
  with patch.object(m,'raw',self.raw),self.assertRaises(ValueError):m.validate_cpu(cpu,context,cpu['critical_source_sha256'])
 def test_actual_goal_is_newsource_human_configuration_not_native(self):
  goal=(ACTUAL/'latest-human-goal-behavior.json').read_bytes()
  self.assertEqual(hashlib.sha256(goal).hexdigest(),'a3c57fdc1a2a74f226dd194d1e7e7a4325e0721c716769fce2647f26ff52be16')
  fixed=json.loads((ACTUAL/'frozen-policy-safety-config.json').read_text());self.assertEqual(fixed['source_commit'],m.SOURCE);self.assertEqual(fixed['target_old_policy_kl'],.2)
if __name__=='__main__':unittest.main()
