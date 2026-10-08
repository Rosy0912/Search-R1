"""Read local original actual bytes via path map; no remote execution/admission."""
import copy,importlib.util,json,hashlib,unittest
from pathlib import Path
from unittest.mock import patch
B=Path(__file__).resolve().parents[1];W=B.parent
s=importlib.util.spec_from_file_location('numeric_publisher',B/'publish-launch-approval.CANDIDATE.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class ActualNumeric(unittest.TestCase):
 def setUp(self):
  self.man=json.loads((W/'evidence/numeric3311-derived-terminal-evidence-v2/numeric-terminal-evidence.json').read_text())
  self.result=json.loads((W/'evidence/fresh200-numeric-diag3311-numeric-probe.actual.json').read_text())
  self.review=json.loads((W.parent/'outputs/fresh200/3311-CONDITIONAL-FIRST5-INDEPENDENT-INTERPRETATION.json').read_text())
  candidates=list((W/'evidence/fresh200-numeric-diag3311-terminal-raw').rglob('*'))+list((W/'evidence/numeric3311-derived-terminal-evidence-v2').rglob('*'))
  hashes={hashlib.sha256(p.read_bytes()).hexdigest():p for p in candidates if p.is_file()}
  self.mapping={v['path']:hashes[v['sha256']]for v in self.man['artifacts'].values()}
 def raw(self,p):return self.mapping[str(p)].read_bytes()
 def binding(self,p,expected=None):
  b=self.raw(p);h=hashlib.sha256(b).hexdigest()
  m.require(expected is None or expected==h,'Artifact bytes changed')
  return dict(path=str(p),bytes=len(b),sha256=h)
 def validate(self,man,result,review):
  with patch.object(m,'raw',self.raw),patch.object(m,'binding',self.binding),patch('getpass.getuser',return_value='root'):return m.validate_numeric_interpretation(man,result,review)
 def test_actual_all12_artifacts_and_math_interpretation_replay(self):self.assertEqual(len(self.validate(self.man,self.result,self.review)),12)
 def test_actual_negative_identity_scope_and_binding(self):
  for key,value in [('native_acceptance',True),('source_commit',m.AEC_SOURCE),('whole_200_accepted',True),('run_root','/foreign')]:
   r=copy.deepcopy(self.review);r[key]=value
   with self.assertRaises(ValueError):self.validate(self.man,self.result,r)
  r=copy.deepcopy(self.result);r['optimizer_writes']=1
  with self.assertRaises(ValueError):self.validate(self.man,r,self.review)
  r=copy.deepcopy(self.man);r['artifacts']['resource_replay']['sha256']='0'*64
  with self.assertRaises(ValueError):self.validate(r,self.result,self.review)
if __name__=='__main__':unittest.main()
