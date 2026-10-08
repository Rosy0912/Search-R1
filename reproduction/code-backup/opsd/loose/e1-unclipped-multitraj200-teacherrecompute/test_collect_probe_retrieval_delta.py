"""Two authorized optional members only; local CPU/stubs, never remote."""
import unittest,json,copy,base64
from pathlib import Path
from test_collect_probe_originals import M,REQUEST,payload
class Delta(unittest.TestCase):
 def test_two_optional_paths_and_standalone_rank_names(self):
  self.assertEqual(len(M.ALLOWED),52)
  self.assertIn('evidence/service.json',M.OPTIONAL);self.assertIn('evidence/http-top3-24.json',M.OPTIONAL)
  for rank in range(8):
   for prefix in ('teacher-logits-recompute-rank-','old-logprob-rank-','micro-4-rank-'):
    self.assertIn('evidence/capacity-micro4/'+prefix+str(rank)+'.json',M.ALLOWED)
  self.assertNotIn('resources.ndjson',M.ALLOWED);self.assertNotIn('full_state/driver.pt',M.ALLOWED)
 def test_optional_retrieval_bytes_kept_without_semantic_acceptance(self):
  p=payload()
  for i,n in enumerate(('evidence/service.json','evidence/http-top3-24.json'),100):
   raw=b'{"scope":"LOCAL_CPU_FIXTURE_NOT_REAL_SERVICE"}';st=[1,i,0,len(raw),10,10]
   p['files'][n]=dict(base64=base64.b64encode(raw).decode(),sha256=M.sha(raw),bytes=len(raw),uid=0,physical=REQUEST['run_root']+'/'+n,device=1,inode=i,stat_before=st,stat_after=st,reading_scope='STABLE_FULL_BYTES_AT_OBSERVATION_NOT_TERMINAL_PROOF');p['missing_optional'].remove(n)
  record,bodies=M.decode_snapshot(json.dumps(p).encode(),REQUEST)
  self.assertEqual(bodies['evidence/service.json'],raw);self.assertIn('NOT_ACCEPTANCE',record['scope'])
if __name__=='__main__':unittest.main(verbosity=2)
