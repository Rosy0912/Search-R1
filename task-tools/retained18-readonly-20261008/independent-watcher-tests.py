import contextlib,io,json,runpy,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
SCRIPT=Path.cwd()/'work/retained18-readonly-20261008/watch-short-connections.py'
COMMENT='BAPO_FRESH200:formal200:60f566b3ddabc88ce9de56ead5ad30a3d6795dd2:bab8981d9fd14afc98923eb4ebb89e33'
def doc(state='RUNNING',text=None):return {'job_id':3315,'exit_code':0,'stdout':text or 'JobId=3315 JobState='+state+' Comment='+COMMENT+' Restarts=0 Requeue=0','stderr':''}
class TestActualScript(unittest.TestCase):
 def simulate(self,payload=None,code=0,timeout=False):
  with tempfile.TemporaryDirectory() as td:
   dest=Path(td)/'observations';calls=[]
   def transport(args,**kwargs):
    calls.append(args);self.assertIn('transport.py',args[1]);self.assertEqual(kwargs['timeout'],140)
    if timeout:raise subprocess.TimeoutExpired(args,140)
    if code==0:
     p=Path(args[args.index('--output')+1]);p.write_text(payload if isinstance(payload,str) else json.dumps(payload))
    return subprocess.CompletedProcess(args,code,'mock','')
   with patch.object(sys,'argv',[str(SCRIPT),'--output-dir',str(dest)]),patch('subprocess.run',side_effect=transport),patch('time.sleep'),contextlib.redirect_stdout(io.StringIO()):runpy.run_path(str(SCRIPT),run_name='__main__')
   records=[json.loads(p.read_text()) for p in sorted(dest.glob('observation-*.json'))]
   return records,calls
 def unknown(self,payload):
  records,calls=self.simulate(payload);self.assertEqual(len(records),31);self.assertTrue(all(not r['training_terminal'] for r in records));self.assertTrue(all(r['job_state']=='UNKNOWN' for r in records))
 def test_normal_running_window_is_not_terminal(self):
  rs,calls=self.simulate(doc());self.assertEqual(len(calls),31);self.assertTrue(all(not r['training_terminal'] and r['job_state']=='RUNNING' for r in rs))
 def test_terminal_complete_requires_exact_identity(self):
  rs,calls=self.simulate(doc('COMPLETED'));self.assertEqual(len(calls),1);self.assertTrue(rs[0]['training_terminal'])
 def test_mixed_two_slurm_records_rejected(self):self.unknown(doc(text='JobId=3315 JobState=COMPLETED Comment=foreign\nJobId=9999 JobState=RUNNING Comment='+COMMENT))
 def test_comment_suffix_rejected(self):self.unknown(doc('COMPLETED',text='JobId=3315 JobState=COMPLETED Comment='+COMMENT+'SUFFIX'))
 def test_duplicate_slurm_field_rejected(self):self.unknown(doc(text='JobId=3315 JobId=3315 JobState=COMPLETED Comment='+COMMENT))
 def test_bad_json_rejected_and_recorded(self):self.unknown('{')
 def test_invalid_structure_and_bool_job_rejected(self):
  for p in [[],{'job_id':True,'exit_code':0,'stdout':'','stderr':''},{'job_id':3315}]:self.unknown(p)
 def test_transport255_not_terminal(self):
  rs,calls=self.simulate(code=255);self.assertEqual(len(calls),31);self.assertTrue(all(not r['training_terminal'] and r['job_state']=='UNKNOWN' and r['transport_exit_code']==255 for r in rs))
 def test_local_timeout_not_terminal(self):
  rs,calls=self.simulate(timeout=True);self.assertEqual(len(calls),31);self.assertTrue(all(not r['training_terminal'] and r['transport_exit_code']==124 for r in rs))
if __name__=='__main__':unittest.main()
