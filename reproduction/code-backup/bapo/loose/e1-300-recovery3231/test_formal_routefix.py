"""Isolated controller execution: real files/flock, mocked SSH/scheduler only."""
from pathlib import Path
import tempfile,json,hashlib,subprocess,runpy,unittest,contextlib,io
from unittest.mock import patch
HERE=Path(__file__).resolve().parent;ROOT='/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-recovery-20261005'
class Tests(unittest.TestCase):
 def execute(self,queue='',failure=None,intent=False,old=False,tamper=False,priorid=False):
  with tempfile.TemporaryDirectory()as td:
   root=Path(td).resolve()/'root';base=root/'scheduler-head';d=root/'scheduler-formal-routefix'
   for p in(base,d):p.mkdir(parents=True,exist_ok=True)
   script=d/'submit-formal-routefix.py';script.write_text((HERE/script.name).read_text().replace(ROOT,str(root)))
   sbatch=base/'formal300.sbatch';sbatch.write_text('ORIGINAL');report=d/'FORMAL-ROUTEFIX-INDEPENDENT-REVIEW.md';report.write_text('fixture')
   h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
   (d/'approved-formal-routefix-hashes.json').write_text(json.dumps(dict(controller=h(script),sbatch=h(sbatch),independent_review=h(report),hardware_receipt='6b3cb7185d0043359c2633d2739a0c162f886aabb66c1cf8c1f5dffbda6291c1')))
   if intent:(d/'formal300-submit-intent.txt').write_text('uncertain')
   if old:(base/'formal300-submit-intent.txt').write_text('olduncertain')
   if priorid:(d/'formal300-job-id.txt').write_text('3240\n')
   if tamper:sbatch.write_text('TAMPER')
   calls=[];worker_intent=False
   def command(args,**kw):
    nonlocal worker_intent
    calls.append(args)
    if 'ssh'in args:
     remote=args[-1];assert remote.startswith('cd '+str(root/'repo')+' && env PYTHONPATH='+str(root/'repo')+' '+str(root/'env/bin/python')+' -B ')
     if failure and failure in remote:raise subprocess.CalledProcessError(1,args)
     if 'submission-worker-intent.py'in remote:worker_intent=True
    if args[0]=='sbatch':
     assert worker_intent and (d/'formal300-submit-intent.txt').exists()
     return subprocess.CompletedProcess(args,1 if failure=='sbatch'else 0,stdout='3240\n',stderr='mockfailure')
    return subprocess.CompletedProcess(args,0)
   error=None
   with patch('subprocess.run',side_effect=command),patch('subprocess.check_output',return_value=queue),contextlib.redirect_stdout(io.StringIO()):
    try:runpy.run_path(str(script),run_name='__main__')
    except SystemExit as e:error=e if e.code else None
    except Exception as e:error=e
   return dict(error=error,submits=sum(x[0]=='sbatch'for x in calls),calls=calls,headintent=(d/'formal300-submit-intent.txt').exists(),workerintent=worker_intent)
 def test_success_all_worker_calls_use_fixed_source_route(self):
  r=self.execute();self.assertIsNone(r['error']);self.assertEqual(r['submits'],1);self.assertEqual(sum('ssh'in x for x in r['calls']),4)
 def test_all_actual_gates_fail_before_intent(self):
  for f in ('import search_r1.baselines.recovery','verify_launch.py','verify_hardware_receipt.py'):
   r=self.execute(failure=f);self.assertIsNotNone(r['error']);self.assertEqual(r['submits'],0);self.assertFalse(r['headintent'])
 def test_queue_prior_bapo_and_malformed_rejected(self):
  for q in ('3239|x','999|Other_BAPO','malformed'):
   r=self.execute(queue=q);self.assertIsNotNone(r['error']);self.assertEqual(r['submits'],0);self.assertFalse(r['headintent'])
 def test_unrelated_foreign_queue_preserved(self):self.assertEqual(self.execute(queue='777|LTX')['submits'],1)
 def test_old_and_new_uncertain_intent_refuse(self):
  for opts in (dict(intent=True),dict(old=True)):
   r=self.execute(**opts);self.assertIsNotNone(r['error']);self.assertEqual(r['submits'],0)
 def test_existing_new_job_returns_without_calls(self):
  r=self.execute(priorid=True);self.assertIsNone(r['error']);self.assertEqual(r['calls'],[])
 def test_original_sbatch_tamper_refuses(self):
  r=self.execute(tamper=True);self.assertIsNotNone(r['error']);self.assertEqual(r['calls'],[])
 def test_unknown_sbatch_keeps_both_intents(self):
  r=self.execute(failure='sbatch');self.assertIsNotNone(r['error']);self.assertEqual(r['submits'],1);self.assertTrue(r['headintent']and r['workerintent'])
if __name__=='__main__':unittest.main()
