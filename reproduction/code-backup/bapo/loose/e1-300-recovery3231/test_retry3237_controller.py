"""Execute copied controller against isolated fake scheduler/SSH; never remote."""
import hashlib,json,os,subprocess,tempfile,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT='/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-recovery-20261005'
class RetryTests(unittest.TestCase):
 def case(self,queue='',fail='',existing=None,uncertain=False,runs=False,terminal=None,bad_hash=False):
  with tempfile.TemporaryDirectory()as td:
   t=Path(td).resolve();root=t/'root';base=root/'scheduler-head';d=root/'scheduler-recovery3237';bin=t/'bin'
   for p in (base,d,bin,root/'logs',root/'runs'):p.mkdir(parents=True,exist_ok=True)
   (root/'source-expected.txt').write_text('7de6494bff08f35ffeefbf23ed9a9e1b73e5612e\n')
   if runs:(root/'runs/live').mkdir()
   script=d/'submit-hardware-after3237.sh';script.write_text((HERE/script.name).read_text().replace(ROOT,str(root)))
   (base/'hardware-job-id.txt').write_text('3237\n');(base/'hardware.sbatch').write_text('fixed_original_sbatch')
   (base/'hardware-3237-terminal.txt').write_text(terminal or 'JobId=3237 JobState=FAILED ExitCode=1:0 Restarts=0 NodeList=h100-5\n')
   review=d/'RETRY3237-INDEPENDENT-REVIEW.md';review.write_text('mockreview')
   h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
   (d/'approved-retry-hashes.json').write_text(json.dumps(dict(retry_script=h(script),sbatch=h(base/'hardware.sbatch'),prior_terminal=h(base/'hardware-3237-terminal.txt'),independent_review=h(review))))
   if bad_hash:(base/'hardware.sbatch').write_text('tamper')
   if existing:(d/'hardware-job-id.txt').write_text(existing+'\n')
   if uncertain:(d/'hardware-submit-intent.txt').write_text('prioruncertain')
   for name,body in {
    'flock':'''import sys
sys.exit(0)
''',
    'ssh':'''import sys,os,subprocess
c=sys.argv[-1]
with open(os.environ['CALLS'],'a')as f:f.write(c+'\\n')
if os.environ.get('FAIL') and os.environ['FAIL']in c:sys.exit(1)
if c.startswith("python3 -c"):sys.exit(subprocess.run(c,shell=True).returncode)
''',
    'squeue':'''import os
print(os.environ.get('QUEUE',''))
''',
    'sbatch':'''import os,sys
with open(os.environ['SUBMITS'],'a')as f:f.write('submit\\n')
if os.environ.get('FAIL')=='sbatch':sys.exit(1)
print('3240')
''',
    'scontrol':'''print('JobId=3240 JobState=PENDING')
''',
   }.items():
    p=bin/name;p.write_text('#!/usr/bin/env python3\n'+body);p.chmod(0o755)
   env=dict(os.environ,PATH=str(bin)+':'+os.environ['PATH'],QUEUE=queue,FAIL=fail,CALLS=str(t/'calls'),SUBMITS=str(t/'submits'))
   r=subprocess.run(['bash',str(script)],env=env,text=True,capture_output=True)
   count=len((t/'submits').read_text().splitlines())if(t/'submits').exists()else 0
   out=dict(rc=r.returncode,submitted=count,intent=(d/'hardware-submit-intent.txt').exists(),id=(d/'hardware-job-id.txt').read_text().strip()if(d/'hardware-job-id.txt').exists()else None,old=(base/'hardware-job-id.txt').read_text())
   if count and r.returncode==0:
    second=subprocess.run(['bash',str(script)],env=env,text=True,capture_output=True);out['second_rc']=second.returncode;out['total_submits']=len((t/'submits').read_text().splitlines())
   return out
 def test_success_once_and_old_ledger_retained(self):
  r=self.case();self.assertEqual((r['rc'],r['submitted'],r['second_rc'],r['total_submits']),(0,1,0,1));self.assertEqual(r['old'],'3237\n')
 def test_active_prior_and_case_insensitive_bapo_rejected(self):
  for q in ('3237|foreign','999|BAPO-other','999| foo-bapo-bar'):
   r=self.case(queue=q);self.assertNotEqual(r['rc'],0);self.assertEqual(r['submitted'],0)
 def test_unrelated_external_queue_preserved(self):
  r=self.case(queue='777|ltx-video');self.assertEqual(r['rc'],0);self.assertEqual(r['submitted'],1)
 def test_malformed_queue_fails_closed(self):
  r=self.case(queue='malformed');self.assertNotEqual(r['rc'],0);self.assertEqual(r['submitted'],0)
 def test_terminal_fields_exact(self):
  for field,bad in [('JobState=FAILED','JobState=FAILEDPLUS'),('ExitCode=1:0','ExitCode=1:00'),('NodeList=h100-5','NodeList=h100-5,h100-6'),('Restarts=0','Restarts=1')]:
   r=self.case(terminal=('JobId=3237 JobState=FAILED ExitCode=1:0 Restarts=0 NodeList=h100-5').replace(field,bad));self.assertNotEqual(r['rc'],0);self.assertEqual(r['submitted'],0)
 def test_uncertain_refuses(self):
  r=self.case(uncertain=True);self.assertNotEqual(r['rc'],0);self.assertEqual(r['submitted'],0)
 def test_existing_job_returns_without_submission(self):
  r=self.case(existing='3240');self.assertEqual(r['rc'],0);self.assertEqual(r['submitted'],0)
 def test_live_run_refuses_preserving_intent(self):
  r=self.case(runs=True);self.assertNotEqual(r['rc'],0);self.assertEqual(r['submitted'],0);self.assertTrue(r['intent'])
 def test_both_actual_gate_failures(self):
  for fail in ('verify_launch.py','verify_shared_inputs.py'):
   r=self.case(fail=fail);self.assertNotEqual(r['rc'],0);self.assertEqual(r['submitted'],0)
 def test_sbatch_failure_keeps_uncertain_intent(self):
  r=self.case(fail='sbatch');self.assertNotEqual(r['rc'],0);self.assertEqual(r['submitted'],1);self.assertTrue(r['intent']);self.assertIsNone(r['id'])
 def test_original_sbatch_tamper_rejected(self):
  r=self.case(bad_hash=True);self.assertNotEqual(r['rc'],0);self.assertEqual(r['submitted'],0)
if __name__=='__main__':unittest.main()
