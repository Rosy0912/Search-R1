"""Local temp controller plus fake SSH/Slurm; never remote or submits a real job."""
import unittest,tempfile,subprocess,os,json,hashlib,sys
from pathlib import Path
SOURCE=Path(__file__).parent/'deployment-tools/submit.sh'
class Tests(unittest.TestCase):
 def run_case(self,phase='hardware',mode='',id=None,intent=False,queue='',tamper=False):
  with tempfile.TemporaryDirectory()as t:
   p=Path(t);d=p/'scheduler';d.mkdir();root=p/'worker';root.mkdir();binary=p/'bin';binary.mkdir();events=p/'events'
   script=SOURCE.read_text().replace('D=/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-freeze-scalability-20261006/scheduler-head','D='+str(d)).replace('ROOT=/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-freeze-scalability-20261006','ROOT='+str(root));(d/'submit.sh').write_text(script)
   for q in ('hardware','formal300'):(d/(q+'.sbatch')).write_text('# mock sbatch never real\n')
   manifest={n:hashlib.sha256((d/n).read_bytes()).hexdigest()for n in ('submit.sh','hardware.sbatch','formal300.sbatch')}
   if tamper:manifest['submit.sh']='0'*64
   (d/'approved-launch-hashes.json').write_text(json.dumps(manifest))
   if id is not None:(d/(phase+'-job-id.txt')).write_text(id)
   if intent:(d/(phase+'-submit-intent.txt')).write_text(phase)
   mock='''#!%s
import os,sys,json,fcntl
from pathlib import Path
name=Path(sys.argv[0]).name
with open(os.environ['TEST_EVENT_LOG'],'a')as f:f.write(json.dumps([name,sys.argv[1:]])+'\\n')
m=os.environ['TEST_MODE']
if name=='flock':
 if m=='lockfail':sys.exit(1)
 fcntl.flock(int(sys.argv[-1]),fcntl.LOCK_EX|fcntl.LOCK_NB)
elif name=='ssh':
 if 'verify_launch.py'in sys.argv[-1]and m=='launchfail':sys.exit(9)
 if 'verify_hardware_receipt.py'in sys.argv[-1]and m=='hardwarefail':sys.exit(9)
 if 'submission-worker-intent.py'in sys.argv[-1]and m=='workerfail':sys.exit(9)
elif name=='squeue':
 if m=='queuefail':sys.exit(8)
 print(os.environ['TEST_QUEUE'])
elif name=='sbatch':
 if m=='submitfail':sys.exit(7)
 print('bad'if m=='badid'else'999')
'''%sys.executable
   for n in ('flock','ssh','squeue','sbatch','scontrol'):(binary/n).write_text(mock);(binary/n).chmod(0o755)
   env=dict(os.environ,PATH=str(binary)+os.pathsep+os.environ['PATH'],TEST_EVENT_LOG=str(events),TEST_MODE=mode,TEST_QUEUE=queue)
   r=subprocess.run(['bash',str(d/'submit.sh'),phase],env=env,capture_output=True,text=True,timeout=10)
   calls=[json.loads(x)for x in events.read_text().splitlines()]if events.exists()else[]
   return r,calls,(d/(phase+'-submit-intent.txt')).exists(),(d/(phase+'-job-id.txt')).read_text()if(d/(phase+'-job-id.txt')).exists()else None,str(root)
 def test_hardware_order_and_single_worker_intent(self):
  r,c,i,j,root=self.run_case();self.assertEqual(r.returncode,0);self.assertEqual(j,'999\n');self.assertTrue(i)
  names=[x[0]for x in c];self.assertEqual(names,['flock','ssh','squeue','ssh','sbatch','scontrol'])
  ssh=[x[1][-1]for x in c if x[0]=='ssh'];self.assertEqual(sum('submission-worker-intent.py'in x for x in ssh),1)
  for x in ssh:self.assertIn('cd '+root+'/repo && env PYTHONPATH='+root+'/repo '+root+'/env/bin/python -B ',x)
 def test_formal_requires_both_gates_before_submission(self):
  r,c,*_=self.run_case('formal300');self.assertEqual(r.returncode,0);self.assertEqual([x[0]for x in c],['flock','ssh','ssh','squeue','ssh','sbatch','scontrol']);self.assertIn('verify_hardware_receipt.py',c[2][1][-1])
 def test_existing_id_return_has_no_gate_or_submit(self):
  r,c,i,j,_=self.run_case(id='123');self.assertEqual(r.returncode,0);self.assertEqual(r.stdout,'123\n');self.assertEqual([x[0]for x in c],['flock']);self.assertFalse(i)
 def test_invalid_existing_id(self):
  r,c,*_=self.run_case(id='123;x');self.assertEqual(r.returncode,2);self.assertEqual([x[0]for x in c],['flock'])
 def test_unknown_prior_intent_refuses(self):
  r,c,*_=self.run_case(intent=True);self.assertEqual(r.returncode,3);self.assertEqual([x[0]for x in c],['flock'])
 def test_mixed_case_bapo_queue_blocks_before_intent(self):
  for q in ('BAPO-other','bApO-active','prefix_Bapo_active'):
   r,c,i,j,_=self.run_case(queue=q);self.assertEqual(r.returncode,3);self.assertFalse(i);self.assertIsNone(j);self.assertEqual([x[0]for x in c],['flock','ssh','squeue'])
 def test_gate_failure_before_intent(self):
  for phase,mode in (('hardware','launchfail'),('formal300','hardwarefail')):
   r,c,i,j,_=self.run_case(phase,mode);self.assertNotEqual(r.returncode,0);self.assertFalse(i);self.assertIsNone(j);self.assertNotIn('sbatch',[x[0]for x in c])
 def test_worker_failure_preserves_head_intent(self):
  r,c,i,j,_=self.run_case(mode='workerfail');self.assertNotEqual(r.returncode,0);self.assertTrue(i);self.assertIsNone(j);self.assertNotIn('sbatch',[x[0]for x in c])
 def test_uncertain_sbatch_and_badid_preserve_intent(self):
  for mode in ('submitfail','badid'):
   r,c,i,j,_=self.run_case(mode=mode);self.assertEqual(r.returncode,3);self.assertTrue(i);self.assertIsNone(j);self.assertEqual(sum(x[0]=='sbatch'for x in c),1)
 def test_queue_failure_not_assumed_empty(self):
  r,c,i,j,_=self.run_case(mode='queuefail');self.assertNotEqual(r.returncode,0);self.assertFalse(i);self.assertIsNone(j)
 def test_lock_or_manifest_failure_no_remote(self):
  for mode,tamper in (('lockfail',False),('',True)):
   r,c,i,j,_=self.run_case(mode=mode,tamper=tamper);self.assertNotEqual(r.returncode,0);self.assertEqual([x[0]for x in c],['flock']);self.assertFalse(i)
 def test_invalid_phase(self):
  r,c,*_=self.run_case('bogus');self.assertEqual(r.returncode,2);self.assertEqual(c,[])
if __name__=='__main__':unittest.main()
