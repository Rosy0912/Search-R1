import importlib.util,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
TOOLS=Path(__file__).resolve().parents[1]/'tools'
spec=importlib.util.spec_from_file_location('launch',TOOLS/'launch_transaction.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Transactions(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
  self.original=m.ROOT;m.ROOT=self.root
  (self.root/'scheduler-head').mkdir();(self.root/'runs').mkdir();(self.root/'source-expected.txt').write_text('a'*40)
 def tearDown(self):m.ROOT=self.original;self.tmp.cleanup()
 def intent(self,phase=m.PHASES[0]):
  with patch('builtins.print'):m.prepare(phase)
  return json.loads((self.root/'scheduler-head'/(phase+'-submit-intent.json')).read_text())
 def job(self,intent,id='123'):
  return dict(JobId=id,Comment=intent['comment'],WorkDir=str(m.ROOT),JobName='bapo-fresh200-'+intent['phase'],UserId=intent['user']+'(1000)',NumNodes='1',ReqNodeList='h100-5',TresPerNode='gres:gpu:h100:8',Partition='gpu',OverSubscribe='NO',Command=str(m.ROOT/'tools'/(intent['phase']+'.sbatch')))
 def test_unknown_outcome_blocks_different_phase(self):
  self.intent()
  with self.assertRaisesRegex(RuntimeError,'unresolved intent'):m.prepare(m.PHASES[1])
 def test_records_exclusive_and_durable_fsync(self):
  path=self.root/'record.json'
  with patch.object(m.os,'fsync',wraps=m.os.fsync) as fs:m.durable(path,{'x':1});self.assertGreaterEqual(fs.call_count,2)
  with self.assertRaisesRegex(RuntimeError,'already exists'):m.durable(path,{'x':2})
  self.assertEqual(json.loads(path.read_text()),{'x':1})
 def test_id_nonce_owner_source_command_node_gpu_all_bound(self):
  intent=self.intent();intent['job_id']='123'
  for field,value in [('JobId','999'),('Comment','bad'),('UserId','foreign(42)'),('Command','/old/job.sbatch'),('ReqNodeList','h100-6'),('TresPerNode','gres/gpu:h100=4'),('WorkDir','/old/root')]:
   job=self.job(intent);job[field]=value
   with self.assertRaises(RuntimeError):m.identity(intent,job)
 def test_worker_intent_prevents_duplicate_manual_execution(self):
  intent=self.intent()
  with patch.object(m,'scheduler',return_value=self.job(intent)),patch('builtins.print'):m.worker(intent['phase'],'123')
  with patch.object(m,'scheduler',return_value=self.job(intent)),self.assertRaisesRegex(RuntimeError,'already exists'):m.worker(intent['phase'],'123')
 def test_existing_receipt_must_revalidate_scheduler(self):
  intent=self.intent()
  with patch.object(m,'scheduler',return_value=self.job(intent)):m.record(intent['phase'],'123')
  bad=self.job(intent);bad['Comment']='othernonce'
  with patch.object(m,'scheduler',return_value=bad),self.assertRaisesRegex(RuntimeError,'comment mismatch'):m.prepare(intent['phase'])
 def test_optimized_interpreter_rejected_before_imports(self):
  import subprocess,sys
  result=subprocess.run([sys.executable,'-O',str(TOOLS/'verify-launch.py'),'formal200'],capture_output=True,text=True)
  self.assertNotEqual(result.returncode,0);self.assertIn('optimized Python forbidden',result.stderr)
 def worker_root(self):
  root=self.root/'separate-worker-filesystem';root.mkdir()
  (root/'scheduler-head').mkdir();(root/'runs').mkdir();(root/'source-expected.txt').write_text('a'*40)
  return root
 def test_explicit_mirror_on_separate_filesystem_idempotent(self):
  intent=self.intent();raw=(self.root/'scheduler-head'/(intent['phase']+'-submit-intent.json')).read_bytes()
  head=m.ROOT;worker=self.worker_root();m.ROOT=worker
  m.admit('intent',raw);m.admit('intent',raw)
  self.assertEqual((worker/'scheduler-head'/(intent['phase']+'-submit-intent.json')).read_bytes(),raw)
  job=self.job(intent)
  receipt=dict(intent,job_id='123');data=(json.dumps(receipt,sort_keys=True)+'\n').encode()
  with patch.object(m,'scheduler',return_value=job):m.admit('receipt',data);m.admit('receipt',data)
  self.assertEqual(json.loads((worker/'scheduler-head'/(intent['phase']+'-receipt.json')).read_text()),receipt)
  self.assertFalse((head/'scheduler-head'/(intent['phase']+'-receipt.json')).exists())
 def test_worker_unknown_outcome_blocks_other_phase_even_if_head_lost(self):
  intent=self.intent();raw=(self.root/'scheduler-head'/(intent['phase']+'-submit-intent.json')).read_bytes()
  worker=self.worker_root();m.ROOT=worker;m.admit('intent',raw)
  other=dict(intent,phase=m.PHASES[1],nonce='b'*32)
  other['comment']='BAPO_FRESH200:'+other['phase']+':'+other['source']+':'+other['nonce']
  with self.assertRaisesRegex(RuntimeError,'unresolved intent'):m.admit('intent',(json.dumps(other,sort_keys=True)+'\n').encode())
 def test_worker_rejects_new_nonce_same_phase_and_noncannonical_bytes(self):
  intent=self.intent();raw=(self.root/'scheduler-head'/(intent['phase']+'-submit-intent.json')).read_bytes()
  worker=self.worker_root();m.ROOT=worker;m.admit('intent',raw)
  with self.assertRaisesRegex(RuntimeError,'canonical'):m.admit('intent',json.dumps(intent).encode())
  other=dict(intent,nonce='b'*32);other['comment']='BAPO_FRESH200:'+other['phase']+':'+other['source']+':'+other['nonce']
  with self.assertRaises(RuntimeError):m.admit('intent',(json.dumps(other,sort_keys=True)+'\n').encode())
 def test_worker_receipt_without_intent_and_wrong_scheduler_rejected(self):
  intent=self.intent();raw=(self.root/'scheduler-head'/(intent['phase']+'-submit-intent.json')).read_bytes()
  worker=self.worker_root();m.ROOT=worker
  receipt=dict(intent,job_id='123');data=(json.dumps(receipt,sort_keys=True)+'\n').encode()
  with self.assertRaisesRegex(RuntimeError,'without exact'):m.admit('receipt',data)
  m.admit('intent',raw);bad=self.job(intent);bad['Comment']='wrong'
  with patch.object(m,'scheduler',return_value=bad),self.assertRaisesRegex(RuntimeError,'comment mismatch'):m.admit('receipt',data)
  self.assertFalse((worker/'scheduler-head'/(intent['phase']+'-receipt.json')).exists())
 def test_restarted_allocation_has_no_execution_intent(self):
  intent=self.intent()
  with patch.dict(m.os.environ,{'SLURM_RESTART_COUNT':'1'}),self.assertRaisesRegex(RuntimeError,'restarted'):m.worker(intent['phase'],'123')
  self.assertFalse((self.root/'scheduler-head'/(intent['phase']+'-worker-intent.json')).exists())
 def test_scripts_explicit_mirror_failclosed_and_no_requeue(self):
  text=(TOOLS/'submit.sh').read_text()
  self.assertLess(text.rindex('mirror intent'),text.index('record "$phase" "$id"'))
  self.assertIn('mirror receipt\necho',text)
  self.assertIn('env -u LD_LIBRARY_PATH ssh',text);self.assertIn('ConnectTimeout=15',text)
  self.assertIn('BatchMode=yes',text);self.assertIn('sbatch --no-requeue',text)
  for phase in m.PHASES:self.assertIn('#SBATCH --no-requeue',(TOOLS/(phase+'.sbatch')).read_text())
if __name__=='__main__':unittest.main()
