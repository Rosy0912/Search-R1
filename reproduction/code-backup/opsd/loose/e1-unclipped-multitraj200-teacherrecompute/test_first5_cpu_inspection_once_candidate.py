"""Local CPU fixtures; Slurm/PID owner mocks do not prove actual node execution."""
import hashlib,importlib.util,json,os,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('candidate',HERE/'first5_cpu_inspection_once_candidate.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Tests(unittest.TestCase):
 def test_command_fixed_cpu_only(self):
  x=m.srun_command(Path('/fixture/launch'), 'a'*64)
  for v in ['--jobid=3304','--overlap','--gres=none','--nodelist=h100-10','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=1']:self.assertIn(v,x)
  self.assertNotIn('sbatch',x);self.assertIn('cpu-step',x)
 def test_exact_current_contract(self):
  self.assertEqual(m.CONTRACT,{'global_mini_batch':256,'optimizer_steps_per_rank_per_outer':10,'saved_adam_steps':50,'horizon':200,'warmup_steps':57,'base_learning_rate':1e-6})
 def test_duplicate_json_rejected(self):
  with self.assertRaises(ValueError):m.unique(b'{"job_id":"3304","job_id":"3304"}')
 def test_nonfinite_and_invalid_pin_rejected(self):
  for b in [b'{"x":NaN}',b'{"x":Infinity}']:
   with self.assertRaises(ValueError):m.unique(b)
  for x in [True,'f'*63,'G'*64]:
   with self.assertRaises(ValueError):m.pin(x)
 def test_exclusive_file(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'x';m.write_once(p,{'fixture':True});b=p.read_bytes()
   with self.assertRaises(FileExistsError):m.write_once(p,{})
   self.assertEqual(b,p.read_bytes())
 def waited(self,code):
  with tempfile.TemporaryDirectory() as d,patch.object(m,'process',lambda pid,**k:{'pid':pid,'fixture':'LOCAL_CPU_NOT_PRODUCTION_IDENTITY'}):
   p=Path(d);r=m.wait_child([sys.executable,'-c',code],p,'child');raw=json.loads((p/'child-exit.original.json').read_bytes());self.assertTrue(raw['waited']);self.assertEqual(raw['waited_by_pid'],os.getpid());self.assertEqual(raw['returncode'],r if r<128 else -(r-128));return r
 def test_real_child_exit0(self):self.assertEqual(self.waited('print("rawfixture");raise SystemExit(0)'),0)
 def test_real_child_exit7(self):self.assertEqual(self.waited('raise SystemExit(7)'),7)
 def test_real_child_sigterm(self):self.assertEqual(self.waited('import os,signal;os.kill(os.getpid(),signal.SIGTERM)'),143)
 def test_metadata_error_finally_true_waited_exit_saved(self):
  with tempfile.TemporaryDirectory() as d,patch.object(m,'process',side_effect=ValueError('LOCAL_CPU_FIXTURE_METADATA_FAILURE')):
   p=Path(d);t=time.monotonic()
   with self.assertRaisesRegex(ValueError,'METADATA_FAILURE'):m.wait_child([sys.executable,'-c','import time;time.sleep(.08);raise SystemExit(6)'],p,'child')
   e=json.loads((p/'child-exit.original.json').read_bytes());self.assertEqual(e['returncode'],6);self.assertTrue(e['waited']);self.assertGreater(time.monotonic()-t,.07)
 def test_context_precheck_no_start(self):
  args=type('A',(),dict(job_id='3304',controller_sha256='a'*64,checkpoint_manifest_sha256='b'*64,pause_nonce='c'*64,driver_pid=999,driver_start_ticks='123'))()
  with tempfile.TemporaryDirectory() as d:
   run=Path(d)/'run';(run/'evidence').mkdir(parents=True);out=run/'evidence/first5-cpu-inspection-once-v1'
   with patch.object(m,'RUN',run),patch.object(m,'OUTPUT',out),patch.object(m,'self_pin',lambda *a:None),patch.object(m,'physical',lambda p,*a:p),patch.object(m,'context',side_effect=ValueError('LOCAL_CPU_NOT_ACTUALLY_PAUSED')):
    with self.assertRaisesRegex(ValueError,'NOT_ACTUALLY_PAUSED'):m.start(args)
   self.assertFalse(out.exists())
 def test_existing_marker_not_retried_or_modified(self):
  args=type('A',(),dict(job_id='3304',controller_sha256='a'*64,checkpoint_manifest_sha256='b'*64,pause_nonce='c'*64,driver_pid=999,driver_start_ticks='123'))()
  with tempfile.TemporaryDirectory() as d:
   run=Path(d)/'run';out=run/'evidence/first5-cpu-inspection-once-v1';out.mkdir(parents=True);(out/'sentinel').write_bytes(b'original')
   with patch.object(m,'RUN',run),patch.object(m,'OUTPUT',out),patch.object(m,'self_pin',lambda *a:None),patch.object(m,'physical',lambda p,*a:p),patch.object(m,'context',lambda *a:{'scope':'LOCAL_CPU_MOCK'}),patch.object(m.subprocess,'Popen') as pop:
    with self.assertRaisesRegex(ValueError,'never retried'):m.start(args)
    pop.assert_not_called()
   self.assertEqual((out/'sentinel').read_bytes(),b'original')
 def test_cpu_step_exec_keeps_identity_and_exact_inspector(self):
  with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,{'SLURM_JOB_ID':'3304','SLURM_STEP_ID':'1','CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1'}),patch.object(m,'host',lambda:None),patch.object(m,'context',lambda x:{'scope':'LOCAL_CPU_CONTEXT'}),patch.object(m,'process',lambda pid,*a:{'pid':pid,'start_ticks':'FIXTURE'}),patch.object(m,'OUTPUT',Path(d)),patch.object(m.os,'execv') as ev:
   m.cpu_step({});ev.assert_called_once();self.assertEqual(ev.call_args.args[0],str(m.PYTHON));self.assertEqual(ev.call_args.args[1][1],str(m.FIRST5/'inspect_first5_cpu.py'));self.assertIn('--tools-manifest-sha256',ev.call_args.args[1])
 def test_cpu_step_cuda_or_wrong_job_rejected(self):
  for env in [{'SLURM_JOB_ID':'foreign','CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1'},{'SLURM_JOB_ID':'3304','CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'1'}]:
   with patch.dict(os.environ,env),patch.object(m,'host',lambda:None),patch.object(m.os,'execv') as ev:
    with self.assertRaises(ValueError):m.cpu_step({})
    ev.assert_not_called()
if __name__=='__main__':unittest.main(verbosity=2)
