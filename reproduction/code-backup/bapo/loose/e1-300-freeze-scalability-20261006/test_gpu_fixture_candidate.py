"""CPU negative tests of the GPU fixture contract, never GPU acceptance."""
import copy,importlib.util,math,os,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('gpu_fixture',Path(__file__).with_name('freeze_scalability_gpu_fixture.py'));f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
class Tests(unittest.TestCase):
 def record(self):return dict(rank=0,world_size=8,backend='nccl',source_commit='f'*40,job_id=12,run_root='/new/run',watch_sha256=f.WATCH_SHA,fixture_sha256=f.sha(f.__file__),pid=100,start_ticks=999,completed_collectives=4,allreduce_value=36.,cuda_device=0,nccl_socket_ifname='lo',nccl_ib_disable='1')
 def test_rank_valid_schema_only(self):self.assertEqual(f.validate_rank_record(self.record(),0,'f'*40,12,Path('/new/run'),dict(pid=100,start_ticks=999)),4)
 def test_rank_wrong_semantics_rejected(self):
  for key,value in [('rank',True),('backend','gloo'),('source_commit','old'),('job_id',1),('watch_sha256','fake'),('fixture_sha256','fake'),('pid',1),('start_ticks',1),('completed_collectives',False),('allreduce_value',float('nan')),('allreduce_value',35.),('cuda_device',True),('nccl_socket_ifname','eth0'),('nccl_ib_disable','0')]:
   with self.subTest(key=key,value=value):
    row=self.record();row[key]=value
    with self.assertRaises(ValueError):f.validate_rank_record(row,0,'f'*40,12,Path('/new/run'),dict(pid=100,start_ticks=999))
 def test_paused_contract_requires_310_all_stopped_and_no_progress(self):
  first={i:5 for i in range(8)};states={i:dict(pid=i,ppid=1,start_ticks=i+100,uid=os.getuid(),state='T')for i in range(310)};created=copy.deepcopy(states);f.validate_paused(first,dict(first),states,created)
  for a,b,c in [(first,dict(first,**{}),dict(list(states.items())[:309])),(first,{i:6 for i in range(8)},states),(first,dict(first),dict(states,**{}))]:
   if len(c)==310 and a==b:c=copy.deepcopy(c);c[0]['state']='S'
   with self.assertRaises(ValueError):f.validate_paused(a,b,c,created)
 def test_paused_reused_epoch_rejected(self):
  counts={i:5 for i in range(8)};states={i:dict(pid=i,ppid=1,start_ticks=i+100,uid=os.getuid(),state='T')for i in range(310)};created=copy.deepcopy(states);states[100]['start_ticks']=9999
  with self.assertRaises(ValueError):f.validate_paused(counts,dict(counts),states,created)
 def test_outer_actual_slot_is_fresh_scheduling_only(self):
  row=dict(time=100.,run='/new/run',slurm_job_id='12',pause=False,reason='within_threshold');self.assertTrue(f.fresh_outer_slot(row,12,Path('/new/run'),101.))
  for key,value in [('time',True),('time',98.),('time',103.),('time',float('nan')),('run','foreign'),('slurm_job_id','1'),('pause',True),('reason','monitor_unknown_fail_closed'),('event','recovery_started')]:
   with self.subTest(key=key):
    bad=dict(row);bad[key]=value;self.assertFalse(f.fresh_outer_slot(bad,12,Path('/new/run'),101.))
 def test_scope_does_not_hide_other_same_run_process(self):
  class W:
   def freeze_inventory(self,*a):return {1:dict(start_ticks=10,uid=os.getuid()),2:dict(start_ticks=11,uid=os.getuid())}
  with self.assertRaisesRegex(ValueError,'Unrelated'):f.checked_inventory(W(),'/new/run','12',set(),1,{1:dict(start_ticks=10,uid=os.getuid())})
  with self.assertRaisesRegex(ValueError,'epoch'):f.checked_inventory(W(),'/new/run','12',set(),1,{1:dict(start_ticks=99,uid=os.getuid()),2:dict(start_ticks=11,uid=os.getuid())})
 def test_cleanup_reused_pid_never_opens_or_signals(self):
  with patch.object(f,'stat',return_value=dict(start_ticks=99,uid=os.getuid())),patch.object(os,'pidfd_open',create=True)as opened,patch.object(f.signal,'pidfd_send_signal',create=True)as sent:
   events=[];f.bounded_created_cleanup({100:dict(start_ticks=10)},'/new/run',12,events.append);opened.assert_not_called();sent.assert_not_called();self.assertEqual(events[0]['event'],'fixture_cleanup_refused')
 def test_cleanup_foreign_marker_never_opens_or_signals(self):
  with patch.object(f,'stat',return_value=dict(start_ticks=10,uid=os.getuid())),patch.object(Path,'read_bytes',return_value=b'BAPO_RUN_ID=/foreign\0SLURM_JOB_ID=12'),patch.object(os,'pidfd_open',create=True)as opened,patch.object(f.signal,'pidfd_send_signal',create=True)as sent:
   f.bounded_created_cleanup({100:dict(start_ticks=10)},'/new/run',12,lambda x:None);opened.assert_not_called();sent.assert_not_called()
 def test_cleanup_exec_to_foreign_after_pidfd_refuses_signal(self):
  reads=[b'BAPO_RUN_ID=/new/run\0SLURM_JOB_ID=12',b'BAPO_RUN_ID=/foreign\0SLURM_JOB_ID=12']
  def proc_read(path):
   if path.name=='cmdline':return ('python\0'+str(Path(f.__file__).resolve())+'\0--mode\0rank\0').encode()
   return reads.pop(0)
  with patch.object(f,'stat',return_value=dict(start_ticks=10,uid=os.getuid())),patch.object(Path,'read_bytes',proc_read),patch.object(os,'pidfd_open',return_value=99,create=True)as opened,patch.object(f.signal,'pidfd_send_signal',create=True)as sent,patch.object(os,'close')as close:
   f.bounded_created_cleanup({100:dict(start_ticks=10)},'/new/run',12,lambda x:None);opened.assert_called_once_with(100,0);sent.assert_not_called();close.assert_called_once_with(99)
 def test_foreign_sentinel_killed_or_failed_cannot_claim_natural_exit(self):
  f.validate_sentinel_exit(0)
  for status in (-9,-15,1,None,False):
   with self.subTest(returncode=status),self.assertRaisesRegex(ValueError,'naturally'):f.validate_sentinel_exit(status)
  # The actual controller must enforce this after its wait; no signal shortcut.
  import ast
  tree=ast.parse(Path(f.__file__).read_text());controller=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='controller')
  final=next(n for n in controller.body if isinstance(n,ast.With)).body
  block=next(n for n in final if isinstance(n,ast.Try)).finalbody
  calls=[ast.unparse(n.value)for n in block if isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)]
  self.assertLess(calls.index('foreign.wait(timeout=5)'),calls.index('validate_sentinel_exit(foreign.returncode)'))
if __name__=='__main__':unittest.main(verbosity=2)
