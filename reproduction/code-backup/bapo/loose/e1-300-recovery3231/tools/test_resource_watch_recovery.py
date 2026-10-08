import importlib.util,json,os,signal,tempfile,unittest,subprocess,sys,time
from pathlib import Path
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('watch',Path(__file__).with_name('resource_watch.py'));w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
UUIDS={'GPU-'+str(i)for i in range(8)}
class Clock:
 def __init__(self):self.t=0
 def __call__(self):return self.t
 def sleep(self,n):self.t+=n
class Child:
 pid=101
 def poll(self):return None
class FakeFreezer:
 def __init__(self,*a):self.frozen=0;self.thawed=0;FakeFreezer.last=self
 def freeze(self):self.frozen+=1
 def thaw(self,abort=False):self.thawed+=1;self.abort=abort
class Tests(unittest.TestCase):
 def setUp(self):
  self.env=patch.dict(os.environ,SLURM_JOB_ID='3231',SEARCH_R1_GPU_LEASES=json.dumps([{'uuid':u}for u in UUIDS]));self.env.start()
 def tearDown(self):self.env.stop()
 def recovery(self,events,child=Child(),factory=FakeFreezer,grace=6):
  clock=Clock();logs=[];it=iter(events)
  result=w.recover('/run',[],child,logs.append,grace,1,3,sampler=lambda *a:next(it,dict(pause=True)),factory=factory,clock=clock,sleep=clock.sleep)
  return result,logs
 def test_unknown_then_three_valid(self):
  good,logs=self.recovery([dict(pause=True),dict(pause=False),dict(pause=False),dict(pause=False)])
  self.assertTrue(good);self.assertEqual(FakeFreezer.last.thawed,1);self.assertEqual(sum(x.get('event')=='recovery_sample'for x in logs),4)
 def test_persistent_unknown(self):
  good,logs=self.recovery([dict(pause=True)]*20);self.assertFalse(good);self.assertEqual(FakeFreezer.last.thawed,1)
 def test_persistent_real_exceed(self):
  good,logs=self.recovery([dict(pause=True,exceeded={'0':{'external_sm_percent':98}})]*20);self.assertFalse(good)
 def test_safe_counter_resets(self):
  good,logs=self.recovery([dict(pause=False),dict(pause=False),dict(pause=True),dict(pause=False),dict(pause=False),dict(pause=False)],grace=7);self.assertTrue(good)
 def test_unstarted_child_never_freezes(self):
  def forbidden(*a):raise AssertionError('factory must not run')
  good,logs=self.recovery([dict(pause=False)]*3,None,forbidden);self.assertTrue(good)
 def test_freeze_failure_thaws(self):
  class Failed(FakeFreezer):
   def freeze(self):raise ValueError('cannot stop')
  with self.assertRaises(ValueError):self.recovery([],factory=Failed)
  self.assertEqual(FakeFreezer.last.thawed,1)
 def test_signal_interrupt_thaws(self):
  class Interrupted(FakeFreezer):
   def freeze(self):raise w.RecoveryInterrupted(signal.SIGTERM)
  with self.assertRaises(w.RecoveryInterrupted):self.recovery([],factory=Interrupted)
  self.assertEqual(FakeFreezer.last.thawed,1)
 def freezer(self,rows,mutator=None,sender=None):
  clock=Clock();state={p:dict(r)for p,r in rows.items()};calls=[];logs=[]
  def inventory(*a):
   if mutator:mutator(state)
   return {p:dict(r)for p,r in state.items()}
  def send(fd,sig,*a):
   calls.append((fd,sig))
   if sender:sender(fd,sig,state)
   elif fd in state:state[fd]['state']='T'if sig==signal.SIGSTOP else'S'
  f=w.OwnedFreezer('/run',101,logs.append,clock,clock.sleep,inventory,lambda p,flags:p,send,lambda fd:None)
  return f,state,calls,logs
 def rows(self):return {101:dict(pid=101,ppid=100,allocated_uuid_set=sorted(UUIDS),gpu_leases_inherited=False,inherited_gpu_lease_uuids=[],start_ticks=7,uid=os.getuid(),run='/run',slurm_job_id='3231',state='S',role='phase',command=['bash','phase.sh'])}
 def test_full_certificate_preserves_cpu_without_inherited_leases(self):
  f,state,calls,logs=self.freezer(self.rows());f.freeze()
  event=next(x for x in logs if x.get('event')=='owned_freeze_signal')
  self.assertEqual(event['certificate'],dict(self.rows()[101]));self.assertFalse(event['certificate']['gpu_leases_inherited'])
  self.assertEqual(event['certificate']['inherited_gpu_lease_uuids'],[])
  self.assertEqual(event['certificate']['allocated_uuid_set'],sorted(UUIDS))
  f.thaw()
 def test_lease_identity_change_after_pidfd_rejected(self):
  count=[0]
  def mutate(s):
   count[0]+=1
   if count[0]==2:s[101]['gpu_leases_inherited']=True;s[101]['inherited_gpu_lease_uuids']=sorted(UUIDS)
  f,state,calls,logs=self.freezer(self.rows(),mutate)
  with self.assertRaisesRegex(ValueError,'identity changed'):f.freeze()
  self.assertEqual(calls,[])
 def test_parent_change_after_pidfd_rejected(self):
  count=[0]
  def mutate(s):
   count[0]+=1
   if count[0]==2:s[101]['ppid']=999
  f,state,calls,logs=self.freezer(self.rows(),mutate)
  with self.assertRaisesRegex(ValueError,'identity changed'):f.freeze()
  self.assertEqual(calls,[])
 def test_pid_reuse_before_stop_rejected(self):
  count=[0]
  def mutate(s):
   count[0]+=1
   if count[0]==2:s[101]['start_ticks']=8
  f,state,calls,logs=self.freezer(self.rows(),mutate)
  with self.assertRaises(ValueError):f.freeze()
  self.assertEqual(calls,[])
 def test_spawn_caught_until_stable(self):
  def send(fd,sig,state):
   state[fd]['state']='T'if sig==signal.SIGSTOP else'S'
   if fd==101 and sig==signal.SIGSTOP:state[102]=dict(state[101],pid=102,start_ticks=9,state='S')
  f,state,calls,logs=self.freezer(self.rows(),sender=send);f.freeze();self.assertEqual(set(f.held),{101,102});self.assertTrue(all(r['state']=='T'for r in state.values()));f.thaw();self.assertTrue(all(r['state']=='S'for r in state.values()))
 def test_unfreezable_running_worker_rejects(self):
  f,state,calls,logs=self.freezer(self.rows(),sender=lambda *a:None)
  with self.assertRaises(ValueError):f.freeze(timeout=.3)
  f.thaw();self.assertIn((101,signal.SIGCONT),calls)
 def test_stopped_original_pidfd_not_replacement(self):
  f,state,calls,logs=self.freezer(self.rows());f.freeze();state[101]['start_ticks']=999;f.thaw()
  self.assertEqual(calls[-1],(101,signal.SIGCONT)) # injected fd points original epoch, never os.kill
 def test_cleanup_exited_original_pid(self):
  f,state,calls,logs=self.freezer(self.rows());f.freeze()
  f.sender=lambda *a:(_ for _ in ()).throw(ProcessLookupError());f.thaw();self.assertFalse(f.held)
 def test_close_failure_does_not_leave_other_workers_stopped(self):
  rows=self.rows();rows[102]=dict(rows[101],pid=102,start_ticks=9)
  f,state,calls,logs=self.freezer(rows);f.freeze()
  f.closer=lambda fd:(_ for _ in ()).throw(OSError('close'))
  with self.assertRaisesRegex(ValueError,'close'):f.thaw()
  self.assertIn((101,signal.SIGCONT),calls);self.assertIn((102,signal.SIGCONT),calls);self.assertFalse(f.held)
 def test_emit_failure_does_not_leave_other_workers_stopped(self):
  rows=self.rows();rows[102]=dict(rows[101],pid=102,start_ticks=9)
  f,state,calls,logs=self.freezer(rows);f.freeze()
  f.log=lambda row:(_ for _ in ()).throw(OSError('audit write'))
  with self.assertRaisesRegex(ValueError,'audit'):f.thaw()
  self.assertIn((101,signal.SIGCONT),calls);self.assertIn((102,signal.SIGCONT),calls);self.assertFalse(f.held)
 def test_transient_resume_failure_is_retried(self):
  f,state,calls,logs=self.freezer(self.rows());f.freeze();real=f.sender;counter=[0]
  def send(*args):
   counter[0]+=1
   if counter[0]<3:raise OSError('transient')
   return real(*args)
  f.sender=send;f.thaw();self.assertEqual(counter[0],3)
 def test_stop_signal_exception_preserves_pidfd_for_cleanup(self):
  f,state,calls,logs=self.freezer(self.rows());real=f.sender
  def send(fd,sig,*a):
   real(fd,sig,*a)
   if sig==signal.SIGSTOP:raise w.RecoveryInterrupted(signal.SIGTERM)
  f.sender=send
  with self.assertRaises(w.RecoveryInterrupted):f.freeze()
  f.thaw();self.assertIn((101,signal.SIGCONT),calls)
 def test_abort_kills_workers_before_phase_cleanup_cont(self):
  rows=self.rows();rows[102]=dict(rows[101],pid=102,start_ticks=9,role='ray_worker')
  f,state,calls,logs=self.freezer(rows);f.freeze();calls.clear();f.thaw(abort=True)
  self.assertEqual(calls,[(102,signal.SIGKILL),(101,signal.SIGTERM),(101,signal.SIGCONT)])
  self.assertNotIn((102,signal.SIGCONT),calls)
 def test_persistent_failure_cleanup_is_abort_not_compute_resume(self):
  good,logs=self.recovery([dict(pause=True)]*20);self.assertFalse(good);self.assertTrue(FakeFreezer.last.abort)
 def test_safe_recovery_is_resume_not_abort(self):
  good,logs=self.recovery([dict(pause=False)]*3);self.assertTrue(good);self.assertFalse(FakeFreezer.last.abort)
 def test_fail_abort_logging_failure_still_kills_all_workers(self):
  rows=self.rows();rows[102]=dict(rows[101],pid=102,start_ticks=9,role='ray_worker')
  f,state,calls,logs=self.freezer(rows);f.freeze();calls.clear();f.log=lambda x:(_ for _ in ()).throw(OSError('log'))
  with self.assertRaises(ValueError):f.thaw(abort=True)
  self.assertIn((102,signal.SIGKILL),calls);self.assertIn((101,signal.SIGCONT),calls)
 def test_slow_third_safe_after_deadline_cannot_resume(self):
  clock=Clock();logs=[];counter=[0]
  def sample(*a):
   counter[0]+=1
   if counter[0]==3:clock.sleep(20)
   return dict(pause=False)
  result=w.recover('/run',[],Child(),logs.append,6,1,3,sampler=sample,factory=FakeFreezer,clock=clock,sleep=clock.sleep)
  self.assertFalse(result);self.assertTrue(FakeFreezer.last.abort)
  self.assertTrue(any(x.get('reason')=='recovery_deadline_during_sample'for x in logs))
 @unittest.skipUnless(hasattr(os,'pidfd_open') and hasattr(signal,'pidfd_send_signal'),'Linux actual pidfd only')
 def test_actual_cpu_pidfd_stop_resume_and_fail_cleanup(self):
  with tempfile.TemporaryDirectory()as t:
   run=str(Path(t)/'runs/own-phase');Path(run).mkdir(parents=True)
   env=dict(os.environ,BAPO_RUN_ID=run)
   child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)'],env=env)
   f=None
   try:
    logs=[];f=w.OwnedFreezer(run,child.pid,logs.append);f.freeze()
    self.assertIn(child.pid,f.held);self.assertIn(Path('/proc',str(child.pid),'stat').read_text().rsplit(')',1)[1].split()[0],('T','t'))
    f.thaw();time.sleep(.05);self.assertIsNone(child.poll())
    f.freeze();f.thaw(abort=True);child.wait(timeout=5)
    self.assertTrue(any(x.get('event')=='owned_abort_signal'for x in logs))
   finally:
    if f and f.held:f.thaw(abort=True)
    if child.poll()is None:child.kill();child.wait()
 def test_main_unknown_before_start_exit42_no_child(self):
  with tempfile.TemporaryDirectory()as t,patch.dict(os.environ,BAPO_RUN_ID=t),patch.object(sys,'argv',['watch','--run',t,'--','python','noop']),patch.object(w,'observed_sample',return_value=dict(pause=True)),patch.object(w,'recover',return_value=False),patch.object(w.subprocess,'Popen')as start:
   self.assertEqual(w.main(),42);start.assert_not_called()
 def test_main_starts_child_only_after_safe_recovery(self):
  with tempfile.TemporaryDirectory()as t,patch.dict(os.environ,BAPO_RUN_ID=t),patch.object(sys,'argv',['watch','--run',t,'--','python','noop']),patch.object(w,'observed_sample',return_value=dict(pause=True)),patch.object(w,'recover',return_value=True),patch.object(w.subprocess,'Popen')as start:
   start.return_value.poll.return_value=0
   self.assertEqual(w.main(),0);start.assert_called_once();self.assertFalse(start.call_args.kwargs['close_fds'])
 def test_no_pidfd_refuses(self):
  with patch.object(os,'pidfd_open',None,create=True),patch.object(signal,'pidfd_send_signal',None,create=True):
   with self.assertRaises(ValueError):w.OwnedFreezer('/run',101,lambda x:None)
 def proc(self,base,pid,ppid,env,state='S',tick=5):
  p=base/str(pid);p.mkdir();fields=[state,str(ppid)]+['0']*17+[str(tick)];(p/'stat').write_text(f'{pid} (actor) '+' '.join(fields));(p/'environ').write_bytes(b'\0'.join(x.encode()for x in env));(p/'cmdline').write_bytes(b'python\0-m\0verl.trainer.main_ppo\0')
 def test_foreign_descendant_never_certified(self):
  with tempfile.TemporaryDirectory()as t:
   base=Path(t);self.proc(base,101,0,['BAPO_RUN_ID=/run','SLURM_JOB_ID=3231']);self.proc(base,102,101,['BAPO_RUN_ID=/other','SLURM_JOB_ID=3231'])
   with self.assertRaisesRegex(ValueError,'Foreign'):w.freeze_inventory('/run','3231',UUIDS,101,base)
 def test_cpu_child_without_gpu_leases_certified(self):
  with tempfile.TemporaryDirectory()as t:
   base=Path(t);self.proc(base,101,0,['BAPO_RUN_ID=/run','SLURM_JOB_ID=3231']);self.proc(base,102,101,['BAPO_RUN_ID=/run','SLURM_JOB_ID=3231']);self.proc(base,200,0,['BAPO_RUN_ID=/external','SLURM_JOB_ID=3231'])
   self.assertEqual(set(w.freeze_inventory('/run','3231',UUIDS,101,base)),{101,102})
 def test_same_run_foreign_allocation_refused(self):
  with tempfile.TemporaryDirectory()as t:
   base=Path(t);self.proc(base,101,0,['BAPO_RUN_ID=/run','SLURM_JOB_ID=999'])
   with self.assertRaisesRegex(ValueError,'allocation'):w.freeze_inventory('/run','3231',UUIDS,101,base)
 def test_uuid_change_refused(self):
  with tempfile.TemporaryDirectory()as t:
   base=Path(t);self.proc(base,101,0,['BAPO_RUN_ID=/run','SLURM_JOB_ID=3231','SEARCH_R1_GPU_LEASES=[]'])
   with self.assertRaisesRegex(ValueError,'UUID'):w.freeze_inventory('/run','3231',UUIDS,101,base)
 def test_exact_monitor_and_descendant_subtree_excluded(self):
  with tempfile.TemporaryDirectory()as t:
   base=Path(t)
   for pid,parent in ((101,0),(102,101),(103,102),(104,101)):
    self.proc(base,pid,parent,['BAPO_RUN_ID=/run','SLURM_JOB_ID=3231'])
   (base/'102/cmdline').write_bytes(b'bash\0/tools/monitor.sh\0/run\0')
   (base/'103/cmdline').write_bytes(b'sleep\01\0')
   self.assertEqual(set(w.freeze_inventory('/run','3231',UUIDS,101,base)),{101,104})
 def test_unrelated_monitor_name_is_not_excluded(self):
  with tempfile.TemporaryDirectory()as t:
   base=Path(t);self.proc(base,101,0,['BAPO_RUN_ID=/run','SLURM_JOB_ID=3231']);self.proc(base,102,101,['BAPO_RUN_ID=/run','SLURM_JOB_ID=3231'])
   (base/'102/cmdline').write_bytes(b'bash\0/other/monitor.sh\0/run\0')
   self.assertEqual(set(w.freeze_inventory('/run','3231',UUIDS,101,base)),{101,102})
 def test_unclassified_reparented_same_run_refused(self):
  with tempfile.TemporaryDirectory()as t:
   base=Path(t);self.proc(base,101,0,['BAPO_RUN_ID=/run','SLURM_JOB_ID=3231']);self.proc(base,102,0,['BAPO_RUN_ID=/run','SLURM_JOB_ID=3231'])
   (base/'102/cmdline').write_bytes(b'python\0unknown.py\0')
   with self.assertRaisesRegex(ValueError,'role'):w.freeze_inventory('/run','3231',UUIDS,101,base)
 def complete_row(self):
  return dict(status='ok',api='nvmlDeviceGetProcessUtilization',complete_query=True,last_seen_timestamp_us=0,mig_mode=0,returned_sample_count=1,query_start_us=900000,query_end_us=1000000,samples=[dict(pid=1,timestamp_us=999999,sm_percent=80,mem_percent=1,enc_percent=0,dec_percent=0)])
 def test_api_proven_fresh_complete_period_absence(self):
  a={'0':dict(uuid='GPU-0',total_mib=100,whole_gpu_sm_percent=98)};row=self.complete_row()
  result=w.assess_with_bounds(a,{('0',2):'-'},{('0',2):1},lambda pid:pid==1,{'GPU-0':row},900000)
  self.assertFalse(result['pause']);self.assertEqual(result['external_processes'][0]['sm_evidence'],'nvml_complete_observed_period_absence');self.assertEqual(result['external_processes'][0]['nvml_observed_period_us'],999999)
 def test_old_foreign_record_not_filtered_into_absence(self):
  row=self.complete_row();row['samples'].append(dict(row['samples'][0],pid=2,timestamp_us=1));row['returned_sample_count']=2
  self.assertIsNone(w.nvml_absence_period({},row,2,900000))
 def test_all_sm_zero_whole98_remains_unknown(self):
  row=self.complete_row();row['samples'][0]['sm_percent']=0
  self.assertIsNone(w.nvml_absence_period({},row,2,900000))
 def test_missing_full_fields_count_mig_or_ambiguous_epochs_refuse(self):
  for mutate in [lambda r:r.pop('complete_query'),lambda r:r.update(returned_sample_count=2),lambda r:r.update(mig_mode=1),lambda r:r['samples'][0].pop('mem_percent'),lambda r:r['samples'].append(dict(r['samples'][0],pid=3,timestamp_us=999990))]:
   row=self.complete_row();mutate(row)
   if len(row['samples'])==2:row['returned_sample_count']=2
   self.assertIsNone(w.nvml_absence_period({},row,2,900000))
 def test_nvml_corrupt_boolean_zero_duplicate_pid_and_nonpositive_times_refuse(self):
  for mutate in [lambda r:r.update(last_seen_timestamp_us=False),lambda r:r.update(query_start_us=0),lambda r:r['samples'][0].update(timestamp_us=0),lambda r:r['samples'].append(dict(r['samples'][0]))]:
   row=self.complete_row();mutate(row);row['returned_sample_count']=len(row['samples'])
   self.assertIsNone(w.nvml_absence_period({},row,2,900000))
 def test_owned_nvml_only_is_not_external_zero(self):
  a={'0':dict(uuid='GPU-0',total_mib=100,whole_gpu_sm_percent=98)};n={'GPU-0':dict(status='ok',query_end_us=1000000,samples=[dict(pid=1,timestamp_us=999999,sm_percent=80)])}
  with self.assertRaisesRegex(ValueError,'unknown'):w.assess_with_bounds(a,{('0',2):'-'},{('0',2):1},lambda pid:pid==1,n,900000)
 def test_whole_under15_is_upper_bound_not_zero(self):
  a={'0':dict(uuid='GPU-0',total_mib=100,whole_gpu_sm_percent=12)}
  result=w.assess_with_bounds(a,{('0',2):'-'},{('0',2):1},lambda pid:False,{},0)
  self.assertFalse(result['pause']);self.assertIsNone(result['external_totals']['0']['external_sm_percent']);self.assertEqual(result['external_totals']['0']['external_sm_upper_bound_percent'],12)
if __name__=='__main__':unittest.main(verbosity=2)
