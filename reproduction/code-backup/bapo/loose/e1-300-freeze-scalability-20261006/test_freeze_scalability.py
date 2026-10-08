"""Scalability/identity tests; actual pidfd cases execute only on target Linux."""
import importlib.util,json,os,signal,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('scale_watch',Path(__file__).with_name('resource_watch.py'));w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
UUIDS={'GPU-'+str(i)for i in range(8)}
class Clock:
 def __init__(self):self.t=0.
 def __call__(self):return self.t
 def sleep(self,n):self.t+=n
class Tests(unittest.TestCase):
 def setUp(self):
  self.env=patch.dict(os.environ,SLURM_JOB_ID='TEST301',SEARCH_R1_GPU_LEASES=json.dumps([{'uuid':u}for u in UUIDS]));self.env.start()
 def tearDown(self):self.env.stop()
 def rows(self,count=301):
  return {p:dict(pid=p,ppid=0 if p==101 else 101,start_ticks=p+500,parent_start_ticks=None if p==101 else 601,uid=os.getuid(),run='/run',slurm_job_id='TEST301',state='S',role='phase'if p==101 else'phase_descendant',command=['python','worker.py',''],allocated_uuid_set=sorted(UUIDS),gpu_leases_inherited=False,inherited_gpu_lease_uuids=[])for p in range(101,101+count)}
 def test_301_linear_inventory_and_two_stable_snapshots(self):
  clock=Clock();state=self.rows();counts=dict(inventory=0,local=0);logs=[];signals=[]
  def inventory(*a):counts['inventory']+=1;clock.sleep(.15);return {p:dict(v)for p,v in state.items()}
  def local(proof,held):
   counts['local']+=1;clock.sleep(.0001)
   if proof['role']=='phase_descendant':self.assertIn(101,held);self.assertEqual(state[101]['state'],'T')
   return dict(state[proof['pid']])
  def send(pid,sig,*a):signals.append((pid,sig));state[pid]['state']='T'if sig==signal.SIGSTOP else'S'
  f=w.OwnedFreezer('/run',101,logs.append,clock,clock.sleep,inventory,lambda p,x:p,send,lambda p:None,recheck_fn=local)
  f.freeze();self.assertEqual(counts,dict(inventory=6,local=301));self.assertLess(clock.t,5);self.assertEqual(len(f.held),301)
  self.assertEqual(sum(x==signal.SIGSTOP for _,x in signals),301);self.assertEqual(sum(x.get('event')=='owned_freeze_stable'for x in logs),1)
  f.thaw();self.assertEqual(sum(x==signal.SIGCONT for _,x in signals),301)
 def test_long_chain_order_is_linear_and_cycle_rejects(self):
  rows=self.rows()
  for p in range(102,402):rows[p]['ppid']=p-1
  self.assertEqual(w.freeze_parent_order(rows),list(range(101,402)))
  rows[101]['ppid']=401
  with self.assertRaisesRegex(ValueError,'cycle'):w.freeze_parent_order(rows)
 def proc(self,base,pid,ppid,state='S',tick=5,env=None):
  p=base/str(pid);p.mkdir();fields=[state,str(ppid)]+['0']*17+[str(tick)];(p/'stat').write_text(f'{pid} (actor) '+' '.join(fields));(p/'environ').write_bytes(b'\0'.join(x.encode()for x in (env or ['BAPO_RUN_ID=/run','SLURM_JOB_ID=TEST301'])));(p/'cmdline').write_bytes(b'python\0worker.py\0')
 def test_local_exact_identity_and_parent_epoch_rejects(self):
  for case in ['foreign_run','job','uuid','command','child_epoch','parent_epoch','parent_running','uid','duplicate','reparent']:
   with self.subTest(case=case),tempfile.TemporaryDirectory()as t:
    base=Path(t);self.proc(base,101,0,state='T',tick=601);self.proc(base,102,101,tick=602)
    proof=self.rows(2)[102];held={101:dict(proof=self.rows(2)[101],fd=101)}
    self.assertEqual(w.local_freeze_recheck(proof,held,base)['pid'],102)
    child=base/'102'
    if case=='foreign_run':(child/'environ').write_bytes(b'BAPO_RUN_ID=/foreign\0SLURM_JOB_ID=TEST301')
    elif case=='job':(child/'environ').write_bytes(b'BAPO_RUN_ID=/run\0SLURM_JOB_ID=OTHER')
    elif case=='uuid':(child/'environ').write_bytes(b'BAPO_RUN_ID=/run\0SLURM_JOB_ID=TEST301\0SEARCH_R1_GPU_LEASES=[]')
    elif case=='command':(child/'cmdline').write_bytes(b'python\0changed.py\0')
    elif case in ('child_epoch','reparent'):
     text=(child/'stat').read_text();parts=text.rsplit(')',1)[1].split();parts[19]='999'if case=='child_epoch'else'602';parts[1]='999'if case=='reparent'else'101';(child/'stat').write_text('102 (actor) '+' '.join(parts))
    elif case=='parent_epoch':(base/'101/stat').write_text('101 (actor) '+' '.join(['T','0']+['0']*17+['999']))
    elif case=='parent_running':(base/'101/stat').write_text('101 (actor) '+' '.join(['S','0']+['0']*17+['601']))
    elif case=='uid':proof=dict(proof,uid=proof['uid']+1)
    else:(child/'environ').write_bytes(b'BAPO_RUN_ID=/run\0BAPO_RUN_ID=/run\0SLURM_JOB_ID=TEST301')
    with self.assertRaises((ValueError,FileNotFoundError)):w.local_freeze_recheck(proof,held,base)
 def test_running_before_stopped_after_cannot_fake_stability(self):
  state=self.rows(1);clock=Clock();count=[0];events=[]
  def inventory(*a):
   count[0]+=1
   return {101:dict(state[101],state='S'if count[0]%2 else'T')}
  f=w.OwnedFreezer('/run',101,events.append,clock,clock.sleep,inventory,lambda p,x:p,lambda *a:None,lambda p:None,recheck_fn=lambda p,h:dict(state[101]))
  with self.assertRaises(ValueError):f.freeze(timeout=.3)
  self.assertFalse(any(x.get('event')=='owned_freeze_stable'for x in events));f.thaw()
 def test_monitor_argument_substring_does_not_exclude_cpu_worker(self):
  with tempfile.TemporaryDirectory()as t:
   base=Path(t);self.proc(base,101,0,tick=601);self.proc(base,102,101,tick=602)
   (base/'102/cmdline').write_bytes(b'python\0worker.py\0/tools/monitor.sh\0/run\0')
   self.assertEqual(set(w.freeze_inventory('/run','TEST301',UUIDS,101,base)),{101,102})
 def test_slow_parent_stop_wait_is_deadline_bounded(self):
  with tempfile.TemporaryDirectory()as t:
   base=Path(t);self.proc(base,101,0,state='D',tick=601);self.proc(base,102,101,tick=602);clock=Clock()
   with self.assertRaisesRegex(ValueError,'deadline'):w.local_freeze_recheck(self.rows(2)[102],{101:dict(proof=self.rows(2)[101])},base,deadline=.02,clock=clock,sleep=clock.sleep)
   self.assertEqual(clock.t,.02)
 def test_deadline_before_each_signal_prevents_overshoot(self):
  state=self.rows();clock=Clock();signals=[]
  def inventory(*a):return {p:dict(v)for p,v in state.items()}
  def local(proof,held):clock.sleep(.02);return dict(state[proof['pid']])
  def send(pid,sig,*a):signals.append((pid,sig));state[pid]['state']='T'if sig==signal.SIGSTOP else'S'
  f=w.OwnedFreezer('/run',101,lambda x:None,clock,clock.sleep,inventory,lambda p,x:p,send,lambda p:None,recheck_fn=local)
  with self.assertRaisesRegex(ValueError,'deadline'):f.freeze(timeout=.05)
  f.thaw();self.assertLess(sum(s==signal.SIGSTOP for _,s in signals),301)
 @unittest.skipUnless(hasattr(os,'pidfd_open')and hasattr(signal,'pidfd_send_signal'),'Linux actual pidfd only')
 def test_actual_linux_301_owned_cpu_descendants_and_foreign_sentinel(self):
  with tempfile.TemporaryDirectory()as t:
   run=str(Path(t)/'runs/owned');Path(run).mkdir(parents=True)
   env=dict(os.environ,BAPO_RUN_ID=run);env.pop('SEARCH_R1_GPU_LEASES',None)
   foreign_done=str(Path(t)/'sentinel-done')
   foreign=subprocess.Popen([sys.executable,'-c','import sys,time,pathlib; p=pathlib.Path(sys.argv[1]); end=time.monotonic()+90\nwhile not p.exists() and time.monotonic()<end:time.sleep(.01)',foreign_done],env=dict(env,BAPO_RUN_ID=run+'-foreign'))
   code='''import os,time,json
children=[]
for i in range(300):
 p=os.fork()
 if p==0:time.sleep(60);os._exit(0)
 children.append(p)
print(json.dumps(children),flush=True)
time.sleep(60)
'''
   phase=subprocess.Popen([sys.executable,'-c',code],env=env,stdout=subprocess.PIPE,text=True);f=None;logs=[]
   try:
    children=json.loads(phase.stdout.readline());self.assertEqual(len(children),300);owned={phase.pid,*children}
    f=w.OwnedFreezer(run,phase.pid,logs.append);start=time.monotonic();f.freeze();elapsed=time.monotonic()-start
    self.assertEqual(set(f.held),owned);self.assertNotIn(foreign.pid,f.held)
    self.assertTrue(all(Path('/proc',str(p),'stat').read_text().rsplit(')',1)[1].split()[0]in ('T','t')for p in owned))
    self.assertNotIn(Path('/proc',str(foreign.pid),'stat').read_text().rsplit(')',1)[1].split()[0],('T','t'))
    f.thaw();self.assertIsNone(phase.poll());self.assertIsNone(foreign.poll())
    f.freeze();f.thaw(abort=True);phase.wait(timeout=5)
    self.assertNotIn(foreign.pid,{x['pid']for x in logs if 'pid'in x})
    print('ACTUAL_LINUX_301_CPU_PIDFD_SCALABILITY_JSON='+json.dumps(dict(owned_count=301,freeze_seconds=elapsed,foreign_signals=0,stopped_epochs=301,thaw=True,abort_cleanup=True)),flush=True)
   finally:
    if f and f.held:f.thaw(abort=True)
    if phase.poll()is None:phase.kill();phase.wait()
    Path(foreign_done).touch();foreign.wait(timeout=5)
 @unittest.skipUnless(hasattr(os,'pidfd_open')and hasattr(signal,'pidfd_send_signal'),'Linux actual pidfd only')
 def test_actual_301_tree_exec_foreign_after_pidfd_gets_no_signal(self):
  with tempfile.TemporaryDirectory()as t:
   run=str(Path(t)/'runs/owned');Path(run).mkdir(parents=True);trigger=str(Path(t)/'exec-trigger')
   env=dict(os.environ,BAPO_RUN_ID=run);env.pop('SEARCH_R1_GPU_LEASES',None)
   code="""import os,time,json,sys,pathlib
children=[]
for i in range(300):
 p=os.fork()
 if p==0:
  if i==299:
   end=time.monotonic()+60
   while not pathlib.Path(sys.argv[1]).exists() and time.monotonic()<end:time.sleep(.005)
   changed=dict(os.environ,BAPO_RUN_ID=os.environ['BAPO_RUN_ID']+'-foreign')
   os.execve(sys.executable,[sys.executable,'-c','import time;time.sleep(.5)'],changed)
  time.sleep(60);os._exit(0)
 children.append(p)
print(json.dumps(children),flush=True)
time.sleep(60)
"""
   phase=subprocess.Popen([sys.executable,'-c',code,trigger],env=env,stdout=subprocess.PIPE,text=True);f=None;logs=[]
   try:
    children=json.loads(phase.stdout.readline());target=children[-1];opened=[];closed=[]
    def opener(pid,flags):
     fd=os.pidfd_open(pid,flags);opened.append(fd)
     if pid==target:
      Path(trigger).touch();deadline=time.monotonic()+3
      while time.monotonic()<deadline:
       raw=Path('/proc',str(pid),'environ').read_bytes()
       if ('BAPO_RUN_ID='+run+'-foreign').encode()in raw.split(b'\0'):break
       time.sleep(.005)
      else:raise AssertionError('Actual child exec did not publish changed environment')
     return fd
    def closer(fd):closed.append(fd);os.close(fd)
    f=w.OwnedFreezer(run,phase.pid,logs.append,opener=opener,closer=closer)
    with self.assertRaisesRegex(ValueError,'identity changed'):f.freeze()
    self.assertNotIn(target,f.held);self.assertNotIn(target,{x['pid']for x in logs if 'pid'in x})
    f.thaw(abort=True);phase.wait(timeout=5)
    self.assertEqual(len(opened),len(closed));self.assertNotIn(target,{x['pid']for x in logs if 'pid'in x})
    print('ACTUAL_LINUX_301_CPU_EXEC_RACE_JSON='+json.dumps(dict(initial_owned_count=301,foreign_target_pid=target,foreign_signals=0,all_pidfds_closed=True,fail_closed=True)),flush=True)
   finally:
    if f and f.held:f.thaw(abort=True)
    if phase.poll()is None:phase.kill();phase.wait()
    # Changed foreign epoch exits itself; never send a signal to it.
if __name__=='__main__':unittest.main(verbosity=2)
