"""Synthetic /proc+real local executable CPU tests; no GPU/production approval."""
import copy,hashlib,importlib.util,json,os,signal,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('ray_title_watch',HERE/'resource_watch.py');w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
U=sorted('GPU-'+str(i)for i in range(8))
class Tests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve();self.run=str(self.root/'runs/fixture900')
  for p in ('logs','tools','env','runs/fixture900','proc'): (self.root/p).mkdir(parents=True,exist_ok=True)
  self.proc=self.root/'proc';self.exe=w.executable_identity(sys.executable);self.source='a'*40
  (self.root/'source-expected.txt').write_text(self.source)
  raw=json.dumps({'source_commit':self.source,'root':str(self.root),'sys_prefix':str(self.root/'env'),'executable':self.exe}).encode();self.anchorpath=self.root/'logs/runtime-executable.json';self.anchorpath.write_bytes(raw)
  binding={'path':str(self.anchorpath),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
  m=json.dumps({'source_commit':self.source,'files':{'resource_watch.py':hashlib.sha256((HERE/'resource_watch.py').read_bytes()).hexdigest()}}).encode();(self.root/'logs/tools-manifest.json').write_bytes(m)
  self.approval={'status':'PASS_INDEPENDENT_E1_LAUNCH_REVIEW','source_commit':self.source,'root':str(self.root),'runtime_executable':binding,'tools_manifest_sha256':hashlib.sha256(m).hexdigest()}
  (self.root/'logs/launch-approval.json').write_text(json.dumps(self.approval))
  self.anchor=w.load_runtime_anchor(self.run)
  original=w.executable_identity
  self.addCleanup(patch.stopall)
  patch.object(w.sys,'prefix',str(self.root/'env')).start()
  patch.object(w,'executable_identity',side_effect=lambda path:self.exe if str(path)=='/proc/self/exe'and not Path(path).exists()else original(path)).start()
  patch.dict(os.environ,SLURM_JOB_ID='900',SEARCH_R1_GPU_LEASES=json.dumps([{'uuid':u}for u in U])).start()
 def process(self,pid,parent,command,state='S'):
  p=self.proc/str(pid);p.mkdir();(p/'stat').write_text(str(pid)+' (actor) '+' '.join([state,str(parent)]+['0']*17+[str(100+pid)]))
  (p/'environ').write_bytes(('BAPO_RUN_ID='+self.run+'\0SLURM_JOB_ID=900\0SEARCH_R1_GPU_LEASES='+json.dumps([{'uuid':u}for u in U])).encode())
  (p/'cmdline').write_bytes(('\0'.join(command)).encode());(p/'exe').symlink_to(Path(sys.executable).resolve());return p
 def snapshot(self,title='ray::WorkerDict'):
  self.process(101,0,['python','phase.py','']);self.process(102,101,[title,'',''])
  return w.freeze_inventory(self.run,'900',set(U),101,self.proc)
 def test_exact_finite_policy_and_nonempty_tail_rejected(self):
  p=self.snapshot()[102]
  for title in w.RAY_APPROVED_TITLES:
   self.assertEqual(w.certificate_identity(dict(p,command=[title]+['']*19),self.anchor),w.certificate_identity(p,self.anchor))
  for command in (['ray::WorkerDict.evil'],['ray::WorkerGroup'],['ray::main_task','foreign.py'],['ray::WorkerDict.execute_func_rank_zero']):
   with self.assertRaises(ValueError):w.certificate_identity(dict(p,command=command),self.anchor)
 def test_real_reader_title_and_padding_transition(self):
  before=self.snapshot()[102];(self.proc/'102/cmdline').write_bytes(b'ray::WorkerDict.actor_rollout_compute_log_prob'+b'\0'*105)
  after=w.local_freeze_recheck(before,{},self.proc,anchor=self.anchor)
  self.assertEqual(w.certificate_identity(before,self.anchor),w.certificate_identity(after,self.anchor));self.assertNotEqual(before['command'],after['command'])
 def test_nonray_complete_argv_changes_still_refused(self):
  p=self.snapshot()[101];(self.proc/'101/cmdline').write_bytes(b'python\0phase.py\0\0')
  with self.assertRaises(w.IdentityMismatch)as err:w.local_freeze_recheck(p,{},self.proc)
  self.assertIn('command_identity',err.exception.diagnostic['immutable_fields'])
 def test_exe_change_on_same_pid_epoch_and_leases_refused(self):
  p=self.snapshot()[102];(self.proc/'102/exe').unlink();(self.proc/'102/exe').symlink_to('/bin/sh')
  with self.assertRaises(w.IdentityMismatch)as err:w.local_freeze_recheck(p,{},self.proc,anchor=self.anchor)
  self.assertIn('executable',err.exception.diagnostic['immutable_fields'])
 def test_immutable_fields_and_bool_types_cannot_hide_as_title_transition(self):
  p=self.snapshot()[102]
  for key,value in [('pid',False),('uid',False),('start_ticks',p['start_ticks']+1),('ppid',1),('parent_start_ticks',999),('run','/foreign'),('slurm_job_id','901'),('role','trainer'),('allocated_uuid_set',U[:-1]),('gpu_leases_inherited',False)]:
   changed=dict(p,**{key:value})
   try:same=w.certificate_identity(changed,self.anchor)==w.certificate_identity(p,self.anchor)
   except (ValueError,KeyError,TypeError):same=False
   self.assertFalse(same,key)
 def test_beforeafter_exact_diagnostics_without_environment_dump(self):
  p=self.snapshot()[102];(self.proc/'102/environ').write_bytes(('BAPO_RUN_ID=/foreign\0SLURM_JOB_ID=900\0SECRET=must-not-log').encode())
  with self.assertRaises(w.IdentityReadMismatch)as err:w.local_freeze_recheck(p,{},self.proc,anchor=self.anchor)
  self.assertEqual(err.exception.diagnostic['immutable_fields']['run'],{'before':self.run,'after':'/foreign'});self.assertNotIn('must-not-log',json.dumps(err.exception.diagnostic))
 def test_runtime_anchor_hash_source_path_and_exe_forgery(self):
  original=self.anchorpath.read_bytes();self.anchorpath.write_bytes(original+b' ')
  with self.assertRaisesRegex(ValueError,'bytes/hash'):w.load_runtime_anchor(self.run)
  self.anchorpath.write_bytes(original);r=json.loads(original);r['source_commit']='b'*40;raw=json.dumps(r).encode();self.anchorpath.write_bytes(raw)
  self.approval['runtime_executable'].update(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw));(self.root/'logs/launch-approval.json').write_text(json.dumps(self.approval))
  with self.assertRaisesRegex(ValueError,'source/environment'):w.load_runtime_anchor(self.run)
 def test_shared_projection_stop_stable_and_original_pidfd_resume(self):
  rows=self.snapshot();events=[];sent=[];closed=[];ticks=[0.]
  def clock():return ticks[0]
  def sleep(n):ticks[0]+=n
  def inventory(*args):return copy.deepcopy(rows)
  def recheck(proof,held):
   current=copy.deepcopy(rows[proof['pid']])
   if current['role']=='ray_worker':current['command']=['ray::WorkerDict.actor_rollout_generate_sequences']+['']*16;rows[current['pid']]['command']=current['command']
   return current
  def send(pid,sig,*args):sent.append((pid,sig));rows[pid]['state']='T'if sig==signal.SIGSTOP else'S'
  f=w.OwnedFreezer(self.run,101,events.append,clock,sleep,inventory,lambda pid,flags:pid,send,closed.append,recheck_fn=recheck)
  f.freeze();stable=next(e for e in events if e['event']=='owned_freeze_stable');self.assertEqual(set(stable['before_certificates']),{'101','102'});self.assertEqual(len([e for e in events if e['event']=='owned_title_transition']),1)
  self.assertEqual(f.held[102]['proof']['command'][0],'ray::WorkerDict.actor_rollout_generate_sequences');f.thaw();self.assertEqual(len(sent),4);self.assertEqual(set(closed),{101,102})
 def test_unsafe_title_after_pidfd_logs_diff_and_never_signals_target(self):
  rows=self.snapshot();events=[];sent=[];closed=[];ticks=[0.]
  def recheck(p,h):return dict(p,command=['ray::WorkerDict.evil'])if p['pid']==102 else dict(p)
  f=w.OwnedFreezer(self.run,101,events.append,lambda:ticks[0],lambda n:ticks.__setitem__(0,ticks[0]+n),lambda *a:copy.deepcopy(rows),lambda pid,flags:pid,lambda pid,sig,*a:sent.append((pid,sig)),closed.append,recheck_fn=recheck)
  with self.assertRaises(w.IdentityMismatch):f.freeze()
  self.assertNotIn((102,signal.SIGSTOP),sent);self.assertIn(102,closed);self.assertTrue(any(e['event']=='owned_identity_rejected'and e['pid']==102 for e in events));f.thaw(abort=True)
 def replay_rows(self,mutator=None):
  spec=importlib.util.spec_from_file_location('base_journal_fixture',HERE/'test_guard_journal.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
  base.RUN=self.run;base.g.NEW_ROOT=self.root
  rows=base.recovery()
  for row in rows:
   if row.get('event')=='owned_freeze_signal'and row['pid']==102:
    row['certificate'].update(command=['ray::WorkerDict'],executable=self.exe,runtime_anchor_sha256=self.anchor['sha256'])
   if row.get('event')=='owned_freeze_stable':
    for key in ('before_certificates','after_certificates'):
     row[key]['102'].update(command=['ray::WorkerDict.actor_rollout_compute_log_prob']+['']*8,executable=self.exe,runtime_anchor_sha256=self.anchor['sha256'])
  if mutator:mutator(rows)
  path=self.root/'watch.jsonl';path.write_text(''.join(json.dumps(row)+'\n'for row in rows))
  return base.g.replay(path,base.JOB,self.run,base.SOURCE,HERE/'resource_watch.py')
 def test_replay_uses_original_anchor_not_local_sys_executable(self):
  with patch.object(sys,'executable','/no/local/interpreter'):
   self.assertEqual(self.replay_rows()['status'],'PASS_ACTUAL_GUARD_JOURNAL_REPLAY')
 def test_replay_raw_snapshots_rebound_sha_cannot_hide_unknown_title_or_exe(self):
  for field,value in [('command',['ray::WorkerDict.evil']),('executable',dict(self.exe,inode=self.exe['inode']+1)),('runtime_anchor_sha256','0'*64),('state','S')]:
   def mutate(rows):rows[4]['after_certificates']['102'][field]=value
   with self.subTest(field=field),self.assertRaises(ValueError):self.replay_rows(mutate)
 def test_replay_requires_both_actual_stopped_snapshots(self):
  with self.assertRaises(ValueError):self.replay_rows(lambda rows:rows[4].pop('before_certificates'))
 def test_replay_transition_must_match_following_stop_certificate(self):
  def mutate(rows):
   after=copy.deepcopy(rows[3]['certificate']);before=copy.deepcopy(after);before['command']=['ray::WorkerDict.actor_rollout_generate_sequences'];rows.insert(3,dict(run=self.run,slurm_job_id='900',phase_pid=101,time=1.,event='owned_title_transition',pid=102,before_certificate=before,after_certificate=after))
  self.assertEqual(self.replay_rows(mutate)['status'],'PASS_ACTUAL_GUARD_JOURNAL_REPLAY')
  def bad(rows):
   mutate(rows);rows[4]['certificate']['command']=['ray::WorkerDict.actor_rollout_compute_log_prob']
  with self.assertRaises(ValueError):self.replay_rows(bad)
if __name__=='__main__':unittest.main()
