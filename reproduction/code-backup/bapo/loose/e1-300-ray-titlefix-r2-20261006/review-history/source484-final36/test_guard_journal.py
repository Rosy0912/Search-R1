"""True assessor + generated raw telemetry fixtures, no process/GPU calls."""
import copy,importlib.util,json,tempfile,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('journal',HERE/'guard_journal.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
spec=importlib.util.spec_from_file_location('watch',HERE/'resource_watch.py');w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
RUN='/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-titlefix-r2-20261006/runs/fixture-900';JOB=900;SOURCE='a'*40;UUIDS=['GPU-'+str(i)for i in range(8)]
def raw(unknown=False):
 start=1000000;end=1200000
 inventory='\n'.join(f'{i}, {UUIDS[i]}, 1000, {98 if unknown and i==0 else 0}'for i in range(8))+'\n'
 compute='\n'.join(f'{UUIDS[i]}, {500+i}, 1'for i in range(8))+'\n'
 pmon='# gpu pid type sm mem enc dec jpg ofa command\n'+'\n'.join(f'{i} {500+i} C {"-"if unknown and i==0 else "0"} 0 - - - - python'for i in range(8))+'\n'
 a=w.inventory(inventory,[str(i)for i in range(8)]);nv={}
 for i in range(8):
  nv[UUIDS[i]]=dict(status='ok',whole_gpu_sm_percent=98 if unknown and i==0 else 0,whole_gpu_sm_query_us=end-5,query_start_us=end-20,query_end_us=end,api='nvmlDeviceGetProcessUtilization',complete_query=True,last_seen_timestamp_us=0,mig_mode=0,returned_sample_count=1,samples=[dict(pid=102,timestamp_us=end-10,sm_percent=0,mem_percent=0,enc_percent=0,dec_percent=0)])
  a[str(i)].update(whole_gpu_sm_source='nvml_query',whole_gpu_sm_query_us=end-5)
 own={str(p):dict(start_ticks=10+p,uid=0,run=RUN,slurm_job_id=str(JOB),lease_uuids=UUIDS)for p in (101,102)}
 ticks={str(500+i):dict(before=900+i,after=900+i)for i in range(8)};ticks['102']=dict(before=112,after=112)
 return dict(raw_inventory=inventory,raw_compute=compute,raw_pmon=pmon,raw_nvml=nv,allocated_gpus=a,window_start_us=start,window_end_us=end,owned_before_query=own,pid_start_ticks=ticks)
def sample(unknown=False,event=None):
 d=raw(unknown);row=dict(run=RUN,slurm_job_id=str(JOB),phase_pid=101,time=1.)
 if event:row['event']=event;row['elapsed_seconds']=1
 if unknown:
  d['error']='ValueError: External SM unknown without current NVML coverage GPU=0 PID=500';row.update(pause=True,reason='monitor_unknown_fail_closed',diagnostic=dict(samples=[d]),error='SamplingError: '+d['error'])
 else:
  a=d['allocated_gpus'];r=w.assess_with_bounds(a,w.parse_pmon(d['raw_pmon'],a),w.parse_memory(d['raw_compute'],a),lambda p:p in (101,102),d['raw_nvml'],d['window_start_us'])
  row.update(r);row.update(d);row['reason']='within_threshold'
 return row
def event(name,**kw):
 if name=='owned_freeze_stable':
  kw.update(before_certificates={str(p):dict(certificate(p),state='T')for p in kw['pids']},after_certificates={str(p):dict(certificate(p),state='T')for p in kw['pids']})
 return dict(run=RUN,slurm_job_id=str(JOB),phase_pid=101,time=1.,event=name,**kw)
def certificate(p,role=None,lease=True,ppid=101):
 role=role or ('phase'if p==101 else'ray_worker')
 command=['/bin/bash','phase.sh']if role=='phase'else(['raylet']if role=='ray_worker'else['python','cpu_child.py'])
 return dict(pid=p,ppid=1 if p==101 else ppid,parent_start_ticks=11 if p==101 else 10+ppid,state='S',start_ticks=10+p,uid=0,run=RUN,slurm_job_id=str(JOB),role=role,executable={'resolved_path':'/fixture/python','device':0,'inode':1},command=command,allocated_uuid_set=sorted(UUIDS),gpu_leases_inherited=lease,inherited_gpu_lease_uuids=sorted(UUIDS)if lease else[])
def recovery():
 rows=[sample(True),event('recovery_started',grace_seconds=60)]
 rows += [event('owned_freeze_signal',pid=p,start_ticks=10+p,uid=0,role='phase'if p==101 else'ray_worker',lease_uuids=UUIDS,certificate=certificate(p))for p in (101,102)]
 stable=event('owned_freeze_stable',pids=[101,102],epochs={'101':111,'102':112});rows.append(stable)
 rows.append(sample(True,'recovery_sample'))
 for _ in range(3):rows.extend([copy.deepcopy(stable),sample(False,'recovery_sample')])
 rows += [event('recovery_validated',consecutive_safe_samples=3),event('recovery_resume',pause=False)]
 rows += [event('owned_resume_signal',pid=p,start_ticks=10+p,signal=18)for p in (101,102)]
 rows += [event('recovery_finished',elapsed_seconds=6,reserved_gpu_hours=6*8/3600),sample()]
 return rows
class Tests(unittest.TestCase):
 def run_rows(self,rows):
  with tempfile.TemporaryDirectory()as t:
   p=Path(t).resolve()/'watch.jsonl';p.write_text(''.join(json.dumps(r)+'\n'for r in rows));return g.replay(p,JOB,RUN,SOURCE,HERE/'resource_watch.py')
 def test_normal_real_assessor(self):
  r=self.run_rows([sample(),sample()]);self.assertEqual(r['samples'],2);self.assertEqual(r['safe_resume_events'],0)
 def test_frozen_unknown_then_actual_safe3_and_all_cont(self):
  r=self.run_rows(recovery());self.assertEqual(r['safe_resume_events'],1);self.assertEqual(r['recovery_samples'],4);self.assertEqual(r['freeze_events'],2);self.assertEqual(r['abort_events'],0)
 def test_tampered_results_not_trusted(self):
  for key,value in [('pause',True),('external_totals',{}),('exceeded',{'fake':1}),('raw_inventory','bad')]:
   with self.subTest(key=key):
    x=sample();x[key]=value
    with self.assertRaises((g.JournalError,ValueError)):self.run_rows([x])
 def test_foreign_or_reused_freeze(self):
  for key,val in [('pid',500),('start_ticks',999),('uid',1),('lease_uuids',['GPU-foreign']),('role','monitor')]:
   x=recovery();x[2][key]=val
   with self.subTest(key=key),self.assertRaises(g.JournalError):self.run_rows(x)
 def test_no_stable_or_no_phase_freeze(self):
  x=recovery();x=[r for r in x if r.get('event')!='owned_freeze_stable']
  with self.assertRaises(g.JournalError):self.run_rows(x)
 def test_resuming_requires_safe3(self):
  x=recovery();x[9]=sample(True,'recovery_sample')
  with self.assertRaises(g.JournalError):self.run_rows(x)
 def test_missing_cont_unfinished_or_abort(self):
  cases=[recovery()[:-3],recovery()+[event('owned_abort_signal',pid=102,signal=9,start_ticks=112)],[sample(True)],recovery()+[event('invented')]]
  for x in cases:
   with self.assertRaises(g.JournalError):self.run_rows(x)
 def test_gpu_epoch_and_uuid_change(self):
  for mutator in [lambda x:x['pid_start_ticks']['500'].update(after=901),lambda x:x['owned_before_query']['101'].update(run='/foreign'),lambda x:x['allocated_gpus']['0'].update(uuid='GPU-other')]:
   x=sample();mutator(x)
   with self.assertRaises(g.JournalError):self.run_rows([x])
 def test_wrong_job_run_time_and_late_observation(self):
  for key,value in [('run','/foreign'),('slurm_job_id','999'),('time',float('nan'))]:
   x=sample();x[key]=value
   with self.assertRaises(g.JournalError):self.run_rows([x])
  x=recovery();x[5]['elapsed_seconds']=60
  with self.assertRaises(g.JournalError):self.run_rows(x)
 def test_raw_unknown_cannot_hide_safe(self):
  x=sample(True);x['diagnostic']['samples']=[raw(False)];x['diagnostic']['samples'][0]['error']='pretend'
  with self.assertRaises(g.JournalError):self.run_rows([x])
 def test_never_invokes_process_or_gpu_functions(self):
  from unittest.mock import patch
  with patch('subprocess.run',side_effect=AssertionError('no process')),patch('subprocess.Popen',side_effect=AssertionError('no process')):self.run_rows(recovery())
 def test_bad_watcher_and_symlink(self):
  with tempfile.TemporaryDirectory()as t:
   p=Path(t).resolve()/'watch.jsonl';p.write_text(json.dumps(sample())+'\n');watch=Path(t).resolve()/'bad.py';watch.write_text('raise AssertionError("must not execute")')
   with self.assertRaises(g.JournalError):g.replay(p,JOB,RUN,SOURCE,watch)
   link=Path(t).resolve()/'link';link.symlink_to(p)
   with self.assertRaises(g.JournalError):g.replay(link,JOB,RUN,SOURCE,HERE/'resource_watch.py')
 def prelaunch(self):
  x=[sample(True),event('recovery_started',grace_seconds=60)]+[sample(False,'recovery_sample')for _ in range(3)]+[event('recovery_validated',consecutive_safe_samples=3),event('recovery_finished',elapsed_seconds=6,reserved_gpu_hours=6*8/3600)]
  for row in x:row['phase_pid']=None
  return x
 def test_prelaunch_unknown_then_safe3_has_no_stop_or_cont(self):
  r=self.run_rows(self.prelaunch()+[sample()]);self.assertEqual(r['prelaunch_recoveries'],1);self.assertEqual(r['freeze_events'],0);self.assertEqual(r['safe_resume_events'],0)
 def test_prelaunch_signal_or_phase_before_finished_rejected(self):
  for row in [event('owned_freeze_signal',pid=101,start_ticks=111,uid=0,role='phase',lease_uuids=UUIDS),sample()]:
   x=self.prelaunch();x.insert(2,row)
   with self.assertRaises(g.JournalError):self.run_rows(x)
 def test_prelaunch_incomplete_safe3_rejected(self):
  x=self.prelaunch();x.pop(2)
  with self.assertRaises(g.JournalError):self.run_rows(x)
 def test_legacy_root_rejected(self):
  with tempfile.TemporaryDirectory()as t:
   p=Path(t).resolve()/'watch.jsonl';p.write_text(json.dumps(sample())+'\n')
   with self.assertRaises(g.JournalError):g.replay(p,JOB,'/dataset/old/runs/3231',SOURCE,HERE/'resource_watch.py')
 def test_duplicate_json_keys_and_partial_line_rejected(self):
  with tempfile.TemporaryDirectory()as t:
   p=Path(t).resolve()/'watch.jsonl';p.write_text('{"run":"a","run":"b"}\n')
   with self.assertRaises(g.JournalError):g.replay(p,JOB,RUN,SOURCE,HERE/'resource_watch.py')
   p.write_text('{"run":')
   with self.assertRaises(ValueError):g.replay(p,JOB,RUN,SOURCE,HERE/'resource_watch.py')
 def test_own_exit_certified_original_epoch_can_finish(self):
  x=recovery();x.insert(4,event('owned_freeze_signal',pid=103,start_ticks=113,uid=0,role='phase_descendant',lease_uuids=UUIDS,certificate=certificate(103,'phase_descendant')))
  for r in x:
   data=r.get('diagnostic',{}).get('samples',[r])[-1]
   if 'owned_before_query'in data:data['owned_before_query']['103']=dict(data['owned_before_query']['101'],start_ticks=113)
  x.insert(-2,event('owned_resume_already_exited',pid=103,start_ticks=113))
  self.assertEqual(self.run_rows(x)['safe_resume_events'],1)
 def test_forged_already_exited_live_owner_rejected(self):
  x=recovery()
  for i,r in enumerate(x):
   if r.get('event')=='owned_resume_signal'and r.get('pid')==102:x[i]=event('owned_resume_already_exited',pid=102,start_ticks=112)
  with self.assertRaises(g.JournalError):self.run_rows(x)
 def test_duplicate_cont_and_unfrozen_gpu_rejected(self):
  x=recovery();x.insert(-2,copy.deepcopy(x[-3]))
  with self.assertRaises(g.JournalError):self.run_rows(x)
  x=recovery();x=[r for r in x if not(r.get('event')=='owned_freeze_signal'and r.get('pid')==102)]
  for r in x:
   if r.get('event')=='owned_freeze_stable':r.update(pids=[101],epochs={'101':111})
  with self.assertRaises(g.JournalError):self.run_rows(x)
 def test_cpu_without_gpu_lease_has_full_certificate_and_ancestry(self):
  x=recovery();x.insert(4,event('owned_freeze_signal',pid=103,start_ticks=113,uid=0,role='phase_descendant',lease_uuids=UUIDS,certificate=certificate(103,'phase_descendant',False)))
  x.insert(-2,event('owned_resume_already_exited',pid=103,start_ticks=113))
  self.assertEqual(self.run_rows(x)['freeze_events'],3)
 def test_certificate_forgery_or_missing_identity_rejected(self):
  for key,val in [('run','/foreign'),('slurm_job_id','1'),('uid',1),('pid',500),('ppid',False),('state','Z'),('start_ticks',False),('allocated_uuid_set',[]),('gpu_leases_inherited',False),('command',['foreign']),('role','monitor')]:
   x=recovery();x[3]['certificate'][key]=val
   with self.subTest(key=key),self.assertRaises(g.JournalError):self.run_rows(x)
  x=recovery();del x[2]['certificate']
  with self.assertRaises(g.JournalError):self.run_rows(x)
 def test_cpu_descendant_bad_or_cyclic_ancestry_rejected(self):
  for parent in (999,103):
   x=recovery();x.insert(4,event('owned_freeze_signal',pid=103,start_ticks=113,uid=0,role='phase_descendant',lease_uuids=UUIDS,certificate=certificate(103,'phase_descendant',False,parent)))
   with self.assertRaises(g.JournalError):self.run_rows(x)
 def test_boolean_and_wrong_phase_role_rejected(self):
  for key,val in [('uid',False),('start_ticks',True),('role','ray_worker')]:
   x=recovery();x[2][key]=val
   with self.assertRaises(g.JournalError):self.run_rows(x)
 def test_held_parent_epoch_tampering_rejected(self):
  for value in (None,0,False,999):
   x=recovery();x[3]['certificate']['parent_start_ticks']=value
   with self.subTest(value=value),self.assertRaises(g.JournalError):self.run_rows(x)
  x=recovery();del x[3]['certificate']['parent_start_ticks']
  with self.assertRaises(g.JournalError):self.run_rows(x)
 def test_exact_cpu_parent_epoch_chain_accepted(self):
  x=recovery();x[3]['role']='phase_descendant';x[3]['certificate']=certificate(102,'phase_descendant')
  for row in x:
   if row.get('event')=='owned_freeze_stable':
    for key in ('before_certificates','after_certificates'):row[key]['102']=dict(certificate(102,'phase_descendant'),state='T')
  self.assertEqual(self.run_rows(x)['status'],'PASS_ACTUAL_GUARD_JOURNAL_REPLAY')
if __name__=='__main__':unittest.main()
