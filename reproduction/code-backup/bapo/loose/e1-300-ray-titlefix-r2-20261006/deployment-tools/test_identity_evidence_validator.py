"""Synthetic complete raw proof semantic attacks; never actual Linux/GPU acceptance."""
import copy,hashlib,importlib.util,json,os,pathlib,shutil,tempfile,unittest
P=pathlib.Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('identity_validator',P/'identity_evidence_validator.py');v=importlib.util.module_from_spec(s);s.loader.exec_module(v)
class Tests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=pathlib.Path(self.tmp.name).resolve();self.job=555;self.source='3f85dabc7aa05bc45ec7af015d65d9fa4a4c17e4';self.run=self.root/'runs/bapo-instruct300-hardware-555';self.d=self.run/'identity-linux-fixture';self.u=sorted('GPU-'+str(i)for i in range(8))
  for d in (self.root/'tools',self.root/'logs',self.d):d.mkdir(parents=True)
  for n in ('resource_watch.py','identity-linux-fixture.py'):shutil.copy2(P/n,self.root/'tools'/n)
  self.w=v.load_module(self.root/'tools/resource_watch.py','synthetic_bound_w');self.f=v.load_module(self.root/'tools/identity-linux-fixture.py','synthetic_bound_f');exe=self.w.executable_identity(__import__('sys').executable)
  self.put(self.root/'source-expected.txt',self.source,raw=True)
  anchor={'source_commit':self.source,'root':str(self.root),'sys_prefix':str(self.root/'env'),'executable':exe};self.put(self.root/'logs/runtime-executable.json',anchor)
  manifest={'source_commit':self.source,'files':{n:v.sha(self.root/'tools'/n)for n in ('resource_watch.py','identity-linux-fixture.py')}};self.put(self.root/'logs/tools-manifest.json',manifest)
  ap=self.root/'logs/runtime-executable.json';self.put(self.root/'logs/launch-approval.json',{'status':'PASS_INDEPENDENT_E1_LAUNCH_REVIEW','source_commit':self.source,'root':str(self.root),'tools_manifest_sha256':v.sha(self.root/'logs/tools-manifest.json'),'runtime_executable':{'path':str(ap),'sha256':v.sha(ap),'bytes':ap.stat().st_size}})
  self.anchor=self.w.load_runtime_anchor(str(self.run));self.records=[]
  for i,title in enumerate(sorted(self.w.RAY_APPROVED_TITLES)):
   phase=1001+i*2;pid=phase+1;out=self.d/f'positive-{i}';out.mkdir();pc=self.cert(phase,42,'phase',[str(self.root/'env/bin/python'),str(self.root/'tools/identity-linux-fixture.py'),'--mode','phase','--output',str(out)]);before=self.cert(pid,phase,'ray_worker',['ray::IDLE','']);after=dict(before,command=[title,'']if title=='ray::IDLE'else[title,'',''])
   events=[]
   events.append(self.stop(pc))
   if before['command']!=after['command']:events.append({'event':'owned_title_transition','pid':pid,'before_certificate':before,'after_certificate':after})
   events.append(self.stop(after));events.append({'event':'owned_freeze_stable','pids':[phase,pid],'epochs':{str(phase):pc['start_ticks'],str(pid):before['start_ticks']},'before_certificates':{str(phase):dict(pc,state='T'),str(pid):dict(after,state='T')},'after_certificates':{str(phase):dict(pc,state='T'),str(pid):dict(after,state='T')}})
   events.extend({'event':'owned_resume_signal','pid':c['pid'],'start_ticks':c['start_ticks'],'signal':18}for c in (pc,after))
   replay=v.replay_linux_identity(self.f,events,self.w,self.anchor,phase,9999)
   self.records.append({'kind':'positive','title':title,'freeze_seconds':.1,'before':before,'after':after,'difference':self.w.identity_difference(before,after,self.anchor),'journal':events,'replay':replay})
   self.put(out/'phase.json',{'phase_pid':phase,'worker_pid':pid});self.put(out/'ready.json',{'pid':pid,'setproctitle_file':'/bound/env/setproctitle/__init__.py'});instruction={'sequence':i+1,'op':'title','title':title};self.put(out/'instruction.json',instruction);self.put(out/'ack.json',instruction);self.put(out/'stop','',raw=True)
  for i,case in enumerate(('illegal_title','extra_argv','foreign_job','foreign_uuid','replaced_executable','parent_epoch')):
   phase=2001+2*i;pid=phase+1;original=self.cert(pid,phase,'ray_worker',['ray::IDLE','']);before={k:copy.deepcopy(original[k])for k in ('pid','ppid','state','start_ticks','uid','parent_start_ticks','command','executable','run','slurm_job_id')};before['inherited_gpu_leases_raw']=json.dumps([{'uuid':u}for u in self.u]);after=copy.deepcopy(before)
   if case=='illegal_title':after['command']=['ray::UNAPPROVED','']
   elif case=='extra_argv':after['command']=['ray::IDLE','-c','code','UNAPPROVED_TAIL']
   elif case=='foreign_job':after['slurm_job_id']='999999999'
   elif case=='foreign_uuid':after['inherited_gpu_leases_raw']=json.dumps([{'uuid':'foreign-'+str(j)}for j in range(8)])
   elif case=='replaced_executable':after['executable']=dict(before['executable'],resolved_path='/usr/bin/sleep',inode=before['executable']['inode']+1);after['command']=['/bin/sleep','2']
   elif case=='parent_epoch':original['parent_start_ticks']+=1
   diff={k:{'before':before.get(k),'after':after.get(k)}for k in sorted(set(before)|set(after))if before.get(k)!=after.get(k)}
   self.records.append({'kind':'negative','case':case,'original':original,'raw_before':before,'raw_after':after,'raw_difference':diff,'command_before':self.w.command_observation(original['command']),'rejection':{'type':'ValueError','message':'synthetic expected rejection','diagnostic':None},'target_signals':0})
   out=self.d/('negative-'+case);out.mkdir();self.put(out/'phase.json',{'phase_pid':phase,'worker_pid':pid});self.put(out/'ready.json',{'pid':pid});self.put(out/'stop','',raw=True)
  self.result={'status':'PASS_ACTUAL_LINUX_TITLE_EXEC_IDENTITY_ONLY','scope':v.SCOPE,'source_commit':self.source,'run_root':str(self.run),'job_id':self.job,'fixture_sha256':v.sha(self.root/'tools/identity-linux-fixture.py'),'watcher_sha256':v.sha(self.root/'tools/resource_watch.py'),'runtime_anchor':self.anchor,'records':self.records,'foreign_pid':9999,'foreign_signals':0,'foreign_exit_code':0,'actual_ray_actor_executed':False,'gpu_initialized':False};self.put(self.d/'foreign-stop','',raw=True);self.refresh()
 def cert(self,pid,ppid,role,command):
  c={'pid':pid,'ppid':ppid,'start_ticks':500+pid,'parent_start_ticks':500+ppid,'uid':os.getuid(),'run':str(self.run),'slurm_job_id':str(self.job),'state':'S','role':role,'command':command,'executable':self.anchor['executable'],'allocated_uuid_set':self.u,'gpu_leases_inherited':True,'inherited_gpu_lease_uuids':self.u}
  if role=='ray_worker':c['runtime_anchor_sha256']=self.anchor['sha256']
  return c
 def stop(self,c):return {'event':'owned_freeze_signal','pid':c['pid'],'start_ticks':c['start_ticks'],'uid':c['uid'],'role':c['role'],'slurm_job_id':str(self.job),'lease_uuids':self.u,'certificate':c}
 def put(self,p,value,raw=False):p.write_text(value if raw else json.dumps(value))
 def refresh(self):
  self.put(self.d/'result.json',self.result);self.put(self.run/'identity-linux-fixture.log','ACTUAL_TITLE_EXEC_LINUX_FIXTURE_JSON='+json.dumps(self.result)+'\n',raw=True)
  self.bound={str(p):v.sha(p)for p in self.root.rglob('*')if p.is_file()}
 def validate(self):return v.validate_identity_linux_fixture(self.run,self.root,self.source,self.job,self.bound,self.u)
 def test_complete_synthetic31_records_shared_replay(self):self.assertEqual(self.validate()['positive_titles'],25);self.assertEqual(self.validate()['negative_cases'],6)
 def test_status_source_anchor_and_foreign_fail_with_all_hashes_rebound(self):
  original=copy.deepcopy(self.result)
  for field,value in [('source_commit','b'*40),('job_id',False),('foreign_exit_code',-9),('foreign_signals',False),('gpu_initialized',True),('runtime_anchor',dict(self.anchor,sha256='0'*64))]:
   self.result=copy.deepcopy(original);self.result[field]=value;self.refresh()
   with self.subTest(field=field),self.assertRaises(ValueError):self.validate()
 def test_omitted_or_duplicate_title_after_record_and_log_rebinding(self):
  self.result['records'][1]=copy.deepcopy(self.result['records'][0]);self.refresh()
  with self.assertRaises(ValueError):self.validate()
 def test_raw_snapshot_and_stored_replay_cannot_claim_pass(self):
  r=self.result['records'][0];pid=r['before']['pid'];next(e for e in r['journal']if e['event']=='owned_freeze_stable')['after_certificates'][str(pid)]['state']='S';self.refresh()
  with self.assertRaises(ValueError):self.validate()
 def test_negative_must_show_real_adversary_not_just_exception_string(self):
  r=next(r for r in self.records if r.get('case')=='foreign_job');r['raw_after']['slurm_job_id']=str(self.job);r['raw_difference']={};self.refresh()
  with self.assertRaises(ValueError):self.validate()
 def test_negative_count_bool_signal_and_source_marker_after_rebind(self):
  self.records[-1]['target_signals']=False;self.refresh()
  with self.assertRaises(ValueError):self.validate()
 def test_title_ack_and_extra_raw_artifact_must_be_bound(self):
  self.put(self.d/'positive-0/ack.json',{'sequence':1,'op':'title','title':'ray::UNAPPROVED'});self.refresh()
  with self.assertRaises(ValueError):self.validate()
 def test_original_identity_stdout_marker_is_mandatory(self):
  self.put(self.run/'identity-linux-fixture.log','',raw=True);self.bound[str(self.run/'identity-linux-fixture.log')]=v.sha(self.run/'identity-linux-fixture.log')
  with self.assertRaises(ValueError):self.validate()
 def test_linux_cont_constant_no_host_global_mutation(self):
  import signal
  host=signal.SIGCONT
  self.assertEqual(self.validate()['positive_titles'],25)
  self.assertEqual(signal.SIGCONT,host)
  for bad in (19,True):
   e=next(e for e in self.records[0]['journal']if e['event']=='owned_resume_signal');e['signal']=bad;self.refresh()
   with self.subTest(signal=bad),self.assertRaises(ValueError):self.validate()
 def test_transition_cannot_claim_different_actual_stop_certificate(self):
  r=self.records[1];event=next(e for e in r['journal']if e['event']=='owned_title_transition');event['after_certificate']=dict(event['after_certificate'],start_ticks=90000);self.refresh()
  with self.assertRaises(ValueError):self.validate()
 def test_all_original_raw_files_bound_not_only_result(self):
  extra=self.d/'additional-original-raw.json';self.put(extra,{'raw':True})
  with self.assertRaises(ValueError):self.validate()
 def test_marker_cannot_be_only_old_hardware_log(self):
  original=(self.run/'identity-linux-fixture.log').read_text();self.put(self.run/'hardware.log',original,raw=True);self.put(self.run/'identity-linux-fixture.log','',raw=True);self.bound={str(p):v.sha(p)for p in self.root.rglob('*')if p.is_file()}
  with self.assertRaises(ValueError):self.validate()
if __name__=='__main__':unittest.main()
