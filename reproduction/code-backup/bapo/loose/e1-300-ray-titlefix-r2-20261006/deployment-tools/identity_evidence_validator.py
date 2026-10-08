"""Strict original Linux identity-only raw proof, supplementary to GPU gates."""
import hashlib,importlib.util,json,math,pathlib,re,types
SCOPE='ACTUAL_LINUX_SETPROCTITLE_IDENTITY_ONLY_NOT_REAL_RAY_ACTOR_OR_GPU_ACCEPTANCE'
CASES={'illegal_title','extra_argv','foreign_job','foreign_uuid','replaced_executable','parent_epoch'}
def need(ok,msg):
 if not ok:raise ValueError(msg)
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def parse(raw):
 def pairs(items):
  d={}
  for key,value in items:need(key not in d,'Duplicate identity JSON key');d[key]=value
  return d
 return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda value:(_ for _ in ()).throw(ValueError('Nonfinite identity JSON')))
def positive_int(v):return type(v)is int and v>0
def load_module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def replay_linux_identity(f,events,w,anchor,phasepid,foreign):
 # Raw evidence is Linux-format. Reuse the bound pure parser's exact code,
 # replacing only its process-local signal constant; never mutate signal module.
 need(all(e.get('signal')==18 and type(e.get('signal'))is int for e in events if e.get('event')=='owned_resume_signal'),'Original Linux SIGCONT must be18')
 fn=f.strict_identity_replay
 globals_linux=dict(fn.__globals__,signal=types.SimpleNamespace(SIGCONT=18))
 parser=types.FunctionType(fn.__code__,globals_linux,fn.__name__,fn.__defaults__,fn.__closure__)
 return parser(events,w,anchor,phasepid,foreign)

def validate_identity_linux_fixture(run,root,source,job,bound,allocated_uuids):
 run=pathlib.Path(run);root=pathlib.Path(root);directory=run/'identity-linux-fixture'
 need(positive_int(job)and run==root/'runs'/f'bapo-instruct300-hardware-{job}','Exact hardware identity input')
 def file(p):
  p=pathlib.Path(p);need(p.is_file()and not p.is_symlink()and p.resolve()==p and root in p.parents,'Canonical identity evidence file')
  need(bound.get(str(p))==sha(p),'Unbound or changed identity evidence '+str(p));return p
 def load(p):return parse(file(p).read_text())
 watchpath=file(root/'tools/resource_watch.py');fixturepath=file(root/'tools/identity-linux-fixture.py')
 w=load_module(watchpath,'bound_identity_evidence_watcher');f=load_module(fixturepath,'bound_identity_evidence_fixture')
 anchor=w.load_runtime_anchor(str(run),verify_current=False)
 approval=load(root/'logs/launch-approval.json');binding=approval['runtime_executable'];file(pathlib.Path(binding['path']))
 need(anchor['source_commit']==source and anchor['path']==str(root/'logs/runtime-executable.json'),'Original identity runtime anchor differs')
 manifest=load(root/'logs/tools-manifest.json');need(manifest['source_commit']==source and manifest['files']['identity-linux-fixture.py']==sha(fixturepath)and manifest['files']['resource_watch.py']==sha(watchpath),'Original identity fixture tool manifest differs')
 result=load(directory/'result.json')
 need(result['status']=='PASS_ACTUAL_LINUX_TITLE_EXEC_IDENTITY_ONLY'and result['scope']==SCOPE,'Identity result scope/status')
 need(result['source_commit']==source and result['run_root']==str(run)and type(result['job_id'])is int and result['job_id']==job,'Identity result allocated source/job/run')
 need(result['fixture_sha256']==sha(fixturepath)and result['watcher_sha256']==sha(watchpath)and result['runtime_anchor']==anchor,'Identity result source/tool/runtime binding')
 need(result['actual_ray_actor_executed']is False and result['gpu_initialized']is False,'Identity fixture must remain supplementary CPU scope')
 need(type(result['foreign_signals'])is int and result['foreign_signals']==0 and type(result['foreign_exit_code'])is int and result['foreign_exit_code']==0 and positive_int(result['foreign_pid']),'Foreign sentinel natural exit0/zero signals')
 foreign=result['foreign_pid'];records=result['records'];need(type(records)is list and len(records)==31,'Exactly25 positive and6 negative original records required')
 positives=[r for r in records if r.get('kind')=='positive'];negatives=[r for r in records if r.get('kind')=='negative']
 titles=sorted(w.RAY_APPROVED_TITLES);need(len(titles)==25 and len(positives)==25 and [r['title']for r in positives]==titles,'Every exact registered title tested once in declared order')
 need(len(negatives)==6 and {r['case']for r in negatives}==CASES,'Six distinct declared adversarial cases required')
 uuids=sorted(allocated_uuids);need(len(uuids)==8 and len(set(uuids))==8,'Original allocated eight UUIDs')
 all_pids=set();states={};raw_count=0
 for p in directory.rglob('*'):
  need(not p.is_symlink(),'Identity raw tree symlink')
  if p.is_file():file(p);raw_count+=1
 def cert(c):
  w.certificate_identity(c,anchor)
  need(c['run']==str(run)and c['slurm_job_id']==str(job)and c['allocated_uuid_set']==uuids and c['gpu_leases_inherited']is True and c['inherited_gpu_lease_uuids']==uuids,'Identity raw process allocation differs')
  need(c['executable']==anchor['executable'],'Identity raw process actual executable differs')
  need(c['pid']!=foreign,'Foreign sentinel included in owned certificate');states.setdefault('uid',c['uid']);need(c['uid']==states['uid'],'Identity process UID changed')
 for index,r in enumerate(positives):
  before=r['before'];after=r['after'];cert(before);cert(after)
  need(before['role']==after['role']=='ray_worker'and before['command'][0]=='ray::IDLE'and not any(before['command'][1:]),'Original real worker IDLE/title layout')
  need(after['command'][0]==r['title']and not any(after['command'][1:]),'Measured exact title/NUL layout')
  need(w.certificate_identity(before,anchor)==w.certificate_identity(after,anchor),'Positive changed immutable process identity')
  need(r['difference']==w.identity_difference(before,after,anchor),'Positive difference cannot be a supplied summary')
  seconds=r['freeze_seconds'];need(type(seconds)in(int,float)and math.isfinite(seconds)and 0<seconds<=5,'Actual original5s freeze deadline')
  out=directory/f'positive-{index}';phase=load(out/'phase.json');ready=load(out/'ready.json');instruction=load(out/'instruction.json');ack=load(out/'ack.json');file(out/'stop')
  pid=before['pid'];phasepid=phase['phase_pid'];need(positive_int(phasepid)and phase['worker_pid']==pid and ready['pid']==pid and before['ppid']==after['ppid']==phasepid,'Actual child/phase PID topology')
  need(instruction==ack and instruction['op']=='title'and instruction['title']==r['title']and positive_int(instruction['sequence']),'Actual title instruction/ack proof')
  need(type(ready['setproctitle_file'])is str and pathlib.Path(ready['setproctitle_file']).is_absolute(),'Installed setproctitle module observation required')
  events=r['journal'];need(type(events)is list and events,'Raw identity STOP/stable/CONT journal required')
  stops=[e for e in events if e.get('event')=='owned_freeze_signal'];need(len(stops)==2 and {e['pid']for e in stops}=={phasepid,pid},'Exactly original own phase/worker stopped')
  for e in stops:
   c=e['certificate'];cert(c);need(e['pid']==c['pid']and e['start_ticks']==c['start_ticks']and e['uid']==c['uid']and e['role']==c['role']and e['slurm_job_id']==str(job)and e['lease_uuids']==uuids,'Raw STOP envelope differs from full certificate')
  need(next(e['certificate']for e in stops if e['pid']==pid)==after,'Measured after must be actual postpidfd STOP certificate')
  phasecert=next(e['certificate']for e in stops if e['pid']==phasepid);need(phasecert['role']=='phase'and before['parent_start_ticks']==after['parent_start_ticks']==phasecert['start_ticks'],'Actual parent epoch proof')
  need([v for v in phasecert['command']if v]==[str(root/'env/bin/python'),str(fixturepath),'--mode','phase','--output',str(out)],'Original installed fixture phase command')
  for at,event in enumerate(events):
   need(event.get('pid')!=foreign,'Foreign sentinel event present')
   if event.get('event')=='owned_title_transition':
    need(at+1<len(events)and events[at+1].get('event')=='owned_freeze_signal'and events[at+1]['pid']==event['pid']and events[at+1]['certificate']==event['after_certificate'],'Transition must immediately pair in current freeze')
   if event.get('event')=='owned_freeze_stable':
    for name in ('before_certificates','after_certificates'):
     for c in event[name].values():cert(c)
  replay=replay_linux_identity(f,events,w,anchor,phasepid,foreign);need(r['replay']==replay and replay['stopped']==replay['resumed']==2,'Raw replay differs from stored summary')
  need(not({pid,phasepid}&all_pids),'Identity process PID reused across records');all_pids.update((pid,phasepid))
 for r in negatives:
  case=r['case'];original=r['original'];before=r['raw_before'];after=r['raw_after'];cert(original)
  need(before['pid']==after['pid']==original['pid']and before['start_ticks']==after['start_ticks']==original['start_ticks'],'Adversary must keep original PID/epoch')
  need(before['uid']==after['uid']==original['uid']and before['ppid']==after['ppid']==original['ppid'],'Adversary owner/parent must remain known')
  need(before['run']==after['run']==str(run)and before['slurm_job_id']==str(job),'Original adversarial run/job')
  need(before['executable']==original['executable']and before['command']==original['command'],'Original adversarial /proc proof differs')
  expected_diff={k:dict(before=before.get(k),after=after.get(k))for k in sorted(set(before)|set(after))if before.get(k)!=after.get(k)}
  need(r['raw_difference']==expected_diff and r['command_before']==w.command_observation(original['command']),'Original raw negative difference differs')
  need(type(r['target_signals'])is int and r['target_signals']==0,'Negative target must receive no signal')
  rejection=r['rejection'];need(type(rejection)is dict and rejection['type']in('IdentityMismatch','IdentityReadMismatch','ValueError','KeyError')and type(rejection['message'])is str and rejection['message'],'Original negative rejection proof')
  if case=='illegal_title':need(after['command'][0]=='ray::UNAPPROVED','Illegal title not actually observed')
  elif case=='extra_argv':need('UNAPPROVED_TAIL'in after['command'][1:],'Unapproved extra argv not actually observed')
  elif case=='foreign_job':need(after['slurm_job_id']!=str(job),'Foreign job not actually observed')
  elif case=='foreign_uuid':need(sorted(v['uuid']for v in parse(after['inherited_gpu_leases_raw']))!=uuids,'Foreign leases not actually observed')
  elif case=='replaced_executable':need(after['executable']!=before['executable'],'Changed executable not actually observed')
  elif case=='parent_epoch':need(original['parent_start_ticks']==before['parent_start_ticks']+1 and after['parent_start_ticks']==before['parent_start_ticks'],'Parent epoch counterexample not bound to actual raw epoch')
  if case!='parent_epoch':need(original['parent_start_ticks']==before['parent_start_ticks'],'Original negative parent epoch differs')
  out=directory/('negative-'+case);phase=load(out/'phase.json');ready=load(out/'ready.json');file(out/'stop')
  need(phase['worker_pid']==ready['pid']==original['pid']and phase['phase_pid']==before['ppid'],'Negative actual process topology')
  pid=original['pid'];phasepid=phase['phase_pid'];need(not({pid,phasepid}&all_pids)and foreign not in(pid,phasepid),'Negative/foreign PID reused');all_pids.update((pid,phasepid))
  # Recompute a rejecting shared identity, not only the stored exception string.
  observed=dict(original,command=after['command'],executable=after['executable'],parent_start_ticks=after['parent_start_ticks'],run=after['run'],slurm_job_id=after['slurm_job_id'])
  observed['inherited_gpu_lease_uuids']=sorted(v['uuid']for v in parse(after['inherited_gpu_leases_raw']))
  try:accepted=w.certificate_identity(observed,anchor)==w.certificate_identity(original,anchor)
  except (ValueError,KeyError,TypeError):accepted=False
  need(not accepted,'Adversarial raw identity would pass shared projection')
 file(directory/'foreign-stop');need(len(all_pids)==62,'All31 original own phase/worker pairs required')
 # Original stdout marker binds the emitted final object to its separate pre-guardian identity fixture log.
 text=file(run/'identity-linux-fixture.log').read_text();marker='ACTUAL_TITLE_EXEC_LINUX_FIXTURE_JSON=';lines=[line.split(marker,1)[1]for line in text.splitlines()if marker in line]
 need(len(lines)==1 and json.JSONDecoder().raw_decode(lines[0])[0]==result,'Actual identity fixture output marker missing/different')
 return dict(scope=SCOPE,positive_titles=25,negative_cases=6,original_process_pairs=31,foreign_signals=0,foreign_exit_code=0,raw_files=raw_count,runtime_anchor_sha256=anchor['sha256'],fixture_sha256=sha(fixturepath),watcher_sha256=sha(watchpath),actual_ray_actor_executed=False,gpu_initialized=False)
