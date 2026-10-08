"""Strict original Linux identity-only raw proof, supplementary to GPU gates."""
import base64,hashlib,importlib.util,json,math,pathlib,re,types
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
 titles=sorted(w.RAY_APPROVED_TITLES);need(result['process_title_policy']=='SPT_NOENV=1','Actual title policy required')
 need(len(titles)==25 and len(positives)==25 and [r['title']for r in positives]==titles,'Every exact registered title tested once in declared order')
 need(len(negatives)==6 and {r['case']for r in negatives}==CASES,'Six distinct declared adversarial cases required')
 uuids=sorted(allocated_uuids);need(len(uuids)==8 and len(set(uuids))==8,'Original allocated eight UUIDs')
 def marker_proof(v):
  need(type(v)is dict and set(v)=={'values','original_selected_environ_base64'},'Raw selected environ proof required');entries=base64.b64decode(v['original_selected_environ_base64'],validate=True).split(b'\0');values={}
  for row in entries:
   if not row:continue
   k,val=row.split(b'=',1);k=k.decode();need(k not in values,'Duplicate proc marker');values[k]=val.decode()
  need(values==v['values']and set(values)=={'BAPO_RUN_ID','SLURM_JOB_ID','SEARCH_R1_GPU_LEASES','SPT_NOENV'},'Original proc marker values differ')
  need(values['BAPO_RUN_ID']==str(run)and values['SLURM_JOB_ID']==str(job)and values['SPT_NOENV']=='1','Actual task markers/title policy differ')
  leases=parse(values['SEARCH_R1_GPU_LEASES']);need(type(leases)is list and len(leases)==8 and sorted(x['uuid']for x in leases)==uuids,'Actual proc lease markers differ')
 def initial_markers(ready):
  marker_proof(ready['process_markers_before']);marker_proof(ready['process_markers_after']);need(ready['process_markers_before']==ready['process_markers_after'],'Initial title changed original proc markers')
  need(type(ready['argv_capacity_bytes'])is int and ready['argv_capacity_bytes']>0,'Original argv capacity required')
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
  initial_markers(ready);env_ack=load(out/'environment-ack.json');need(env_ack==r['environment_ack'],'Original environment acknowledgement differs');need(env_ack['sequence']==instruction['sequence']and env_ack['title']==r['title'],'Title environment sequence differs');marker_proof(env_ack['before']);marker_proof(env_ack['after']);need(env_ack['before']==env_ack['after']==ready['process_markers_before'],'Positive title changed actual proc markers');need(type(env_ack['argv_capacity_bytes'])is int and env_ack['argv_capacity_bytes']==ready['argv_capacity_bytes']and len(r['title'].encode())+1<=env_ack['argv_capacity_bytes'],'Original title capacity/truncation proof differs')
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
  initial_markers(ready);need(phase['worker_pid']==ready['pid']==original['pid']and phase['phase_pid']==before['ppid'],'Negative actual process topology')
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


def validate_production_ray_identity(run,root,source,job,bound,allocated_uuids):
 """Independent original-proc parser; no live process/Ray/GPU calls."""
 run=pathlib.Path(run);root=pathlib.Path(root)
 need(positive_int(job)and run==root/'runs'/f'bapo-instruct300-hardware-{job}','Exact actual production identity job/run')
 def file(path):
  path=pathlib.Path(path);need(path.is_file()and not path.is_symlink()and path.resolve()==path and root in path.parents and bound.get(str(path))==sha(path),'Unbound original production Ray evidence '+str(path));return path
 def load(path):return parse(file(path).read_text())
 watch=file(root/'tools/resource_watch.py');sampler=file(root/'tools/actual-title-dependency-preflight.py');file(run/'production-ray-identity.log')
 w=load_module(watch,'bound_production_ray_identity_watcher');anchor=w.load_runtime_anchor(str(run),verify_current=False)
 record=load(run/'production-ray-identity/result.json');manifest=load(root/'logs/tools-manifest.json');runtime=load(root/'logs/runtime-executable.json')
 need(record['status']=='COMPLETE_ACTUAL_PRODUCTION_RAY8_PROC_IDENTITY_NOT_GPU_ACCEPTANCE'and record['root']==str(root)and record['run_root']==str(run)and record['source_commit']==source and type(record['job_id'])is int and record['job_id']==job,'Production Ray original source/job/run')
 need(record['tool_manifest_sha256']==sha(root/'logs/tools-manifest.json')and manifest['source_commit']==source and manifest['files']['actual-title-dependency-preflight.py']==sha(sampler)and record['sampler_sha256']==sha(sampler)and manifest['files']['resource_watch.py']==sha(watch),'Actual production sampler/source/tool original binding')
 need(record['runtime_anchor']==anchor and runtime['source_commit']==source and runtime['root']==str(root),'Actual production runtime anchor original binding')
 for key in ('signals_sent','extra_ray_calls','extra_training_updates'):need(type(record[key])is int and record[key]==0,'Sampler must remain pure read-only')
 terminal=file(run/'slurm-terminal.txt').read_text();uidmatch=re.search(r'(?:^|\s)UserId=[^\s()]+\((\d+)\)(?:\s|$)',terminal);need(uidmatch and type(record['collector_uid'])is int and record['collector_uid']==int(uidmatch.group(1)),'Actual Slurm UID binding')
 rows=record['ranks'];need(type(rows)is list and len(rows)==8 and [v['rank']for v in rows]==list(range(8)),'Actual eight distinct ordered restored ranks')
 expected=record['expected_process_markers'];need(set(expected)=={'BAPO_RUN_ID','SLURM_JOB_ID','SEARCH_R1_GPU_LEASES','SPT_NOENV'}and expected['BAPO_RUN_ID']==str(run)and expected['SLURM_JOB_ID']==str(job)and expected['SPT_NOENV']=='1','Actual production expected original markers')
 lease=parse(expected['SEARCH_R1_GPU_LEASES']);need(type(lease)is list and len(lease)==8 and sorted(v['uuid']for v in lease)==sorted(allocated_uuids)and len(set(allocated_uuids))==8,'Actual production complete allocation leases')
 executable=dict(anchor['executable'],sha256=runtime['executable_sha256'])
 pids=[]
 def snapshot(v,witness):
  need(type(v['pid'])is int and v['pid']==witness['new_pid']and positive_int(v['start_ticks'])and v['start_ticks']==witness['process']['start_ticks'],'Actual restored process PID/epoch')
  need(type(v['uid'])is int and v['uid']==record['collector_uid']and v['executable']==executable and positive_int(v['captured_monotonic_ns']),'Actual restored runtime physical executable/UID/time')
  marker=v['process_markers'];need(set(marker)=={'values','original_selected_environ_base64'},'Original raw proc marker structure');rawenv=base64.b64decode(marker['original_selected_environ_base64'],validate=True);need(rawenv.endswith(b'\0'),'Original proc env NUL terminator');values={}
  for row in rawenv.split(b'\0'):
   if not row:continue
   need(b'='in row,'Original proc env key/value');k,val=row.split(b'=',1);key=k.decode();need(key not in values,'Duplicate original proc env marker');values[key]=val.decode()
  need(values==expected and marker['values']==values,'Original production proc markers clobbered or rebound')
  raw=base64.b64decode(v['cmdline_base64'],validate=True);need(raw==('\0'.join(v['command'])).encode()and len(raw)>0 and raw.endswith(b'\0')and not any(v['command'][1:]),'Original actual Ray title NUL padding')
  title=v['command'][0];need(title in w.RAY_APPROVED_TITLES and len(title.encode())+1<=len(raw),'Actual production approved title/capacity');return title,len(raw)
 for rank,row in enumerate(rows):
  witnesspath=file(run/'recovery-witness'/f'rank_{rank}.json');v3path=file(run/'recovery-native-v3-roundtrip'/f'rank_{rank}.json');witness=load(witnesspath);v3=load(v3path)
  need(row['witness_path']==str(witnesspath)and row['witness_sha256']==sha(witnesspath)and row['v3_report_path']==str(v3path)and row['v3_report_sha256']==sha(v3path),'Actual original restored/v3 report binding')
  need(type(witness['rank'])is int and witness['rank']==rank and witness['world_size']==8 and witness['status']=='EXACT_CROSS_JOB_FULL_STATE_RELOAD_OK','Actual original eight-rank restored witness')
  need(v3['status']=='EXACT_FULL_STATE_RELOAD_OK'and v3['rank']==rank and v3['world_size']==8 and v3['pid']==witness['new_pid']and v3['roundtrip']is True and v3['native_state_version']==3,'Actual native production v3 restored process report')
  active=row['active'];post=row['post_v3'];need(row['snapshots']==[active,post],'Exact original two-boundary snapshot trace')
  title,capacity=snapshot(active,witness);posttitle,postcapacity=snapshot(post,witness)
  need(title in {'ray::WorkerDict.actor_rollout_load_training_state','ray::WorkerDict.actor_rollout_save_training_state'},'Actual registered production restore/save title observed')
  need(positive_int(row['v3_report_observed_monotonic_ns'])and active['captured_monotonic_ns']<row['v3_report_observed_monotonic_ns']<=post['captured_monotonic_ns']and postcapacity==capacity and active['process_markers']==post['process_markers'],'Actual ordered post-v3 capture with unchanged raw markers/capacity')
  pids.append(witness['new_pid'])
 need(len(set(pids))==8,'Actual eight unique production Ray workers')
 return dict(status='RAW_ACTUAL_PRODUCTION_RAY8_PROC_IDENTITY_PASS',actual_ray_workers=8,signals_sent=0,extra_training_updates=0)
