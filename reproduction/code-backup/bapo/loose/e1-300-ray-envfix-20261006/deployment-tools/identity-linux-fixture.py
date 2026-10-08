"""Real Linux proc/pidfd + installed setproctitle fixture. Never a GPU permission."""
import argparse,hashlib,importlib.util,json,os,pathlib,signal,subprocess,sys,time
SCOPE='ACTUAL_LINUX_SETPROCTITLE_IDENTITY_ONLY_NOT_REAL_RAY_ACTOR_OR_GPU_ACCEPTANCE'
def need(v,msg):
 if not v:raise ValueError(msg)
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def put(p,v):
 p=pathlib.Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,allow_nan=False));os.replace(tmp,p)
def get(p):return json.loads(pathlib.Path(p).read_text())
def wait(fn,seconds=5):
 end=time.monotonic()+seconds
 while time.monotonic()<end:
  v=fn()
  if v:return v
  time.sleep(.01)
 raise TimeoutError('Fixture bounded wait')
def module(path,expected):
 p=pathlib.Path(path);need(p.resolve()==p and p.is_file()and sha(p)==expected,'Bound watcher required');spec=importlib.util.spec_from_file_location('identity_fixture_w',p);w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w);need(sha(p)==expected,'Watcher changed');return w

MARKER_KEYS=('BAPO_RUN_ID','SLURM_JOB_ID','SEARCH_R1_GPU_LEASES','SPT_NOENV')
def process_markers(pid):
 entries=(pathlib.Path('/proc')/str(pid)/'environ').read_bytes().split(b'\0');selected=[];values={}
 for row in entries:
  if b'='not in row:continue
  key,value=row.split(b'=',1)
  if key.decode(errors='replace')in MARKER_KEYS:
   name=key.decode();need(name not in values,'Duplicate original process marker');values[name]=value.decode();selected.append(row)
 need(set(values)==set(MARKER_KEYS)and values['SPT_NOENV']=='1','Original process markers missing or SPT policy differs')
 return dict(values=values,original_selected_environ_base64=__import__('base64').b64encode(b'\0'.join(selected)+b'\0').decode())

def worker(a):
 need(os.environ.get('SPT_NOENV')=='1','Title policy must precede Ray import');markers_before=process_markers(os.getpid());capacity=len((pathlib.Path('/proc')/str(os.getpid())/'cmdline').read_bytes())
 import ray
 import setproctitle
 d=pathlib.Path(a.output);deadline=time.monotonic()+90;seen=None
 setproctitle.setproctitle('ray::IDLE');markers_after=process_markers(os.getpid());need(markers_before==markers_after,'Initial real title clobbered ownership markers');put(d/'ready.json',dict(pid=os.getpid(),setproctitle_file=setproctitle.__file__,process_markers_before=markers_before,process_markers_after=markers_after,argv_capacity_bytes=capacity))
 while not(d/'stop').exists()and time.monotonic()<deadline:
  p=d/'instruction.json'
  if p.exists():
   r=get(p)
   if r['sequence']!=seen:
    seen=r['sequence'];op=r['op']
    if op=='title':
     need(len(r['title'].encode())+1<=capacity,'Actual argv cannot fit complete title');before=process_markers(os.getpid());setproctitle.setproctitle(r['title']);after=process_markers(os.getpid());need(before==after,'Real title clobbered process markers');put(d/'environment-ack.json',dict(sequence=r['sequence'],title=r['title'],before=before,after=after,argv_capacity_bytes=capacity));put(d/'ack.json',r)
    elif op=='exec_sleep':os.execve('/bin/sleep',['/bin/sleep','2'],dict(os.environ))
    elif op=='exec_python':
     env=dict(os.environ);env.update(r['environment']);code='import ray,setproctitle,time;setproctitle.setproctitle("ray::IDLE");time.sleep(2)'
     os.execve(sys.executable,[sys.executable,'-c',code],env)
    elif op=='extra_argv':
     os.execve(sys.executable,['ray::IDLE','-c','import time;time.sleep(2)','UNAPPROVED_TAIL'],dict(os.environ))
    else:raise ValueError('Unknown worker operation')
  time.sleep(.005)

def phase(a):
 d=pathlib.Path(a.output);child=subprocess.Popen([sys.executable,str(pathlib.Path(__file__).resolve()),'--mode','worker','--output',str(d)])
 put(d/'phase.json',dict(phase_pid=os.getpid(),worker_pid=child.pid));deadline=time.monotonic()+90
 while child.poll()is None and not(d/'stop').exists()and time.monotonic()<deadline:time.sleep(.01)
 (d/'stop').touch();child.wait(timeout=5);need(child.returncode==0,'Owned fixture worker did not exit naturally')

def sentinel(a):
 d=pathlib.Path(a.output);end=time.monotonic()+90
 while not(d/'foreign-stop').exists()and time.monotonic()<end:time.sleep(.01)

def strict_identity_replay(events,w,anchor,phase_pid,foreign_pid):
 """Identity-only grammar; cannot validate GPU resource samples or grant safe3."""
 held={};stable=False;resumed=set();transitions=0;resuming=False;pending={}
 for row in events:
  event=row['event'];pid=row.get('pid');need(pid!=foreign_pid,'Foreign sentinel signal/transition')
  if event=='owned_title_transition':
   need(not resuming and pid not in held and pid not in pending,'Title transition after STOP/resume or duplicate')
   before=row['before_certificate'];after=row['after_certificate'];diff=w.identity_difference(before,after,anchor)
   need(before['pid']==after['pid']==pid and diff['immutable_fields']=={} and diff['observed_fields'],'Invalid title transition');pending[pid]=dict(after);transitions+=1
  elif event=='owned_freeze_signal':
   need(not resuming,'STOP after resume started')
   c=row['certificate'];w.certificate_identity(c,anchor);need(c['pid']==pid and pid not in held,'Duplicate/invalid STOP')
   if pid in pending:need(c==pending.pop(pid),'Transition not bound to next STOP certificate')
   held[pid]=c;stable=False
  elif event=='owned_freeze_stable':
   need(not resuming and not pending and phase_pid in held and set(row['pids'])==set(held),'Stable set not complete')
   need(row['epochs']=={str(p):c['start_ticks']for p,c in held.items()},'Stable epoch differs')
   for name in ('before_certificates','after_certificates'):
    snapshot=row[name];need(set(snapshot)=={str(p)for p in held},'Missing snapshot')
    for p,c in held.items():
     observed=snapshot[str(p)];need(observed['state']in('T','t')and w.certificate_identity(c,anchor)==w.certificate_identity(observed,anchor),'Stopped immutable identity differs')
   stable=True
  elif event=='owned_resume_signal':
   need(not pending,'Unresolved title transition before CONT')
   need(stable and pid in held and pid not in resumed and row['signal']==int(signal.SIGCONT)and row['start_ticks']==held[pid]['start_ticks'],'Invalid CONT');resuming=True;resumed.add(pid)
  else:raise ValueError('Unexpected identity journal event '+event)
 need(stable and not pending and resumed==set(held),'Unfinished identity freeze');return dict(stopped=len(held),resumed=len(resumed),title_transitions=transitions)

def raw_process_observation(w,pid):
 p=pathlib.Path('/proc')/str(pid);fields=(p/'stat').read_text().rsplit(')',1)[1].split();environment=(p/'environ').read_bytes().split(b'\0');values=dict(x.split(b'=',1)for x in environment if b'='in x)
 parent=int(fields[1]);parent_fields=(pathlib.Path('/proc')/str(parent)/'stat').read_text().rsplit(')',1)[1].split()
 return dict(pid=pid,ppid=parent,state=fields[0],start_ticks=int(fields[19]),uid=p.stat().st_uid,parent_start_ticks=int(parent_fields[19]),command=(p/'cmdline').read_bytes().decode(errors='replace').split('\0'),executable=w.executable_identity(p/'exe'),run=values.get(b'BAPO_RUN_ID',b'').decode(),slurm_job_id=values.get(b'SLURM_JOB_ID',b'').decode(),inherited_gpu_leases_raw=values.get(b'SEARCH_R1_GPU_LEASES',b'').decode())

def controller(a):
 need(os.environ.get('SPT_NOENV')=='1','Inherited task title policy required');need(sys.platform=='linux'and hasattr(os,'pidfd_open')and hasattr(signal,'pidfd_send_signal'),'Actual Linux pidfd required')
 run=pathlib.Path(a.run);need(run.is_absolute()and run.resolve()==run and run.is_dir(),'Canonical existing owned run required')
 need(os.environ.get('BAPO_RUN_ID')==str(run)and os.environ.get('SLURM_JOB_ID','').isdigit(),'Actual run/job markers required')
 need((run.parent.parent/'source-expected.txt').read_text().strip()==a.source,'Actual source marker')
 w=module(a.watch,a.watch_sha);anchor=w.load_runtime_anchor(str(run),verify_current=True);need(anchor['source_commit']==a.source,'Bound source anchor')
 root=run.parent.parent;self_path=pathlib.Path(__file__).resolve();need(self_path==root/'tools/identity-linux-fixture.py'and pathlib.Path(a.watch)==root/'tools/resource_watch.py','Exact installed fixture/watcher tools required')
 manifest=get(root/'logs/tools-manifest.json');need(manifest['source_commit']==a.source and manifest['files']['identity-linux-fixture.py']==sha(self_path)and manifest['files']['resource_watch.py']==a.watch_sha,'Actual fixture manifest binding')
 d=run/'identity-linux-fixture';d.mkdir();records=[];foreign=subprocess.Popen([sys.executable,str(pathlib.Path(__file__).resolve()),'--mode','sentinel','--output',str(d)],env=dict(os.environ,BAPO_RUN_ID=str(run)+'-fixture-foreign',CUDA_VISIBLE_DEVICES=''))
 phases=[];signals=[]
 def send(fd,sig,*rest):signals.append(dict(fd=fd,signal=int(sig)));return signal.pidfd_send_signal(fd,sig,*rest)
 def launch(name):
  out=d/name;out.mkdir();child=subprocess.Popen([sys.executable,str(pathlib.Path(__file__).resolve()),'--mode','phase','--output',str(out)],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
  phases.append((child,out));info=wait(lambda:get(out/'phase.json')if(out/'phase.json').exists()else None);wait(lambda:(out/'ready.json').exists());return child,out,info['worker_pid']
 seq=0
 def instruction(out,op,**kwargs):
  nonlocal seq
  seq+=1;put(out/'instruction.json',dict(sequence=seq,op=op,**kwargs))
  if op=='title':wait(lambda:get(out/'ack.json').get('sequence')==seq if(out/'ack.json').exists()else False)
 def snapshot(child,pid):return w.freeze_inventory(str(run),os.environ['SLURM_JOB_ID'],w.lease_uuids(),child.pid)[pid]
 try:
  # Every fixed registered title is visited in a real /proc process. No actor/FSDP
  # execution is claimed: only installed setproctitle's actual argv rewrite.
  for index,title in enumerate(sorted(w.RAY_APPROVED_TITLES)):
   child,out,pid=launch('positive-'+str(index));before=snapshot(child,pid);events=[];changed=False
   def opener(target,flags):
    nonlocal changed
    if target==pid and not changed:instruction(out,'title',title=title);changed=True
    return os.pidfd_open(target,flags)
   # Restrict inventory to current test-created tree, refuse any extra same-run
   # process instead of filtering it (tests run in their dedicated CPU run).
   def inv(r,j,u,p):
    rows=w.freeze_inventory(r,j,u,p);need(set(rows)=={child.pid,pid},'Unrelated same-run process prevents fixture');return rows
   freezer=w.OwnedFreezer(str(run),child.pid,events.append,inventory_fn=inv,opener=opener,sender=send)
   start=time.monotonic()
   try:
    freezer.freeze();seconds=time.monotonic()-start;need(seconds<=5,'Original freeze deadline exceeded')
    after=freezer.held[pid]['proof'];diff=w.identity_difference(before,after,anchor);need(diff['immutable_fields']=={},'Allowed title changed immutable fields');need(after['command'][0]==title and not any(after['command'][1:]),'True NUL padding/title mismatch')
    need(w.executable_identity('/proc/'+str(foreign.pid)+'/exe')==anchor['executable'],'Foreign sentinel unexpected binary');need((pathlib.Path('/proc')/str(foreign.pid)/'stat').read_text().rsplit(')',1)[1].split()[0]not in('T','t'),'Foreign sentinel stopped')
    freezer.thaw();summary=strict_identity_replay(events,w,anchor,child.pid,foreign.pid)
    environment_ack=get(out/'environment-ack.json');records.append(dict(kind='positive',title=title,environment_ack=environment_ack,freeze_seconds=seconds,before=before,after=after,difference=diff,journal=events,replay=summary))
   finally:
    if freezer.held:freezer.thaw(abort=True)
    (out/'stop').touch();child.wait(timeout=5);need(child.returncode==0,'Positive phase exit not natural zero')
  # Real exec/env/title changes measured with original certificate then actual
  # local recheck. Only after a successful certificate would STOP be sent.
  for name,op,args in [('illegal_title','title',dict(title='ray::UNAPPROVED')),('extra_argv','extra_argv',{}),('foreign_job','exec_python',dict(environment={'SLURM_JOB_ID':'999999999'})),('foreign_uuid','exec_python',dict(environment={'SEARCH_R1_GPU_LEASES':json.dumps([{'uuid':'foreign-'+str(i)}for i in range(8)])})),('replaced_executable','exec_sleep',{}),('parent_epoch','parent_epoch',{})]:
   child,out,pid=launch('negative-'+name);proof=snapshot(child,pid);real_before=raw_process_observation(w,pid);observed_before=w.command_observation(proof['command']);before_signals=len(signals)
   if op=='parent_epoch':proof['parent_start_ticks']+=1
   else:
    instruction(out,op,**args)
    if op!='title':wait(lambda:w.executable_identity('/proc/'+str(pid)+'/exe')!=proof['executable']if op=='exec_sleep'else(pathlib.Path('/proc')/str(pid)/'cmdline').read_bytes()!=('\0'.join(proof['command'])).encode(),seconds=1)
   real_after=raw_process_observation(w,pid)
   raw_difference={k:dict(before=real_before.get(k),after=real_after.get(k))for k in sorted(set(real_before)|set(real_after))if real_before.get(k)!=real_after.get(k)}
   rejected=None
   try:
    fd=os.pidfd_open(pid,0)
    try:w.local_freeze_recheck(proof,{},anchor=anchor);raise AssertionError('Changed identity accepted before signal')
    finally:os.close(fd)
   except (ValueError,KeyError)as exc:rejected=dict(type=type(exc).__name__,message=str(exc),diagnostic=getattr(exc,'diagnostic',None))
   need(rejected is not None and len(signals)==before_signals,'Negative identity was signalled or accepted')
   records.append(dict(kind='negative',case=name,original=proof,raw_before=real_before,raw_after=real_after,raw_difference=raw_difference,command_before=observed_before,rejection=rejected,target_signals=0))
   (out/'stop').touch();child.wait(timeout=5)
 finally:
  for child,out in phases:
   (out/'stop').touch()
   if child.poll()is None:child.wait(timeout=6)
  (d/'foreign-stop').touch();foreign.wait(timeout=5);need(foreign.returncode==0,'Foreign sentinel not natural exit0')
 result=dict(status='PASS_ACTUAL_LINUX_TITLE_EXEC_IDENTITY_ONLY',scope=SCOPE,source_commit=a.source,run_root=str(run),job_id=int(os.environ['SLURM_JOB_ID']),fixture_sha256=sha(__file__),watcher_sha256=a.watch_sha,runtime_anchor=anchor,records=records,foreign_pid=foreign.pid,foreign_signals=0,foreign_exit_code=foreign.returncode,actual_ray_actor_executed=False,gpu_initialized=False,process_title_policy='SPT_NOENV=1')
 put(d/'result.json',result);print('ACTUAL_TITLE_EXEC_LINUX_FIXTURE_JSON='+json.dumps(result,allow_nan=False),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=('controller','phase','worker','sentinel'),default='controller');p.add_argument('--run');p.add_argument('--source');p.add_argument('--watch');p.add_argument('--watch-sha');p.add_argument('--output');a=p.parse_args()
 return globals()[a.mode](a)
if __name__=='__main__':main()
