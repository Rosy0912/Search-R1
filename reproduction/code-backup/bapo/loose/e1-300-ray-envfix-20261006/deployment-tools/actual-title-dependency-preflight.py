"""CPU-only real Ray-bundled setproctitle/proc proof; no Ray.init or signals."""
import argparse,base64,hashlib,importlib.util,inspect,json,os,pathlib,select,subprocess,sys,time
ROOT='/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-envfix-20261006'
SOURCE='a2568d0df306ead6e059760e28b5bd53ee3f6876'
EXT_SHA='9429a267fba07b2ec523ae906878e154dbe053065c4e89c7e450037866c2557b'
def need(v,msg):
 if not v:raise ValueError(msg)
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def ordinary(p):
 p=pathlib.Path(p);need(p.is_file()and not p.is_symlink(),'Ordinary original required');return p

def dependency():
 import ray,setproctitle,ray._private.worker as worker
 p=pathlib.Path(setproctitle.__file__).resolve();expected=(pathlib.Path(ray.__file__).resolve().parent/'thirdparty_files'/'setproctitle.cpython-310-x86_64-linux-gnu.so').resolve()
 need(ray.__version__=='2.40.0'and p==expected and p.stat().st_size==69104 and sha(p)==EXT_SHA,'Existing Ray extension identity differs')
 need(worker.setproctitle.setproctitle is setproctitle.setproctitle,'Ray worker title function differs')
 return setproctitle,dict(ray_version=ray.__version__,ray_init_path=str(pathlib.Path(ray.__file__).resolve()),ray_init_sha256=sha(ray.__file__),extension_path=str(p),extension_bytes=p.stat().st_size,extension_sha256=sha(p),function_module=setproctitle.setproctitle.__module__,function_name=setproctitle.setproctitle.__name__,same_as_ray_worker=True,function_builtin=inspect.isbuiltin(setproctitle.setproctitle),worker_title_source=inspect.getsource(worker._changeproctitle))

MARKER_KEYS=('BAPO_RUN_ID','SLURM_JOB_ID','SEARCH_R1_GPU_LEASES','SPT_NOENV')
def process_markers(pid):
 entries=(pathlib.Path('/proc')/str(pid)/'environ').read_bytes().split(b'\0');selected=[];values={}
 for row in entries:
  if b'='not in row:continue
  key,value=row.split(b'=',1)
  if key.decode(errors='replace')in MARKER_KEYS:
   name=key.decode();need(name not in values,'Duplicate original process marker');values[name]=value.decode();selected.append(row)
 need(set(values)==set(MARKER_KEYS)and values['SPT_NOENV']=='1','Original process markers missing or title policy not1')
 return dict(values=values,original_selected_environ_base64=base64.b64encode(b'\0'.join(selected)+b'\0').decode())

def proc(pid):
 p=pathlib.Path('/proc')/str(pid);stat=(p/'stat').read_text();tail=stat[stat.rfind(')')+2:].split();exe=(p/'exe').resolve();s=exe.stat();raw=(p/'cmdline').read_bytes()
 return dict(pid=pid,start_ticks=int(tail[19]),executable=dict(resolved_path=str(exe),device=s.st_dev,inode=s.st_ino,sha256=sha(exe)),cmdline_base64=base64.b64encode(raw).decode(),command=raw.decode().split('\0'),uid=p.stat().st_uid,process_markers=process_markers(pid))
def validate_snapshot(before,after,title):
 for k in ('pid','start_ticks','executable','uid','process_markers'):need(before[k]==after[k],'Child identity changed '+k)
 raw=base64.b64decode(after['cmdline_base64'],validate=True);need(raw==('\0'.join(after['command'])).encode(),'Raw command differs');capacity=len(base64.b64decode(before['cmdline_base64'],validate=True));need(len(title.encode())+1<=capacity and len(raw)==capacity,'Original argv capacity insufficient or changed');need(after['command'][0]==title and not any(after['command'][1:]),'Real title or NUL padding differs')

def child():
 need(sys.platform=='linux'and os.environ.get('CUDA_VISIBLE_DEVICES')==''and os.environ.get('SPT_NOENV')=='1','CPU-only Linux child required');title,dep=dependency();print(json.dumps(dict(type='ready',pid=os.getpid(),dependency=dep)),flush=True);deadline=time.monotonic()+90
 while time.monotonic()<deadline:
  if not select.select([sys.stdin],[],[],min(1,max(0,deadline-time.monotonic())))[0]:continue
  line=sys.stdin.readline()
  if not line:return
  request=json.loads(line)
  if request=={'stop':True}:return
  value=request['title'];title.setproctitle(value);print(json.dumps(dict(type='ack',title=value,sequence=request['sequence'],snapshot=proc(os.getpid()))),flush=True)
 raise TimeoutError('Own child natural bounded deadline')
def read(child):
 need(select.select([child.stdout],[],[],10)[0],'Child bounded acknowledgement missing');line=child.stdout.readline();need(line,'Own child exited before acknowledgement');return json.loads(line)
def controller(output):
 need(os.environ.get('SPT_NOENV')=='1','Title env protection must be inherited before import');need(sys.platform=='linux'and pathlib.Path(sys.prefix)==pathlib.Path(ROOT)/'env','Actual target Linux interpreter required')
 need(pathlib.Path(sys.executable).resolve()==(pathlib.Path(ROOT)/'env/bin/python').resolve(),'Actual env interpreter differs')
 tools=pathlib.Path(ROOT)/'tools';manifest_path=ordinary(pathlib.Path(ROOT)/'logs/tools-manifest.json');manifest=json.loads(manifest_path.read_text());need(manifest['source_commit']==SOURCE,'Manifest source differs')
 for name in ('actual-title-dependency-preflight.py','identity-linux-fixture.py','resource_watch.py'):
  p=ordinary(tools/name);need(sha(p)==manifest['files'][name],'Actual manifest tool differs '+name)
 need(pathlib.Path(__file__).resolve()==tools/'actual-title-dependency-preflight.py','Manifest-bound tool route required')
 actual=subprocess.check_output(['git','-C',ROOT+'/repo','rev-parse','HEAD'],text=True).strip();need(actual==SOURCE,'Actual fixed source differs')
 need(not subprocess.check_output(['git','-C',ROOT+'/repo','status','--porcelain'],text=True).strip(),'Source must remain clean')
 spec=importlib.util.spec_from_file_location('approved_watcher',tools/'resource_watch.py');w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w);titles=sorted(w.RAY_APPROVED_TITLES);need(len(titles)==25,'Exact25 approved titles required')
 _,parent_dependency=dependency();out=pathlib.Path(output);need(out.parent==pathlib.Path(ROOT)/'logs'and not out.exists(),'Fresh newROOT logs output required');marker_run=str(out.parent/'title-dependency-owned-cpu-child');synthetic_leases=json.dumps([{'uuid':'CPU-TITLE-PREFLIGHT-'+str(i)}for i in range(8)],sort_keys=True);env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',SPT_NOENV='1',BAPO_RUN_ID=marker_run,SLURM_JOB_ID='1',SEARCH_R1_GPU_LEASES=synthetic_leases);env.pop('PYTHONPATH',None)
 c=subprocess.Popen([sys.executable,'-B',str(tools/'actual-title-dependency-preflight.py'),'--child'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=env);record=dict(scope='ACTUAL_LINUX_RAY_VENDORED_TITLE_CPU_ONLY_NOT_GPU_OR_TRAINING',root=ROOT,source_commit=SOURCE,sys_prefix=sys.prefix,tool_manifest_sha256=sha(manifest_path),fixture_sha256=manifest['files']['identity-linux-fixture.py'],self_sha256=sha(__file__),acks=[],signals_sent=0,ray_init_called=False,actual_gpu_accepted=False)
 try:
  ready=read(c);need(ready['type']=='ready'and ready['pid']==c.pid and ready['dependency']==parent_dependency,'Child ready identity differs');record['ready']=ready;baseline=proc(c.pid);expected_markers={key:env[key]for key in MARKER_KEYS};need(baseline['process_markers']['values']==expected_markers,'Actual original CPU child markers differ');record['marker_mode']='SYNTHETIC_CPU_MARKERS_NOT_SLURM_ALLOCATION';record['expected_process_markers']=expected_markers;record['argv_capacity_bytes']=len(base64.b64decode(baseline['cmdline_base64'],validate=True))
  for index,title in enumerate(titles):
   c.stdin.write(json.dumps(dict(sequence=index,title=title))+'\n');c.stdin.flush();ack=read(c);need(ack['type']=='ack'and ack['title']==title and ack['sequence']==index,'Actual acknowledgement differs');external=proc(c.pid);need(ack['snapshot']==external,'Child/proc snapshot differs');validate_snapshot(baseline,external,title);record['acks'].append(ack)
  _,after_dependency=dependency();need(after_dependency==parent_dependency,'Actual dependency changed during25 titles');record['dependency_after']=after_dependency
 finally:
  if c.poll()is None:
   try:c.stdin.write('{"stop":true}\n');c.stdin.flush()
   except BrokenPipeError:pass
  c.stdin.close();c.wait(timeout=95);stderr=c.stderr.read();record.update(child_exit_code=c.returncode,child_stderr=stderr)
 need(c.returncode==0 and stderr==''and len(record['acks'])==25,'Own child did not complete naturally');record['status']='PASS_ACTUAL_CPU25_TITLES_NOT_GPU_OR_PERMISSION';record['original_process']=baseline
 with out.open('x')as f:f.write(json.dumps(record,indent=2,allow_nan=False)+'\n')
 print(json.dumps(dict(status=record['status'],path=str(out),sha256=sha(out),bytes=out.stat().st_size)))
def production_snapshot(pid,exe_cache):
 p=pathlib.Path('/proc')/str(pid)
 def epoch():
  text=(p/'stat').read_text();tail=text[text.rfind(')')+2:].split();return int(tail[19])
 def physical():
  exe=(p/'exe').resolve();info=exe.stat();return (str(exe),info.st_dev,info.st_ino),exe
 before=epoch();uid_before=p.stat().st_uid;key,exe=physical()
 if key not in exe_cache:exe_cache[key]=sha(exe)
 raw=(p/'cmdline').read_bytes();markers=process_markers(pid);after=epoch();key_after,_=physical();uid_after=p.stat().st_uid
 need(before==after,'Ray PID epoch changed during original capture')
 need(key_after==key and uid_after==uid_before,'Ray physical executable or UID changed during original capture')
 return dict(pid=pid,start_ticks=before,uid=uid_before,executable=dict(resolved_path=str(exe),device=key[1],inode=key[2],sha256=exe_cache[key]),cmdline_base64=base64.b64encode(raw).decode(),command=raw.decode().split('\0'),process_markers=markers,captured_monotonic_ns=time.monotonic_ns())
def production_watch(run,source):
 run=pathlib.Path(run);job=os.environ.get('SLURM_JOB_ID');need(sys.platform=='linux'and source==SOURCE and job and job.isdecimal()and run==pathlib.Path(ROOT)/'runs'/('bapo-instruct300-hardware-'+job),'Exact actual hardware sampler identity')
 need(os.environ.get('BAPO_RUN_ID')==str(run)and os.environ.get('SPT_NOENV')=='1'and pathlib.Path(sys.prefix)==pathlib.Path(ROOT)/'env','Actual task/runtime env required')
 tools=pathlib.Path(ROOT)/'tools';manifest=ordinary(pathlib.Path(ROOT)/'logs/tools-manifest.json');m=json.loads(manifest.read_text());need(m['source_commit']==SOURCE and m['files'][pathlib.Path(__file__).name]==sha(__file__),'Original sampler tool/source binding')
 spec=importlib.util.spec_from_file_location('production_ray_bound_watcher',tools/'resource_watch.py');w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w);anchor=w.load_runtime_anchor(str(run),verify_current=True)
 out=run/'production-ray-identity';out.mkdir(exist_ok=False);deadline=time.monotonic()+1200;cache={};ranks={};active={'ray::WorkerDict.actor_rollout_load_training_state','ray::WorkerDict.actor_rollout_save_training_state'}
 expected={key:os.environ[key]for key in MARKER_KEYS};executable=dict(anchor['executable'],sha256=sha(anchor['executable']['resolved_path']))
 while len(ranks)<8 or any('post_v3'not in v for v in ranks.values()):
  need(time.monotonic()<deadline,'Actual eight Ray identity capture window missing')
  for rank in range(8):
   witness=run/'recovery-witness'/('rank_%d.json'%rank)
   if not witness.exists():continue
   try:record=json.loads(witness.read_text())
   except json.JSONDecodeError:continue
   pid=record['new_pid'];need(type(pid)is int and pid>0 and record['rank']==rank and record['world_size']==8,'Actual restored rank PID differs')
   row=ranks.setdefault(rank,dict(rank=rank,witness_path=str(witness),witness_sha256=sha(witness),snapshots=[]))
   need(row['witness_sha256']==sha(witness),'Restored witness changed during capture')
   if 'post_v3'in row:continue
   snap=production_snapshot(pid,cache);need(snap['start_ticks']==record['process']['start_ticks']and snap['process_markers']['values']==expected and snap['uid']==os.getuid()and snap['executable']==executable,'Actual Ray original proc/runtime/epoch markers differ')
   raw=base64.b64decode(snap['cmdline_base64'],validate=True);title=snap['command'][0];need(title in w.RAY_APPROVED_TITLES and not any(snap['command'][1:])and len(title.encode())+1<=len(raw),'Actual Ray finite title/NUL capacity differs')
   if 'active'not in row and title in active:row['active']=snap;row['snapshots'].append(snap)
   v3=run/'recovery-native-v3-roundtrip'/('rank_%d.json'%rank)
   if v3.exists()and 'active'in row:
    try:report=json.loads(v3.read_text())
    except json.JSONDecodeError:continue
    need(report['pid']==pid and report['rank']==rank and report['world_size']==8 and report['status']=='EXACT_FULL_STATE_RELOAD_OK'and report['native_state_version']==3 and report['roundtrip']is True,'Actual native v3 report differs')
    # Capture AFTER the complete producer report was read, not the earlier poll snapshot.
    observed_v3=time.monotonic_ns();post=production_snapshot(pid,cache);need(post['start_ticks']==snap['start_ticks']and post['process_markers']==snap['process_markers']and post['executable']==executable,'Post-v3 original proc identity differs')
    row.update(v3_report_path=str(v3),v3_report_sha256=sha(v3),v3_report_observed_monotonic_ns=observed_v3,post_v3=post);row['snapshots'].append(post)
  time.sleep(.02)
 # Rehash the actual runtime binary at the completion boundary as well.
 need(sha(anchor['executable']['resolved_path'])==executable['sha256'],'Runtime binary changed across actual Ray capture')
 result=dict(status='COMPLETE_ACTUAL_PRODUCTION_RAY8_PROC_IDENTITY_NOT_GPU_ACCEPTANCE',root=ROOT,source_commit=SOURCE,run_root=str(run),job_id=int(job),tool_manifest_sha256=sha(manifest),sampler_sha256=sha(__file__),runtime_anchor=anchor,collector_uid=os.getuid(),expected_process_markers=expected,signals_sent=0,extra_ray_calls=0,extra_training_updates=0,ranks=[ranks[i]for i in range(8)])
 with (out/'result.json').open('x')as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps(dict(status=result['status'],path=str(out/'result.json'),sha256=sha(out/'result.json'))),flush=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--child',action='store_true');p.add_argument('--output');p.add_argument('--production-ray-watch',action='store_true');p.add_argument('--run');p.add_argument('--source');a=p.parse_args()
 if a.production_ray_watch:need(not a.child and a.output is None and a.run and a.source,'Exclusive production sampler mode');production_watch(a.run,a.source)
 elif a.child:need(a.output is None and a.run is None and a.source is None,'No child output path');child()
 else:need(a.output and a.run is None and a.source is None,'Actual output required');controller(a.output)
if __name__=='__main__':main()
