"""Finite actual Linux StageScopeV7 process census stress; no GPU or model."""
import argparse,hashlib,importlib.util,json,os,pathlib,subprocess,threading,time,traceback
PINS={'stage_cleanup.py':'cc55f32fbb7eb65f6d4f13ed5b9a58d54bbb73ee25ae4cd6918a0d68dc9b3a80','owner_support.py':'d4de70880869abbec64f81ccd86f6f8dd8755b01d9db2cb93e2ec112d3031a09'}
def load(directory,name):
 p=pathlib.Path(directory)/name
 if p.is_symlink() or hashlib.sha256(p.read_bytes()).hexdigest()!=PINS[name]:raise ValueError('Deployed source drift '+name)
 spec=importlib.util.spec_from_file_location('actual_stage_'+p.stem,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 if pathlib.Path(m.__file__).resolve()!=p.resolve():raise ValueError('Actual module identity drift')
 return m

def durable(path,record):
 fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
 with os.fdopen(fd,'w') as f:json.dump(record,f,indent=2);f.flush();os.fsync(f.fileno())

def run(producer,out):
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CVD must be empty')
 job=os.environ['SLURM_JOB_ID'];out=pathlib.Path(out);out.mkdir(exist_ok=False)
 tools=load(producer,'owner_support.py');stage=load(producer,'stage_cleanup.py');scope,relative=tools.cgroup_scope(job)
 tracker=stage.StageScope(tools,scope,relative,event_path=out/'stage-registration-events.jsonl')
 record={'scope':'FINITE_NATIVE_LINUX_STAGE_PROCESS_CENSUS_ONLY_NO_GPU_MODEL_CALIBRATION','status':'RUNNING','job':job,'pid':os.getpid(),'source_pins':PINS,'producer_directory':str(pathlib.Path(producer).resolve()),'start_epoch':time.time(),'actual_cgroup':pathlib.Path('/proc/self/cgroup').read_text(),'actual_start_stat':pathlib.Path('/proc/self/stat').read_text(),'rounds':[],'total_spawned':0,'first_error':None}
 children=[];reapers=[];begin=time.monotonic()
 try:
  tracker.register(os.getpid())
  for ri in range(12):
   if time.monotonic()-begin>90:raise TimeoutError('Finite90sec stress budget')
   current=[]
   for index in range(40):
    child=subprocess.Popen(['/bin/true'],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);children.append(child);current.append(child);record['total_spawned']+=1
    reaper=threading.Thread(target=child.wait);reaper.start();reapers.append(reaper)
    tracker.pids(scope)
   for t in reapers:t.join(timeout=2)
   if any(t.is_alive() for t in reapers):raise RuntimeError('Own reaper incomplete')
   actual=tracker.pids(scope);remaining=actual-{os.getpid()}
   if remaining:raise RuntimeError('Stage descendants remain after normal own reap '+str(sorted(remaining)))
   row={'round':ri,'spawned':len(current),'all_child_returncodes':[p.returncode for p in current],'active_stage_descendants':sorted(remaining),'epoch':time.time()}
   if any(p.returncode!=0 for p in current):raise RuntimeError('Own /bin/true child did not exit0')
   record['rounds'].append(row);durable(out/('round-'+str(ri)+'.json'),row)
  record['status']='ACTUAL_CPU_CENSUS_STRESS_COMPLETE_NOT_CALIBRATION_PASS'
 except BaseException as exc:
  record['status']='ACTUAL_CPU_CENSUS_STRESS_FAILED';record['first_error']={'epoch':time.time(),'error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()}
  durable(out/'first-error-before-cleanup.json',record['first_error'])
 finally:
  # Only Popen objects created here are touched. No job-wide or foreign signals.
  for p in children:
   if p.poll() is None:p.terminate()
  for p in children:
   try:p.wait(timeout=2)
   except subprocess.TimeoutExpired:p.kill();p.wait(timeout=2)
  for t in reapers:t.join(timeout=2)
  record['own_children_remaining']=[p.pid for p in children if p.poll() is None]
  record['registration_roots']=tracker.roots;record['historical_ledger']=tracker.ledger;record['events_count']=len(tracker.events);tracker.close()
  record['completed_epoch']=time.time();record['elapsed_seconds']=time.monotonic()-begin;record['parent_cuda_initialized']=False;record['gpu_or_model_imported']=any(n.startswith(('torch','vllm','transformers')) for n in __import__('sys').modules)
  record['raw_events_sha256']=hashlib.sha256((out/'stage-registration-events.jsonl').read_bytes()).hexdigest();durable(out/'result.json',record)
 return record
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--producer',required=True);p.add_argument('--output',required=True);a=p.parse_args();r=run(a.producer,a.output);print(json.dumps(r));raise SystemExit(0 if r['status']=='ACTUAL_CPU_CENSUS_STRESS_COMPLETE_NOT_CALIBRATION_PASS' else 1)
