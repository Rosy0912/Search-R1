"""Actual unreaped own-child Z / held-pidfd / cgroup lifecycle, CPU-only."""
import argparse,json,os,pathlib,select,subprocess,sys,time,traceback,hashlib

def durable(path,row):
 fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
 with os.fdopen(fd,'w') as f:json.dump(row,f,indent=2);f.flush();os.fsync(f.fileno())
def capture(pid,fd):
 p=pathlib.Path('/proc')/str(pid);row={'epoch':time.time(),'pid':pid,'held_fd':fd,'pidfd_poll':select.poll()}
 poll=row.pop('pidfd_poll');poll.register(fd,select.POLLIN);row['poll_events']=[list(v) for v in poll.poll(0)]
 row['fdinfo']=pathlib.Path('/proc/self/fdinfo',str(fd)).read_text()
 row['actual_pidfd_pid']=int(next(v.split(':',1)[1].strip() for v in row['fdinfo'].splitlines() if v.startswith('Pid:')))
 try:
  row['stat']=(p/'stat').read_text();f=row['stat'].rsplit(')',1)[1].split();row.update(state=f[0],ppid=int(f[1]),starttime_ticks=int(f[19]),uid=p.stat().st_uid,cgroup=(p/'cgroup').read_text().splitlines(),proc_absent=False)
 except FileNotFoundError:row['proc_absent']=True
 return row

def verify_exit(live,dead):
 if live['actual_pidfd_pid']!=live['pid'] or live['poll_events'] or live.get('state') in ('Z','X') or live.get('proc_absent'):raise ValueError('Not an actually live held descriptor baseline')
 for k in ('pid','starttime_ticks','uid'):
  if dead.get(k)!=live[k]:raise ValueError('Original child identity drift '+k)
 initial={v.split(':',2)[1]:v.split(':',2)[2] for v in live['cgroup']};final={v.split(':',2)[1]:v.split(':',2)[2] for v in dead['cgroup']}
 if initial.keys()!=final.keys() or any(final[k]!=initial[k] for k in initial if k!='freezer') or final.get('freezer') not in (initial.get('freezer'),'/'):raise ValueError('Non-freezer identity or unknown cgroup migration')
 if dead.get('state') not in ('Z','X') or dead['actual_pidfd_pid']!=live['pid'] or not any(v[1]&select.POLLIN for v in dead['poll_events']):raise ValueError('Actual death descriptor/state evidence missing')
 # Changes are observed, never rewritten. This probe does not approve their use
 # by a cleanup or GPU safety validator.
 return {'same_actual_PID_start_UID':True,'held_pidfd_POLLIN':True,'actual_state':dead['state'],'cgroup_changed':live['cgroup']!=dead['cgroup'],'initial_cgroup':live['cgroup'],'exit_cgroup':dead['cgroup']}

def run(out):
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU CVD must be empty')
 job=os.environ['SLURM_JOB_ID'];out=pathlib.Path(out);out.mkdir(exist_ok=False)
 parent=pathlib.Path('/proc/self/cgroup').read_text();assert '/job_'+job+'/' in parent or parent.rstrip().endswith('/job_'+job)
 result={'scope':'ACTUAL_UNREAPED_OWN_CHILD_Z_CPU_ONLY_NOT_STAGEV8_OR_GPU_ACCEPTANCE','job_id':job,'parent_pid':os.getpid(),'parent_cgroup':parent,'samples':[],'status':'RUNNING','first_error':None};children=[];fds=[]
 try:
  for index in range(8):
   # Pipe keeps the original process live until pidfd registration is measured.
   child=subprocess.Popen([sys.executable,'-c','import os,sys;sys.stdin.buffer.read(1);os.execl("/bin/true","true")'],stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);children.append(child)
   fd=os.pidfd_open(child.pid,0);fds.append(fd);live=capture(child.pid,fd);durable(out/f'child-{index}-live.json',live)
   child.stdin.write(b'x');child.stdin.flush();child.stdin.close()
   deadline=time.monotonic()+5;observed=[]
   # Do NOT child.poll()/wait before the Z observation: those would reap it.
   while time.monotonic()<deadline:
    row=capture(child.pid,fd);observed.append(row)
    if row.get('state') in ('Z','X') and any(v[1]&select.POLLIN for v in row['poll_events']):break
    time.sleep(.01)
   else:raise TimeoutError('No actual unreaped death state')
   dead=observed[-1];check=verify_exit(live,dead)
   time.sleep(.05);held=capture(child.pid,fd);held_check=verify_exit(live,held)
   rc=child.wait(timeout=2);gone=capture(child.pid,fd)
   if rc!=0 or not gone['proc_absent'] or gone['actual_pidfd_pid']!=-1 or not any(v[1]&select.POLLIN for v in gone['poll_events']) or not any(v[1]&select.POLLHUP for v in gone['poll_events']):raise ValueError('Normal reap / actual pidfd -1 mismatch')
   receipt={'index':index,'live':live,'observed':observed,'unreaped_held':held,'after_normal_reap':gone,'child_returncode':rc,'verified':check,'verified_unreaped_held':held_check};durable(out/f'child-{index}-receipt.json',receipt);result['samples'].append(receipt)
  if not any(v['verified']['cgroup_changed'] or v['verified_unreaped_held']['cgroup_changed'] for v in result['samples']):raise RuntimeError('FAIL_COVERAGE_NO_ACTUAL_FREEZER_DETACH')
  result['status']='ACTUAL_UNREAPED_Z_DESCRIPTOR_OBSERVATIONS_COMPLETE_NOT_STAGE_CALIBRATION_PASS'
 except BaseException as exc:
  result.update(status='ACTUAL_UNREAPED_Z_OBSERVATION_FAILED',first_error={'type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()});durable(out/'first-error-before-cleanup.json',result['first_error'])
 finally:
  for child in children:
   if child.stdin and not child.stdin.closed:child.stdin.close()
   try:child.wait(timeout=2)
   except subprocess.TimeoutExpired:child.terminate();child.wait(timeout=2)
  for fd in fds:os.close(fd)
  result['own_children_remaining']=[c.pid for c in children if c.poll() is None];result['source_sha256']=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest();result['GPU_or_model_imported']=any(n.startswith(('torch','vllm','transformers')) for n in sys.modules);durable(out/'result.json',result)
 return result
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--output',required=True);p=a.parse_args();r=run(p.output);print(json.dumps(r));raise SystemExit(0 if r['status'].startswith('ACTUAL_UNREAPED_Z_DESCRIPTOR_OBSERVATIONS_COMPLETE') else 1)
