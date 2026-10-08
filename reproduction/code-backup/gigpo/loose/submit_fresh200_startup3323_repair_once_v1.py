"""One exclusive fresh200 training submission after actual pretraining3323 failure.

May queue behind foreign jobs; no GPU admission occurs until the unchanged
allocated-run preflight, Slurm, UUID/OFD, dual15 and owned-process guards pass.
"""
import base64,hashlib,json,shlex,subprocess,sys,zlib
from pathlib import Path
if sys.flags.optimize or not sys.flags.no_site or not sys.flags.ignore_environment:
    raise RuntimeError('Use -E -B -S with active assertions')
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control');import control
BASE=Path.cwd();OUT=BASE/'outputs/gigpo-e1-instruct200'
ROOT='/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct200-safety-20261006'
STATE=ROOT+'/logs/fresh200-startuprepair3323-once-v1'
ENV={'PATH':'/usr/bin:/bin','LANG':'C','LC_ALL':'C','PYTHONDONTWRITEBYTECODE':'1'}
OPTIONS=['--parsable','--partition=gpu','--nodelist=h100-8','--nodes=1','--ntasks=1','--gres=gpu:h100:8','--cpus-per-task=128','--mem=512G','--exclusive','--no-requeue','--time=1-00:00:00','--export=ALL']
def sha(raw):return hashlib.sha256(raw).hexdigest()
def checked(path,row):
 raw=path.read_bytes()
 if len(raw)!=row['bytes'] or sha(raw)!=row['sha256']:raise RuntimeError('Frozen bytes differ: '+str(path))
 return raw

def main():
 dp=OUT/'FRESH200-STARTUP3323-REPAIR-DERIVATION-v1.json';d=json.loads(dp.read_bytes())
 bindingrow=d['binding_candidate'];bindingraw=checked(Path(bindingrow['path']),bindingrow)
 manifest=json.loads(checked(Path(d['runtime_manifest']['path']),d['runtime_manifest']))
 rp=OUT/'FRESH200-STARTUP3323-REPAIR-SINGLE-SUBMIT-INDEPENDENT-REVIEW-v1.json';rr=rp.read_bytes();review=json.loads(rr)
 if not (review['status']=='PASS' and review['independent_review'] is True and review['blocking_findings']==[]
         and review['scope']=='FRESH200_STARTUP3323_REPAIR_EXCLUSIVE_ONCE_SUBMISSION_ONLY'
         and review['binding_sha256']==bindingrow['sha256'] and review['derivation_sha256']==sha(dp.read_bytes())
         and review['submission_adapter_sha256']==sha(Path(__file__).read_bytes())
         and review['parent_job']=='3323' and review['parent_completed_outer']==0
         and review['parent_actual_Adam_writes_per_rank']==0 and review['repeat_CPU_suite_allowed'] is False
         and review['extra_validation_job_allowed'] is False and review['queued_submission_is_not_GPU_admission'] is True
         and review['first5_count_toward200'] is True and review['same_process_continuation_required'] is True):
  raise RuntimeError('Missing exact genuine code/submission review')
 staging=[]
 for row in manifest['files']:
  raw=checked(Path(manifest['runtime_root'])/row['relative_path'],row)
  staging.append(dict(path=manifest['canonical_future_runtime_root']+'/'+row['relative_path'],bytes=len(raw),sha256=sha(raw),raw=raw))
 for row in d['extra_metadata_staging_local_to_remote']:
  staging.append(dict(path=row['path'],bytes=row['bytes'],sha256=row['sha256'],raw=checked(Path(row['local_path']),row)))
 staging.append(dict(path=ROOT+'/logs/'+rp.name,bytes=len(rr),sha256=sha(rr),raw=rr))
 for row in staging:row['b64']=base64.b64encode(zlib.compress(row.pop('raw'))).decode()
 payload=dict(root=ROOT,state=STATE,environment=ENV,options=OPTIONS,staging=staging,binding=dict(path=STATE+'/BINDING.json',bytes=len(bindingraw),sha256=sha(bindingraw)),binding_b64=base64.b64encode(zlib.compress(bindingraw)).decode())
 code=r'''import base64,hashlib,json,os,socket,subprocess,sys,zlib
from pathlib import Path
if sys.flags.optimize or not sys.flags.no_site or not sys.flags.ignore_environment:raise RuntimeError('Unisolated receiver')
length=int(sys.argv[1])
if not 0<length<20000000:raise RuntimeError('Invalid frame length')
raw=sys.stdin.buffer.read(length)
if len(raw)!=length:raise RuntimeError('Incomplete exact frame')
p=json.loads(raw);root=Path(p['root']);state=Path(p['state'])
if socket.gethostname().split('.')[0]!='h100-8' or root!=Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct200-safety-20261006') or state!=root/'logs/fresh200-startuprepair3323-once-v1':raise RuntimeError('Wrong repair target')
if state.exists() or state.is_symlink():raise RuntimeError('Existing intent/namespace: NEVER REPLAY')
sys.stderr=(root/'logs/startuprepair3323-once-v1-error.log').open('x',buffering=1)
sys.stderr.write('OWNED_ONCE_SUBMIT_RECEIVER pid='+str(os.getpid())+'\n')
def checked(row,b64):
 raw=zlib.decompress(base64.b64decode(b64))
 if len(raw)!=row['bytes'] or hashlib.sha256(raw).hexdigest()!=row['sha256']:raise RuntimeError('Payload differs')
 return raw
for row in p['staging']:
 path=Path(row['path']);data=checked(row,row['b64'])
 if not path.is_relative_to(root/'tools/fresh200-continuation-v13-fsdpinit1') and path.parent!=root/'logs':raise RuntimeError('Staging escaped scope')
 if path.is_symlink():raise RuntimeError('Symlink target')
 path.parent.mkdir(parents=True,exist_ok=True)
 if path.parent.resolve()!=path.parent:raise RuntimeError('Noncanonical target')
 if path.exists():
  if path.read_bytes()!=data:raise RuntimeError('Existing frozen original differs')
 else:
  with path.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
bindingraw=checked(p['binding'],p['binding_b64']);binding=json.loads(bindingraw)
if binding['helpers']!=str(root/'tools/fresh200-continuation-v13-fsdpinit1') or binding['runtime_variant']!='fsdpinit1' or binding['startup_repair_parent_job']!='3323':raise RuntimeError('Different repair binding')
old=root/'logs/fresh200-startuprepair3318-once-v1';oldraw=(old/'SUBMIT-RECEIPT.json').read_bytes()
if hashlib.sha256(oldraw).hexdigest()!='ca6ec95564af2bb960d5c29b9ff40821de727523b77173ada849f2a6ce57682f' or json.loads(oldraw)['job_id']!='3323':raise RuntimeError('Parent3323 submit receipt differs')
for row in binding['runtime_files']+binding['evidence']:
 data=Path(row['path']).read_bytes()
 if len(data)!=row['bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']:raise RuntimeError('Frozen runtime/evidence changed')
fact=json.loads(Path(binding['startup_repair_terminal_factual_record']['path']).read_bytes())
if fact['status']!='ACTUAL_DRIVER_FAILED_BEFORE_TRAINING' or fact['job_id']!='3323' or fact['completed_outer']!=0 or fact['actual_Adam_writes_per_rank']!=0 or fact['cleanup_residual_states']!={}:raise RuntimeError('Parent is not actual zero-training failure')
row=fact['original_run_result'];parentraw=Path(row['path']).read_bytes()
if len(parentraw)!=row['bytes'] or hashlib.sha256(parentraw).hexdigest()!=row['sha256'] or json.loads(parentraw)['status']!='FAILED':raise RuntimeError('Original parent terminal changed')
for pid in fact['owned_pid_exists']:
 if Path('/proc/'+pid).exists():raise RuntimeError('Prior owned PID remains or has been reused; no submission')
if Path(json.loads(parentraw)['cgroup']).exists():raise RuntimeError('Prior owned cgroup remains')
observations={}
for name,cmd in [('queue',['squeue','-h','--all','-o','%i|%j|%T|%R']),('gpu',['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'])]:
 r=subprocess.run(cmd,capture_output=True,text=True,env=p['environment'],timeout=30)
 observations[name]=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
 if r.returncode:raise RuntimeError('Resource observation unreadable')
for line in observations['queue']['stdout'].splitlines():
 fields=line.split('|')
 if len(fields)!=4:raise RuntimeError('Unreadable scheduler row')
 if fields[1].startswith('gigpo-e1-fresh200-live'):raise RuntimeError('Prior own pending/running/completing job: NEVER REPLAY')
expected=['--parsable','--partition=gpu','--nodelist=h100-8','--nodes=1','--ntasks=1','--gres=gpu:h100:8','--cpus-per-task=128','--mem=512G','--exclusive','--no-requeue','--time=1-00:00:00','--export=ALL']
if p['options']!=expected or p['environment']!={'PATH':'/usr/bin:/bin','LANG':'C','LC_ALL':'C','PYTHONDONTWRITEBYTECODE':'1'}:raise RuntimeError('Wrong submission policy')
state.mkdir()
def once(path,data):
 with path.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
once(state/'BINDING.json',bindingraw)
intent=dict(status='CLAIMED_NO_RESUBMISSION',parent_job='3323',parent_outer=0,parent_Adam_writes=0,binding=p['binding'],resource_observations=observations,queued_submission_is_not_GPU_admission=True,allocation_UUID_OFD_dual15_owned_checks_retained=True,first5_count_toward200=True,repeat_CPU_suite=False,extra_validation_job=False)
once(state/'SUBMIT-INTENT.json',(json.dumps(intent,sort_keys=True,indent=2)+'\n').encode())
command=['sbatch']+expected+[binding['helpers']+'/calibration_v1.sbatch',str(state/'BINDING.json'),p['binding']['sha256'],binding['repo'],binding['helpers'],binding['environment_root']]
try:
 r=subprocess.run(command,capture_output=True,text=True,env=p['environment'],timeout=60);job=r.stdout.strip()
 result=dict(status='SUBMITTED_EXACTLY_ONCE' if r.returncode==0 and job.isdecimal() else 'UNCERTAIN_NO_RESUBMISSION',command=command,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,job_id=job if r.returncode==0 and job.isdecimal() else None,binding=p['binding'],parent_job='3323',parent_completed_outer=0,parent_Adam_writes_per_rank=0,training_updates_not_yet_observed=True)
except BaseException as error:
 result=dict(status='UNCERTAIN_NO_RESUBMISSION',error_type=type(error).__name__,error=str(error),binding=p['binding'],parent_job='3323')
once(state/'SUBMIT-RECEIPT.json',(json.dumps(result,sort_keys=True,indent=2)+'\n').encode())
print(json.dumps(result));raise SystemExit(0 if result['status']=='SUBMITTED_EXACTLY_ONCE' else 1)
'''
 serialized=json.dumps(payload).encode()
 inner='env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C /usr/bin/python3 -E -B -S -c '+shlex.quote(code)+' '+str(len(serialized))
 mid='env -u LD_LIBRARY_PATH ssh -T -o ConnectTimeout=15 h100-8 '+shlex.quote(inner)
 management=[arg.replace('ServerAliveInterval=15','ServerAliveInterval=120').replace('ServerAliveCountMax=2','ServerAliveCountMax=6') for arg in control.ssh('mgmt-host')]
 route=management+['env -u LD_LIBRARY_PATH ssh -T -o ConnectTimeout=15 -o HostName=10.33.0.16 -o HostKeyAlias=100.108.173.4 -o StrictHostKeyChecking=yes h100-16 '+shlex.quote(mid)]
 claim=OUT/'FRESH200-STARTUP3323-REPAIR-LOCAL-INVOKE-CLAIM-v1.json'
 with claim.open('x') as f:json.dump(dict(status='CLAIMED_LOCAL_ONE_INVOCATION_NEVER_REPLAY',binding_sha256=bindingrow['sha256'],review_sha256=sha(rr),framed_input_bytes=len(serialized)),f)
 try:
  done=subprocess.run(route,input=serialized,capture_output=True,timeout=240)
  result=dict(returncode=done.returncode,stdout=done.stdout.decode(errors='replace'),stderr=done.stderr.decode(errors='replace'))
 except subprocess.TimeoutExpired as error:
  result=dict(status='UNCERTAIN_NO_RESUBMISSION',error_type=type(error).__name__,stdout=(error.stdout or b'').decode(errors='replace'),stderr=(error.stderr or b'').decode(errors='replace'))
 out=OUT/'FRESH200-STARTUP3323-REPAIR-SINGLE-SUBMIT-INVOKE-v1.json'
 with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result))
 if result.get('returncode')!=0:raise RuntimeError('Inspect durable remote intent/receipt/queue; NEVER REPLAY')
if __name__=='__main__':main()
