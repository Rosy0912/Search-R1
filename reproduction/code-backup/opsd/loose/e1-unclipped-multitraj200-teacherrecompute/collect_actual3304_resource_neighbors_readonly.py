"""Fixed actual3304 resource exit-race originals only; no remote mutation."""
from pathlib import Path
import base64,datetime,gzip,hashlib,json,os,shlex,subprocess,types
H=Path(__file__).resolve().parent
TRANSPORT_SHA='a64bf52a97cbef6e3b0cc7948e095546687c8f1690094469c4937180f9449a85'
REMOTE=r'''import pathlib,os,stat,json,hashlib,base64,gzip,datetime,subprocess
R=pathlib.Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-unclipped-multitraj200-20261006/runs/formal-3304')
N=100995983;PIN='c0a6dccaec7502045c468278e05c6f0de1a69038188d35e5b7b986a75b86f0fb'
def identity(st):return dict(device=st.st_dev,inode=st.st_ino,uid=st.st_uid,bytes=st.st_size,mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns)
def read_file(p,limit,prefix=None):
 assert p.resolve(strict=True)==p
 before=p.lstat();assert stat.S_ISREG(before.st_mode) and before.st_uid==os.getuid()==0
 fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
 try:
  opened=os.fstat(fd);assert (opened.st_dev,opened.st_ino,opened.st_uid)==(before.st_dev,before.st_ino,before.st_uid)
  amount=opened.st_size if prefix is None else prefix
  assert 0<amount<=limit and opened.st_size>=amount
  chunks=[];remaining=amount
  while remaining:
   b=os.read(fd,min(1024*1024,remaining));assert b;chunks.append(b);remaining-=len(b)
  raw=b''.join(chunks);after=os.fstat(fd);path_after=p.lstat()
  if prefix is None:assert identity(opened)==identity(after)==identity(path_after)
  else:assert (opened.st_dev,opened.st_ino,opened.st_uid)==(after.st_dev,after.st_ino,after.st_uid)==(path_after.st_dev,path_after.st_ino,path_after.st_uid) and after.st_size>=amount and path_after.st_size>=amount
  return raw,dict(physical_path=str(p),before=identity(before),opened=identity(opened),after=identity(after),path_after=identity(path_after),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),scope='INFLIGHT_PREFIX_AT_FIXED_HISTORICAL_LENGTH' if prefix else 'STABLE_WHOLE_SMALL_FILE')
 finally:os.close(fd)
raw,meta=read_file(R/'resources.ndjson',512*1024*1024,N);assert hashlib.sha256(raw).hexdigest()==PIN and raw.endswith(b'\n')
lines=raw.splitlines(keepends=True);offsets=[];rows=[];offset=0;bad=[]
for i,line in enumerate(lines):
 row=json.loads(line);assert row['job_id']=='3304';rows.append(row);offsets.append(offset);offset+=len(line)
 if any(a.get('unavailable') or a.get('in_slurm_job') is not True for a in row.get('gpu_processes',[])):bad.append(i)
assert len(bad)==1
index=bad[0];assert rows[index]['time']=='2026-10-06T11:47:26.494869+00:00'
def app(row):return [a for a in row.get('gpu_processes',[]) if a.get('pid')==2544824]
problem=app(rows[index]);assert len(problem)==1 and problem[0]['process_start_ticks']=='1834772278' and problem[0]['in_slurm_job'] is False
prior=[i for i in range(index) if app(rows[i]) and app(rows[i])[0].get('in_slurm_job') is True and app(rows[i])[0]['process_start_ticks']=='1834772278']
later=[i for i in range(index+1,len(rows)) if not app(rows[i]) and '2544824' not in rows[i].get('nvml_apps_original','')]
assert prior and later
wanted={'adjacent_before':index-1,'problem':index,'adjacent_after':index+1,'latest_prior_owned':prior[-1],'first_later_nvml_absent':later[0]}
files={};selection={}
for name,i in wanted.items():
 b=lines[i];selection[name]=dict(line_index_zero_based=i,line_number=i+1,offset=offsets[i],end_offset=offsets[i]+len(b),time=rows[i]['time'],bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),target_pid_records=app(rows[i]))
 files['resource-'+name+'.ndjson']=dict(base64=base64.b64encode(gzip.compress(b,mtime=0)).decode(),gzip=True,bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
for name in ['evidence/pre-training-double15-after-capacity.json','evidence/actual-training-wrapper-environment.json','evidence/capacity-micro4/exit.json','evidence/capacity-micro4/micro-4-rank-5.json']:
 b,m=read_file(R/name,4*1024*1024);files[name]=dict(base64=base64.b64encode(b).decode(),gzip=False,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),physical=m)
proc={}
for name in ['status','stat','cgroup']:
 try:proc[name]={'exists':True,'text':(pathlib.Path('/proc/2544824')/name).read_text()}
 except OSError as e:proc[name]={'exists':False,'errno':e.errno,'type':type(e).__name__}
commands={}
for name,argv in [('nvml_apps',['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits']),('scontrol',['scontrol','show','job','-o','3304'])]:
 q=subprocess.run(argv,capture_output=True,text=True,timeout=15);commands[name]=dict(argv=argv,returncode=q.returncode,stdout=q.stdout,stderr=q.stderr)
print(json.dumps(dict(kind='BOUNDED_ACTUAL3304_HISTORICAL_RESOURCE_EXIT_NEIGHBORS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),job_id='3304',run_root=str(R),prefix=meta,samples=len(rows),problem_samples=len(bad),selection=selection,files=files,current_target_proc=proc,commands=commands,scope='Read-only raw bounded historical evidence and current proc/NVML observation; no snapshot normalization, acceptance, GPU work or remote mutation'),sort_keys=True))
'''
def main():
 raw=(H/'transport.py').read_bytes();assert hashlib.sha256(raw).hexdigest()==TRANSPORT_SHA
 m=types.ModuleType('own_reviewed_transport');m.__file__=str(H/'transport.py');exec(compile(raw,m.__file__,'exec'),m.__dict__)
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out=H/('actual3304-resource-neighbors-'+stamp);out.mkdir(mode=0o700)
 (out/'remote-code.original.py').write_text(REMOTE)
 try:q=m.remote('python3 -c '+shlex.quote(REMOTE),timeout=60)
 except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as e:
  (out/'transport.stdout.original').write_bytes(e.stdout or b'');(out/'transport.stderr.original').write_bytes(e.stderr or b'');raise
 (out/'transport.stdout.original').write_bytes(q.stdout);(out/'transport.stderr.original').write_bytes(q.stderr)
 d=json.loads(q.stdout)
 for name,entry in d['files'].items():
  b=base64.b64decode(entry['base64'],validate=True);b=gzip.decompress(b) if entry['gzip'] else b
  assert len(b)==entry['bytes'] and hashlib.sha256(b).hexdigest()==entry['sha256'];p=out/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
 observation={k:v for k,v in d.items() if k!='files'};observation['files']={k:{x:y for x,y in v.items() if x!='base64'} for k,v in d['files'].items()}
 (out/'observation.json').write_text(json.dumps(observation,indent=2)+'\n')
 execution=dict(exit_code=q.returncode,transport_sha256=TRANSPORT_SHA,remote_code_sha256=hashlib.sha256(REMOTE.encode()).hexdigest(),observer_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),stdout_sha256=hashlib.sha256(q.stdout).hexdigest(),stderr_sha256=hashlib.sha256(q.stderr).hexdigest(),scope='ACTUAL_READ_ONLY_REMOTE_EXECUTION_NO_GPU_OR_MUTATION')
 (out/'execution.original.json').write_text(json.dumps(execution,indent=2)+'\n')
 print(json.dumps(dict(directory=str(out),samples=d['samples'],problem_samples=d['problem_samples'],selection={k:{x:y for x,y in v.items() if x!='target_pid_records'} for k,v in d['selection'].items()},small_files=list(d['files']),current_proc=d['current_target_proc'],command_returncodes={k:v['returncode'] for k,v in d['commands'].items()})))
if __name__=='__main__':main()
