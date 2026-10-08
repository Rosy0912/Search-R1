"""Task-owned metadata and code transfer only; never import or execute candidates."""
import base64,hashlib,json,shlex,subprocess,sys
from pathlib import Path
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control')
import control
base=Path.cwd();root='/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct200-safety-20261006'
snap=base/'outputs/fresh200-compatibility-20261006/FRESH200-SOURCE-STATIC-SNAPSHOT-v11-cpufixturefix1.json'
manifest=base/'outputs/gigpo-e1-policy-safety/FRESH200-CONTINUATION-STATIC-MANIFEST-v12-cpufixturefix1.json'
s=json.loads(snap.read_text());m=json.loads(manifest.read_text());source=s['canonical_future_source_repo'];runtime=m['canonical_future_runtime_root']
assert s['new_code_commit']=='2ddbb0a1619ce7cde28b4c5be110ffe5bd11c963' and s['full_tracked_source_file_count']==494 and m['file_count']==44
items=[]
def add(local,target):
 raw=Path(local).read_bytes();items.append(dict(path=target,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),b64=base64.b64encode(raw).decode()))
for row in m['files']:
 local=Path(m['runtime_root'])/row['relative_path'];raw=local.read_bytes();assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'];add(local,runtime+'/'+row['relative_path'])
for local,target in [(snap,root+'/logs/'+snap.name),(manifest,root+'/logs/'+manifest.name),(base/'outputs/gigpo-e1-instruct200/PREEXEC-REVIEW-TARGET-v3.md',root+'/logs/PREEXEC-REVIEW-TARGET-v3.md'),(base/'work/continuous_cpu_once_v4.py',root+'/tools/management/continuous-cpu-once-v4.py'),(base/'outputs/gigpo-e1-instruct200/CONTINUOUS-CPU-MANAGEMENT-FREEZE-v4.json',root+'/logs/CONTINUOUS-CPU-MANAGEMENT-FREEZE-v4.json'),(base/'work/gigpo-e1-continuous-2ddbb0a.bundle',root+'/logs/gigpo-e1-continuous-2ddbb0a.bundle')]:add(local,target)
payload=dict(root=root,source=source,runtime=runtime,parent_source=root+'/repo-v11-4a45458',commit=s['new_code_commit'],branch=s['branch'],source_rows=s['full_tracked_source_files'],runtime_rows=m['files'],items=items)
code='''import base64,datetime,hashlib,json,os,subprocess,sys
from pathlib import Path
p=json.load(sys.stdin);root=Path(p['root']);source=Path(p['source']);runtime=Path(p['runtime'])
def file_record(path):
 raw=path.read_bytes();return dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
records=[]
for row in p['items']:
 path=Path(row['path']);assert path.is_relative_to(root) and not path.is_symlink();path.parent.mkdir(parents=True,exist_ok=True);assert path.parent.resolve()==path.parent
 raw=base64.b64decode(row['b64']);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 if path.exists():assert path.read_bytes()==raw
 else:
  with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 records.append(file_record(path))
git=['git','-c','core.hooksPath=/dev/null','-c','core.fsmonitor=false']
if not source.exists():
 subprocess.run(git+['clone','--no-hardlinks',p['parent_source'],str(source)],check=True,capture_output=True)
 subprocess.run(git+['-C',str(source),'fetch',str(root/'logs/gigpo-e1-continuous-2ddbb0a.bundle'),p['branch']],check=True,capture_output=True)
 subprocess.run(git+['-C',str(source),'checkout','--detach',p['commit']],check=True,capture_output=True)
assert source.resolve()==source and not source.is_symlink()
assert subprocess.check_output(git+['-C',str(source),'rev-parse','HEAD'],text=True).strip()==p['commit']
assert not subprocess.check_output(git+['-C',str(source),'status','--porcelain'],text=True).strip()
tracked=subprocess.check_output(git+['-C',str(source),'ls-files','-z']).decode().split(chr(0))[:-1]
assert sorted(tracked)==sorted(row['relative_path'] for row in p['source_rows'])
closures={}
for name,directory,rows in [('source',source,p['source_rows']),('runtime',runtime,p['runtime_rows'])]:
 names=[]
 for parent,dirs,files in os.walk(directory,followlinks=False):
  parent=Path(parent)
  if name=='source' and parent==directory:dirs[:]=[d for d in dirs if d!='.git']
  for d in dirs:assert not (parent/d).is_symlink()
  for f in files:
   file=parent/f;assert file.is_file() and not file.is_symlink();names.append(str(file.relative_to(directory)))
 assert sorted(names)==sorted(row['relative_path'] for row in rows)
 actual=[]
 for row in rows:
  rec=file_record(directory/row['relative_path']);assert rec['bytes']==row['bytes'] and rec['sha256']==row['sha256'];actual.append(rec)
 closures[name]=dict(directory=str(directory),actual_file_count=len(actual),full_actual_records=actual,exact_inventory=True)
result=dict(schema='FRESH200_CONTINUOUS_STAGING_V4_METADATA_ONLY',status='EXACT_METADATA_ONLY_STAGED_NOT_EXECUTED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),commit=p['commit'],metadata_original_records=records,closures=closures,CPU_executed=False,GPU_executed=False,READY=False,submitted=False,formal_iterations=0)
print(json.dumps(result,sort_keys=True,indent=2))
'''
inner='/usr/bin/python3 -c '+shlex.quote(code)
mid='env -u LD_LIBRARY_PATH ssh -T -o ConnectTimeout=15 h100-8 '+shlex.quote(inner)
route=control.ssh('mgmt-host')+['env -u LD_LIBRARY_PATH ssh -T -o ConnectTimeout=15 h100-16 '+shlex.quote(mid)]
r=subprocess.run(route,input=json.dumps(payload).encode(),capture_output=True,timeout=120)
if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace'))
x=json.loads(r.stdout);out=base/'outputs/gigpo-e1-instruct200/FRESH200-CONTINUOUS-STAGING-RECEIPT-v4.json'
with out.open('x') as f:json.dump(x,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps(dict(status=x['status'],UTC=x['UTC'],source=source,runtime=runtime,source_files=x['closures']['source']['actual_file_count'],runtime_files=x['closures']['runtime']['actual_file_count'],receipt=str(out),CPU_executed=False,GPU_executed=False,submitted=False)))
