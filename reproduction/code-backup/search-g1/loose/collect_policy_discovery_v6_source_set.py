"""Read-only exact original source/Hydra metadata for failed native3294."""
import hashlib,json,os,shlex,subprocess,sys
from pathlib import Path
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control')
import control
prep=Path('/Users/shaoshitong/Documents/Search-G1/preparation')
raw=(prep/'policy-discovery-v5-native-launch.json').read_bytes()
assert hashlib.sha256(raw).hexdigest()=='1d50b96ed51f916e21891c1db88af03cdc3dea035bdec8471aa66959877c041f'
launch=json.loads(raw)
assert launch['job_id']=='3294' and launch['base']=='/mnt/data/shaoshitong/g1d-a6933c85' and launch['parent_base']=='/mnt/data/shaoshitong/g1p-a6933c85'
assert launch['source_commit']=='e670dfacab4ff0146f7ea0ca60e3f49675a4d2f6' and len(launch['source_sha256'])==353
code=r"""
import hashlib,json,os,stat,sys,time
from pathlib import Path
p=json.load(sys.stdin);out=dict(namespaces=[])
for root in (p['base'],p['parent_base']):
 b=Path(root);repo=b/'repo';mapping=p['source_sha256'];expected=set(mapping)
 assert b.resolve()==b and repo.resolve()==repo
 names={str(x.relative_to(repo)) for x in repo.rglob('*') if x.is_file() and (x.suffix in ('.py','.sh','.yaml','.yml','.json') or str(x.relative_to(repo)) in ('verl/version/version','verl/single_controller/version/version')) and not {'.git','__pycache__','evidence'}.intersection(x.relative_to(repo).parts)}
 extra=[]
 for name in sorted(names-expected):
  path=repo/name;s=path.lstat();assert path.resolve()==path and stat.S_ISREG(s.st_mode) and not path.is_symlink() and s.st_uid==os.getuid() and s.st_size<=131072
  fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
  try:
   with os.fdopen(os.dup(fd),'rb') as f:raw=f.read()
   t=os.fstat(fd);assert (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==(t.st_dev,t.st_ino,t.st_size,t.st_mtime_ns)
  finally:os.close(fd)
  extra.append(dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),mtime=s.st_mtime,symlink=False))
 missing=sorted(expected-names);changed=[]
 for name in sorted(expected&names):
  path=repo/name;assert path.resolve()==path and not path.is_symlink()
  if hashlib.sha256(path.read_bytes()).hexdigest()!=mapping[name]:changed.append(name)
 out['namespaces'].append(dict(base=root,expected_count=len(expected),actual_count=len(names),extra=extra,missing=missing,changed=changed))
out['checked_unix']=time.time();print(json.dumps(out))
"""
r=subprocess.run(control.route('h100-8','python3 -c '+shlex.quote(code)),input=json.dumps(launch),capture_output=True,text=True,timeout=55)
assert r.returncode==0,r.stderr[-1500:]
x=json.loads(r.stdout)
from submit_policy_discovery_v6_native import source_set_schema
source_set_schema(x)
assert x['checked_unix']>=1791256551.399966
raw=json.dumps(x,indent=2).encode();target=prep/('policy-discovery-v6-source-set-readonly-'+str(int(x['checked_unix']*1e6))+'.json')
fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
print(json.dumps(dict(path=str(target),sha256=hashlib.sha256(raw).hexdigest(),result=x),indent=2))
