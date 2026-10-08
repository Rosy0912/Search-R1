"""Copied owned-process and asset primitives; formal wrapper never runs capacity code."""
import csv,hashlib,io,json,math,os,re,subprocess
from pathlib import Path
OLD=Path('/mnt/data/shaoshitong/search-r1-rl/opsd')
ASSET_SOURCE='04f8cc5a777f8dfafbb9e48f877673c2c5751f33'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def unique(raw):
 def pairs(items):
  d={}
  for k,v in items:
   if k in d:raise ValueError('Duplicate key')
   d[k]=v
  return d
 return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in()).throw(ValueError('Nonfinite JSON')))
def physical(p,owned=False):
 p=Path(p)
 if not p.is_absolute() or p.resolve(strict=True)!=p or not p.is_file() or (owned and p.stat().st_uid!=os.getuid()):raise ValueError('Physical owned path required')
 return p
def pinned(p,pin):
 p=physical(p)
 if not re.fullmatch('[0-9a-f]{64}',pin) or sha(p)!=pin:raise ValueError('External SHA differs')
 return unique(p.read_bytes())
def emit(path,value):
 with Path(path).open('x') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False)
def command(args):return subprocess.check_output(args,text=True,timeout=20)
def own(pid):
 p=Path('/proc')/str(pid);c=(p/'cgroup').read_text();status=(p/'status').read_text()
 if not re.search(r'(?:^|/)job_'+re.escape(os.environ['SLURM_JOB_ID'])+r'(?:/|$)',c,re.M):raise ValueError('Foreign process cgroup')
 if {int(x) for x in re.search(r'^Uid:\s+(.*)',status,re.M).group(1).split()}!={os.getuid()}:raise ValueError('Foreign process UIDs')
 return dict(pid=pid,state=re.search(r'^State:\s+(\S)',status,re.M).group(1),start=(p/'stat').read_text().rsplit(')',1)[1].split()[19],cgroup=c,uid=os.getuid())
def same_process(record):
 try:
  current=own(record['pid']);return current['start']==record['start'] and current['state'] not in ('Z','X')
 except (OSError,ValueError):return False
def snapshot():
 inv=command(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'])
 apps=command(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'])
 rows=[dict(index=int(v[0]),uuid=v[1].strip(),name=v[2].strip(),used=float(v[3]),total=float(v[4]),util=float(v[5])) for v in csv.reader(io.StringIO(inv))]
 procs=[dict(uuid=v[0].strip(),pid=int(v[1]),memory=v[2].strip(),owner=own(int(v[1]))) for v in csv.reader(io.StringIO(apps)) if v]
 return dict(inventory=rows,applications=procs)
def validate_snapshot(value,expected_uuids=None,service_pid=None,idle=False):
 rows=value['inventory'];assert len(rows)==8 and len({r['uuid'] for r in rows})==8 and all('H100' in r['name'] for r in rows)
 if expected_uuids is not None:assert {r['uuid'] for r in rows}==set(expected_uuids)
 if service_pid is None:assert not value['applications']
 else:
  assert {r['pid'] for r in value['applications']}=={service_pid}
  assert {r['uuid'] for r in value['applications']}==set(expected_uuids)
 if idle:assert all(r['used']<=.15*r['total'] and r['util']<=15 for r in rows)
def verify_tool_assets(manifest):
 assert manifest['status']=='FULL_TOOL_ASSET_CONTENT_VERIFIED' and manifest['frozen_source_commit']==ASSET_SOURCE
 assert manifest['assets_root']==str(OLD/'assets-root/assets') and manifest['hostname']=='h100-10'
 r=manifest['results'];assert len(r['e5'])==8
 records=r['e5']+[r['wiki_gzip'],r['index'],r['corpus']]
 for v in records:
  p=physical(v['path']);assert OLD/'assets-root/assets' in p.parents
  before=p.stat();assert before.st_size==v['bytes'] and sha(p)==v['sha256']
  after=p.stat();assert (before.st_ino,before.st_dev,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_ino,after.st_dev,after.st_size,after.st_mtime_ns,after.st_ctime_ns)
 assert r['corpus']['sha256']==r['gzip_decompressed']['sha256'] and r['corpus']['bytes']==r['gzip_decompressed']['bytes']
 assert len(r['index']['segment_sha256'])==2
 return records
