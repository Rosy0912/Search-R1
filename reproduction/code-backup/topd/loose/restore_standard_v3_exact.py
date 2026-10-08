import json,pathlib,sys,base64,hashlib
sys.path.insert(0,'work/g1-native-cpu-deploy-v6')
from stream_transport import send
rows=json.loads(pathlib.Path('outputs/remote-evidence/standard-r1-profile3158-independent-source-set-diff.json').read_text())['rows']
code=r'''
import sys
sys.dont_write_bytecode=True
import json,pathlib,hashlib,os,subprocess,time
R=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd');rows=ROWS
out=R/'logs/r1-standard32b-profile3158-source-restoration.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
queue=subprocess.check_output(['squeue','-h','-w','h100-11','-o','%i|%j|%T'],text=True)
assert not any(line.split('|')[1]=='r1-32b-grpo' for line in queue.splitlines())
own=[]
for p in pathlib.Path('/proc').iterdir():
 if p.name.isdigit():
  try:
   cg=(p/'cgroup').read_text()
   if '/job_3158/' in cg or cg.rstrip().endswith('/job_3158'):own.append({'pid':int(p.name),'cgroup':cg})
  except (FileNotFoundError,ProcessLookupError):pass
assert own==[],own
archive=R/'logs/r1-standard32b-profile3158-generated-bytecode';archive.mkdir(mode=0o700,exist_ok=True)
for row in rows:
 root=pathlib.Path(row['root']);manifest=root/('r1-files.json' if root.name.startswith('repo-') else 'files.json');records=json.loads(manifest.read_text())['files']
 expected=set(records)|{manifest.name};actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
 assert actual-expected<=set(e['name'] for e in row['extra']) and not expected-actual
 for e in row['extra']:
  p=root/e['name'];d=archive/e['name'];d.parent.mkdir(parents=True,exist_ok=True)
  if p.exists():
   assert not p.is_symlink() and p.stat().st_size==e['size'] and sha(p)==e['sha256']
   assert not d.exists();os.rename(p,d)
  assert d.stat().st_size==e['size'] and sha(d)==e['sha256']
 assert {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}==expected
 for name,v in records.items():
  p=root/name;assert not p.is_symlink() and p.stat().st_size==v['size'] and sha(p)==v['sha256']
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
assert {int(line.split(',')[1]) for line in gpu.splitlines()}=={1042865,1042866,1042867,1052362,1052363,1052364,1042868,1060572}
report=dict(scope='ACTUAL_SOURCE_EXACT_RESTORATION_AFTER_FAILED3158_NOT_GPU_TRAINING',status='EXACT_RESTORED',epoch=time.time(),archive=str(archive),moved_bytecode=rows[1]['extra'],source_manifest_sha256=sha(pathlib.Path(rows[0]['root'])/'r1-files.json'),tools_manifest_sha256=sha(pathlib.Path(rows[1]['root'])/'files.json'),own3158_processes=own,queue=queue,foreign_compute_roster=gpu)
if out.exists():assert json.loads(out.read_text())['status']=='EXACT_RESTORED'
else:
 with out.open('x') as f:json.dump(report,f,indent=2);f.flush();os.fsync(f.fileno())
print(json.dumps({'report':json.loads(out.read_text()),'path':str(out),'sha256':sha(out)}))
'''.replace('ROWS',repr(rows))
r=send(code,timeout=180);pathlib.Path('outputs/standard-r1-profile3158-source-restoration-transport.log').write_bytes(r.stderr);assert r.returncode==0,r.stderr.decode();data=json.loads(r.stdout);b=json.dumps(data['report'],indent=2).encode();pathlib.Path('outputs/standard-r1-profile3158-source-restoration.json').write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data,indent=2))
