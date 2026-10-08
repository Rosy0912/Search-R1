"""Read-only actual job3198 evidence collection. No signing/GPU/remote writes."""
import json,pathlib,shlex,subprocess,sys
HERE=pathlib.Path(__file__).resolve().parent
code=r'''import pathlib,json,hashlib,subprocess,os,re,socket,base64
root=pathlib.Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004');run=root/'runs/formal-3198';repo=root/'repo-89beac273f78';tools=root/'tools/formal-209513e681cd'
cmd=lambda a:subprocess.check_output(a,text=True,timeout=20)
result=dict(host=socket.gethostname(),uid=os.getuid(),job_id='3198',slurm=cmd(['scontrol','show','job','3198','-o']),source_head=cmd(['git','-C',str(repo),'rev-parse','HEAD']).strip(),source_status=cmd(['git','-C',str(repo),'status','--porcelain']),gpu_inventory=cmd(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits']),gpu_compute=cmd(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits']),ready_exists=(run/'approval-ready.json').is_file(),files={})
paths=[tools/'manifest.json',run/'approval-ready.json',run/'formal-wrapper.log',root/'logs/formal-3198.out',root/'logs/formal-3198.err']
paths+=list((run/'bindings').glob('*'))+list((run/'evidence').glob('pre-gpu*'))
if result['ready_exists']:
 ready=json.loads((run/'approval-ready.json').read_bytes());p=pathlib.Path('/proc')/str(ready['pid']);status=(p/'status').read_text()
 result['actual_ready_process']=dict(pid=ready['pid'],start=(p/'stat').read_text().rsplit(')',1)[1].split()[19],cgroup=(p/'cgroup').read_text(),status=status,cmdline=(p/'cmdline').read_bytes().decode().replace('\0',' '))
 result['source_file_sha256']={n:hashlib.sha256((repo/n).read_bytes()).hexdigest() for n in json.loads((tools/'manifest.json').read_bytes())['source_files']}
 binding=json.loads((run/'bindings/manifest.json').read_bytes())
 for record in binding['files'].values():
  q=pathlib.Path(record['path']);assert q.resolve()==q and q.is_file() and q.stat().st_uid==0
  assert hashlib.sha256(q.read_bytes()).hexdigest()==record['sha256']
 result['current_sources_verified_binding_sha_files']=True
for p in paths:
 if p.is_file():
  raw=p.read_bytes();s=p.stat();result['files'][str(p)]=dict(base64=base64.b64encode(raw).decode(),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),uid=s.st_uid,physical=(p.resolve()==p))
print(json.dumps(result,sort_keys=True))
'''
args=[sys.executable,str(HERE.parent/'remote.py'),'python3 -c '+shlex.quote(code)]
r=subprocess.run(args,capture_output=True,check=True,timeout=80)
(HERE/'formal3198-pre-gpu-actual-collection.json').write_bytes(r.stdout)
(HERE/'formal3198-pre-gpu-collection-stderr.txt').write_bytes(r.stderr)
v=json.loads(r.stdout);print(json.dumps({k:v[k] for k in ['ready_exists','source_head','source_status','gpu_compute','slurm']},indent=2))
