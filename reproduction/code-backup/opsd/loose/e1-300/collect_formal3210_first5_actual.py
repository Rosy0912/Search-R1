"""Read original small first-five records; never sign, release or alter training."""
import base64, datetime, hashlib, json, pathlib, shlex, subprocess, sys
HERE = pathlib.Path(__file__).resolve().parent
code = r'''import pathlib,json,hashlib,subprocess,os,socket,base64
root=pathlib.Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004');run=root/'runs/formal-3210';repo=root/'repo-a0968c2d24be'
cmd=lambda a:subprocess.check_output(a,text=True,timeout=20)
out=dict(host=socket.gethostname(),uid=os.getuid(),job_id='3210',slurm=cmd(['scontrol','show','job','3210','-o']),source_head=cmd(['git','-C',str(repo),'rev-parse','HEAD']).strip(),source_status=cmd(['git','-C',str(repo),'status','--porcelain']),gpu_inventory=cmd(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits']),gpu_compute=cmd(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits']),files={},processes={})
paths=[run/'train.log',run/'full_state/global_step_5/manifest.json',run/'full_state/global_step_5/driver.json',run/'full_state/global_step_5/resolved-config.json',run/'checkpoints/algorithm/global_step_5/state.json',run/'approval-ready.json',run/'bindings/manifest.json']
paths+=list((run/'checkpoints/actor/global_step_5/training_state').glob('*.json'))
paths+=list((run/'evidence').glob('*.json'))
paths+=list((run/'evaluation').glob('*.json'))
for p in paths:
 if p.is_file() and p.stat().st_size<32*1024**2:
  raw=p.read_bytes();s=p.stat();assert p.resolve()==p and s.st_uid==0
  out['files'][str(p)]=dict(base64=base64.b64encode(raw).decode(),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),uid=s.st_uid)
pidset={591546}
for p in (run/'checkpoints/actor/global_step_5/training_state').glob('*.json'):
 v=json.loads(p.read_bytes());pidset.add(int(v['worker_pid'] if 'worker_pid' in v else v['pid']))
for pid in sorted(pidset):
 p=pathlib.Path('/proc')/str(pid)
 if p.exists():out['processes'][str(pid)]=dict(start_ticks=(p/'stat').read_text().rsplit(')',1)[1].split()[19],status=(p/'status').read_text(),cgroup=(p/'cgroup').read_text(),cmdline=(p/'cmdline').read_bytes().decode().replace('\0',' '))
print(json.dumps(out,sort_keys=True))
'''
r = subprocess.run([sys.executable, str(HERE.parent/'remote.py'), 'python3 -c '+shlex.quote(code)],capture_output=True,check=True,timeout=80)
v=json.loads(r.stdout);stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');dest=HERE/('formal3210-first5-observed-'+stamp);dest.mkdir()
(dest/'collection-original.json').write_bytes(r.stdout);(dest/'collection-stderr.txt').write_bytes(r.stderr)
for n, rec in v['files'].items():
 raw=base64.b64decode(rec['base64']);assert hashlib.sha256(raw).hexdigest()==rec['sha256'];relative=pathlib.PurePosixPath(n).relative_to('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004/runs/formal-3210');p=dest/'run'/str(relative);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
print(json.dumps(dict(directory=str(dest),source=v['source_head'],source_status=v['source_status'],slurm=v['slurm'],files=len(v['files']),manifest_present=any(n.endswith('/full_state/global_step_5/manifest.json') for n in v['files']),gpu=v['gpu_inventory']),indent=2))
