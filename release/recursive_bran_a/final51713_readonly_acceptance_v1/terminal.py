"""Read-only terminal processes, GPU applications and model metadata; no signals."""
import json, subprocess, time
from pathlib import Path
B=Path('/dataset/shaoshitong/recursive-bran-a-h100-20261012/recovery-from100-20261013')
C=B/'final51713-batch1024-v1'
def load(p): return json.loads(p.read_text())
def proc(pid):
 p=Path('/proc')/str(pid)
 if not p.exists(): return dict(pid=pid,present=False)
 try:
  s=(p/'stat').read_text().rsplit(')',1)[1].split()
  return dict(pid=pid,present=True,uid=p.stat().st_uid,tick=int(s[19]),state=s[0])
 except FileNotFoundError:return dict(pid=pid,present=False)
plan=load(C/'execution-plan.json');terminal=load(C/'evaluation-actual-exit.json')
model=B/'final51713-tokenizer-recovery-v1/model-canonical-tokenizer'
files=[]
for p in sorted(model.iterdir()):
 if p.is_file():
  s=p.stat();files.append(dict(relative_path=p.name,metadata=dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)))
a=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
print(json.dumps(dict(time=time.time(),run_id=plan['run_id'],actual_exit=terminal,processes=[proc(i) for i in [1500486,1512810,1513109,1513110,1513111,1513112,1513113,1513114,1513115]],gpu_apps=dict(rc=a.returncode,stdout=a.stdout,stderr=a.stderr),model_files=files,oracle=proc(696778),scope='Read-only process identity and current GPU applications; no signals or resource release')))
