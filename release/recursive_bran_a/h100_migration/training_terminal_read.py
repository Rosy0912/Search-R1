"""Read actual training terminal, allocation, and saved checkpoint evidence only."""
import json
from pathlib import Path
import subprocess
import time
base=Path('/dataset/shaoshitong/recursive-bran-a-h100-20261012')
def tail(path,limit=90000):
    try:
        with path.open('rb') as f:f.seek(max(0,path.stat().st_size-limit));return f.read().decode('utf8',errors='replace')
    except FileNotFoundError:return None
def proc(pid):
    p=Path('/proc')/str(pid)
    try:
        s=(p/'stat').read_text().rsplit(')',1)[1].split();return dict(pid=pid,uid=p.stat().st_uid,state=s[0],tick=int(s[19]),ppid=int(s[1]))
    except FileNotFoundError:return dict(pid=pid,absent=True)
manifests=[]
for p in sorted((base/'checkpoints/training_state').glob('global_step_*/manifest.json')):
    d=json.loads(p.read_text());manifests.append(dict(path=str(p),status=d.get('status'),identity=d.get('identity'),hf_checkpoint=d.get('hf_checkpoint'),driver_next=d.get('driver',{}).get('next_step'),worker_files=[{k:v for k,v in w.items() if k!='identity'} for w in d.get('worker_states',[])]))
commands={}
for name,args in [('job',['scontrol','show','job','3389']),('step',['scontrol','show','step','3389.0']),('gpu',['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'])]:
    p=subprocess.run(args,capture_output=True,text=True,timeout=15);commands[name]=dict(rc=p.returncode,stdout=p.stdout,stderr=p.stderr)
print(json.dumps(dict(time=time.time(),processes=[proc(p) for p in [3714909,3721269,3715225,3713677,3713706]],training_log_tail=tail(base/'logs/training.log'),step_log_tail=tail(base/'logs/shared-h10011-step.log',12000),manifests=manifests,commands=commands)))
