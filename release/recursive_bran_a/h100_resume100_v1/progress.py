"""Read live isolated recovery stages, real rank updates, and terminal evidence."""
import json,time,subprocess
from pathlib import Path
b=Path('/dataset/shaoshitong/recursive-bran-a-h100-20261012/recovery-from100-20261013')
def read(p):
 try:return json.loads(p.read_text())
 except FileNotFoundError:return None
 except ValueError:return dict(incomplete_read=True)
def tail(p,n=40000):
 try:
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-n));return f.read().decode(errors='replace')
 except FileNotFoundError:return ''
entry=read(b/'receipts/train-start.json');proc=None
if entry and (Path('/proc')/str(entry['pid'])).exists():
 p=Path('/proc')/str(entry['pid']);s=(p/'stat').read_text().rsplit(')',1)[1].split();proc=dict(pid=entry['pid'],uid=p.stat().st_uid,tick=int(s[19]),state=s[0],argv=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace'))
log=tail(b/'logs/training.log',80000);stages=[];steps=[]
for l in log.splitlines():
 if 'TREE_DRIVER_STAGE=' in l:
  try:stages.append(json.JSONDecoder().raw_decode(l.split('TREE_DRIVER_STAGE=',1)[1])[0])
  except ValueError:pass
 if 'training/actual_outer_updates' in l and 'step:' in l:steps.append(l)
ranks={}
for p in sorted((b/'output/rank-audit/policy-safety').glob('*.json')):
 d=read(p)
 if d and 'rank' in d:ranks[str(d['rank'])]={k:d.get(k) for k in ('outer_step','phase','committed_optimizer_calls','decision','measured_gradient_preclip','measured_gradient_postclip')}
commands={}
for k,a in [('job',['scontrol','show','job','3393']),('gpu',['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'])]:
 p=subprocess.run(a,text=True,capture_output=True,timeout=15);commands[k]=dict(rc=p.returncode,stdout=p.stdout,stderr=p.stderr)
print(json.dumps(dict(time=time.time(),entry=entry,process=proc,stages=stages[-10:],complete_step_lines=steps[-5:],ranks=ranks,migration=read(b/'output/sparse-schedule-migrated.json'),configuration=read(b/'receipts/configuration.json'),actual_entry_exit=tail(b/'actual-entry-exit.txt'),configure_tail=tail(b/'logs/configure.log',2000),gate=read(b/'retriever/receipts/full-scan-http-gate.json'),retriever_tail=tail(b/'logs/retriever.log',2000),training_tail=log[-12000:],commands=commands)))
