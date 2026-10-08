import json,hashlib,time,subprocess
from pathlib import Path
r=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-recovery300-3234');files={}
def put(p,text=None):
 raw=p.read_bytes();files[str(p)]=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),text=raw.decode(errors='replace') if text is None else text)
for n in ['run-result.json','cost.json','termination-cost.json','launch-manifest.json','resolved.yaml','coexistence.json','resource-guard.log']:
 p=r/n
 if p.exists():put(p)
p=r/'stage.log';raw=p.read_bytes();lines=raw.decode(errors='replace').splitlines();files[str(p)]=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),tail='\n'.join(lines[-240:]),completed_lines=[s for s in lines if 'training/actual_outer_updates:' in s],failure_lines=[s for s in lines if any(t in s for t in ['Error','Traceback','refused','quality','threshold','gradient','below'])][-80:])
p=r/'resource-samples.jsonl'
if p.exists():
 raw=p.read_bytes();samples=[json.loads(s) for s in raw.splitlines()];files[str(p)]=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),samples=len(samples),events=[x for x in samples if 'event' in x],last=samples[-2:])
counts={}
for s in range(122,130):
 o=r/'output';counts[str(s)]=dict(ranks=len(list((o/'tree-rank-audit').glob(f'step-{s:06d}-rank-*.json'))),profiles=len(list((o/'tree-rank-audit/memory-profile').glob(f'step-{s:06d}-rank-*.json'))),ratios=len(list((o/'tree-rank-audit/ratio-tail').glob(f'step-{s:06d}-rank-*-optimizer-*.json'))),tree=len(list((o/'tree-audit').glob(f'step_{s}_rollout_*.json'))),trajectory=(o/f'trajectories/outer-{s:06d}/manifest.json').exists())
failures=[]
for p in (r/'output/tree-rank-audit').rglob('*failure*'):
 if p.is_file():failures.append(dict(path=str(p),bytes=p.stat().st_size))
s=subprocess.run(['scontrol','show','job','3234'],capture_output=True,text=True);gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],capture_output=True,text=True)
ps=subprocess.run(['ps','-p','1680782,1680783,1680784,1680785,1696443,1696444,1696445,1696446','-o','pid=,lstart='],capture_output=True,text=True)
print(json.dumps(dict(epoch=time.time(),files=files,counts=counts,failure_files=failures,slurm=dict(rc=s.returncode,stdout=s.stdout,stderr=s.stderr),gpu_processes=dict(rc=gpu.returncode,stdout=gpu.stdout,stderr=gpu.stderr),protected_ps=ps.stdout)),flush=True)
