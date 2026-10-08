"""Read-only same-node metadata and own-job inventory; no model payloads."""
import hashlib, json, socket, subprocess, time
from pathlib import Path

root = Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
parent = root/'runs/tree-recovery300-3234/output/checkpoints/tree-recovery300/training_state/global_step_110/manifest.json'
result = dict(epoch=time.time(), node=socket.gethostname(), parent_path=str(parent))
result['parent_sha256'] = hashlib.sha256(parent.read_bytes()).hexdigest()
result['parent_manifest'] = json.loads(parent.read_bytes())
for name, command in [('own_jobs', ['squeue', '-h', '-n', 'tree-recovery300,tree-safety300,tree-safety-calibration110', '-o', '%i|%j|%T|%N']),
                      ('gpus', ['nvidia-smi', '--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu', '--format=csv,noheader,nounits'])]:
    proc = subprocess.run(command, capture_output=True, text=True)
    result[name] = dict(returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
result['protected_processes'] = []
for pid in (1680782,1680783,1680784,1680785,1696443,1696444,1696445,1696446):
    p = Path('/proc')/str(pid)/'stat'
    stat = p.read_text() if p.exists() else None
    result['protected_processes'].append(dict(pid=pid,starttick=stat.rsplit(') ',1)[1].split()[19] if stat else None))
sampler = root/'env/lib/python3.10/site-packages/vllm/model_executor/layers/sampler.py'
if sampler.exists():
    lines = sampler.read_text().splitlines()
    result['sampler'] = dict(path=str(sampler),sha256=hashlib.sha256(sampler.read_bytes()).hexdigest(),
                           logprob_source=[dict(line=i+1,text=line) for i,line in enumerate(lines) if any(x in line for x in ('log_softmax','_apply_top_k_top_p','_apply_min_p','temperatures','_apply_penalties'))])
print(json.dumps(result))
