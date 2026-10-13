"""Read sparse recovery manifests and native dev metrics; no tensor loading or writes."""
import hashlib,json,re,time
from pathlib import Path
base=Path('/dataset/shaoshitong/recursive-bran-a-h100-20261012/recovery-from100-20261013')
config=json.loads((base/'receipts/configuration.json').read_text())
rows=[]
for path in sorted((base/'checkpoints/training_state').glob('global_step_*/**/manifest.json')):
 try:
  raw=path.read_bytes();m=json.loads(raw)
 except (FileNotFoundError,ValueError):continue
 i=m['identity'];w=m['worker_states']
 rows.append(dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),status=m['status'],identity=i,config_matches=i['config_sha256']==config['config_sha256'],worker_count=len(w),ranks=[x['identity']['rank'] for x in w],driver=m['driver'],boundary_phase=m.get('boundary_phase'),hf_checkpoint=m['hf_checkpoint']))
metrics=[]
with (base/'logs/training.log').open(errors='replace') as stream:
 for line in stream:
  match=re.search(r'\bstep:(\d+)\b',line)
  if match and int(match.group(1)) in (150,200) and 'timing_s/step:' in line:metrics.append(line.strip())
print(json.dumps(dict(time=time.time(),configuration=config,manifest_summaries=rows,native_boundary_metrics=metrics,scope='Small manifests and native aggregate metrics only; no independent model/Adam tensor readback')))
