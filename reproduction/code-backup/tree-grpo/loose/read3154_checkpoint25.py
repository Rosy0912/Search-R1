import json,hashlib,time
from pathlib import Path
root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-instruct300-3154/output/checkpoints/tree-instruct300')
p=root/'training_state/global_step_25/manifest.json'
b=p.read_bytes(); m=json.loads(b)
records=[]
for entry in m['worker_states']+[m['driver']]:
 f=Path(entry['path']); records.append({'path':str(f),'exists':f.is_file(),'bytes':f.stat().st_size,'record':entry})
hf=Path(m['hf_checkpoint'])
print(json.dumps({'epoch':time.time(),'scope':'Actual manifest identity and file presence/size; no full tensor reload or independent payload hash check','manifest_path':str(p),'manifest_sha256':hashlib.sha256(b).hexdigest(),'manifest':m,'state_files':records,'hf_files':[{'file':x.name,'bytes':x.stat().st_size} for x in sorted(hf.iterdir()) if x.is_file()]}))
