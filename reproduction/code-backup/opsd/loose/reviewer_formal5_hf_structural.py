import subprocess,shlex,json
from pathlib import Path
code="""from pathlib import Path
import json,hashlib,subprocess
from safetensors import safe_open
root=Path('/mnt/data/shaoshitong/search-r1-rl/opsd');run=root/'runs/opsd-formal-2999';p=run/'output/checkpoints/opsd-gpu-formal500/actor/global_step_5'
assert subprocess.check_output(['git','-C',str(root/'repo'),'rev-parse','HEAD'],text=True).strip()=='8e60976d364ab7d7ae51714415813c1359dafe9a'
subprocess.run(['git','-C',str(root/'repo'),'diff','--quiet','HEAD'],check=True)
m=json.loads((p/'model.safetensors.index.json').read_text())['weight_map'];assert len(m)==435
actual={};weights={}
for f in p.glob('*.safetensors'):
 with safe_open(str(f),framework='pt',device='cpu') as z:
  for key in z.keys():assert key not in actual;actual[key]=f.name
 h=hashlib.sha256()
 with f.open('rb') as z:
  for b in iter(lambda:z.read(8*1024*1024),b''):h.update(b)
 weights[f.name]={'bytes':f.stat().st_size,'sha256':h.hexdigest()}
assert len(weights)==3 and actual==m
for n in ['config.json','tokenizer.json','tokenizer_config.json']:assert (p/n).is_file()
r=run/'evidence/resources.ndjson';last={}
for line in r.read_text().splitlines()[-20:]:
 try:last=json.loads(line)
 except json.JSONDecodeError:pass
print(json.dumps({'actual_job':'2999','checkpoint_step':5,'source_commit':'8e60976d364ab7d7ae51714415813c1359dafe9a','actual_hf_index_and_keys_count':435,'all3_shards_full_hash':weights,'full_HF_reload':False,'formal500_completed':False,'latest_resource_sample':last}))
"""
cmd='/mnt/data/shaoshitong/search-r1-rl/opsd/env/bin/python -c '+shlex.quote(code)
r=subprocess.run(['python3','work/remote.py',cmd],capture_output=True,text=True)
if r.returncode:print(r.stdout,r.stderr);raise SystemExit(r.returncode)
p=Path('work/opsd/reproduction/baselines/opsd-validation-20261003/evidence/job2999_actual5_HF_structure_sha_reviewer.json');p.write_text(r.stdout);x=json.loads(r.stdout);print({k:v for k,v in x.items() if k!='latest_resource_sample'})
