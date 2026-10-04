import subprocess,json,time,os
from pathlib import Path
b=Path(os.environ['E1_BASE']);old=Path('/weka_3/shaoshitong/search-e1-config-study-20261004')
r=json.loads((old/'receipts/archive-round-1-grpo.json').read_text())
assert r['run_id']=='f04fe593830c4831918ede872812c016' and r['status']=='RUNNING' and time.time()-r['updated_unix']<90
raw=Path('/proc/1438383/environ').read_bytes().split(b'\0')
assert b'E1_RUN_ID=f04fe593830c4831918ede872812c016' in raw
uuid='GPU-21352a4a-5b55-8adf-8e82-c70b7a94ecf3'
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
selected=[x.split(',') for x in apps.splitlines() if x.startswith(uuid)]
assert len(selected)==1 and int(selected[0][1])==1438383,selected
assert int(selected[0][2])<49140-7168,selected
(b/'receipts/owned-shared-gpu-admission.json').write_text(json.dumps(dict(run=r['run_id'],worker=1438383,uuid=uuid,apps=apps,time=time.time()),indent=2))
print('OWNED_GPU7_SHARED_ADMISSION_PASSED',flush=True)
