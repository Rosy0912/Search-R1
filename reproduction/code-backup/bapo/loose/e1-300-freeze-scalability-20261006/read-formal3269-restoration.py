from pathlib import Path
import json,hashlib,base64,time
r=Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-freeze-scalability-20261006/runs/bapo-instruct300-formal300-3269')
names=['recovery-witness/rank_'+str(i)+'.json'for i in range(8)]+['recovery-witness/driver.json','recovery-permission.json','recovery-permission-check.json','recovery-manifest.json','resolved-config.yaml','source-commit.txt','retrieval-assets-locality.json']
files={}
for name in names:
 p=r/name;assert p.is_file()and not p.is_symlink()and p.resolve()==p;b=p.read_bytes();files[name]={'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'base64':base64.b64encode(b).decode(),'text':b.decode()}
print(json.dumps({'status':'ACTUAL_READONLY_FORMAL3269_RECOVERY_WITNESS_BYTES','job_id':3269,'run_root':str(r),'captured_at_unix':time.time(),'files':files}))
