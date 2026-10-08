from pathlib import Path
import base64
mapping={n:n for n in []}
paths=['outputs/bapo-numericfix-accept5-3101-independent-review.md','outputs/bapo-numericfix-accept5-3101-independent-cost.json','outputs/bapo-numericfix-accept5-3101-independent-recomputed.json','work/evidence/numericfix-accept5-3101-independent-current-hashes.json','outputs/bapo-numericfix-accept5-3101-acceptance.json']
records={Path(p).name:base64.b64encode(Path(p).read_bytes()).decode() for p in paths};records['accept5-3101-acceptance.json']=records['bapo-numericfix-accept5-3101-acceptance.json']
s='''import pathlib,base64,subprocess,os
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-rewardfix-numeric')
for name,data in RECORDS.items():
 p=root/'logs'/name;b=base64.b64decode(data)
 assert not p.is_symlink()
 if p.exists():assert p.read_bytes()==b
 else:p.write_bytes(b)
log=root/'logs/accept5-3101-actual-receipt-gate.log';record=root/'logs/accept5-3101-actual-receipt-gate.json'
assert not log.exists() and not record.exists()
import time,json
started=time.time()
with log.open('w') as f:r=subprocess.run([str(root/'env/bin/python'),str(root/'tools/verify_five_receipt-gpu64.py'),'3101'],stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
record.write_text(json.dumps(dict(exit_code=r.returncode,wall_seconds=time.time()-started,job_id=3101),indent=2)+'\\n')
print(record.read_text());print(log.read_text());raise SystemExit(r.returncode)
'''.replace('RECORDS',repr(records))
Path('work/numericfix/deploy-accept5-3101-receipt.py').write_text(s)
