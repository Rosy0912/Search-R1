from pathlib import Path
import json,os,hashlib
r=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-safety-r5-20261006')
assert os.uname().nodename=='h100-5' and (r/'source-expected.txt').read_text().strip()=='aec5e057093b4c491bfb0a47abd7605d98981b56'
names=['target-cpu-gate.json','target-cpu-gate.raw.log','context-budget-audit.json','runtime-tests.actual.json','runtime-tests.actual.log','runtime-executable.json','frozen-policy-safety-config.json','user-fixed-policy-thresholds.json']
result={}
for n in names:
 p=r/'logs'/n;assert p.is_file() and not p.is_symlink();b=p.read_bytes();result[n]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'text':b.decode()}
print(json.dumps(result))
