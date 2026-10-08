from pathlib import Path
import json,hashlib,base64
R=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-behavior-20261006/numeric-diagnostic-3310-20261006')
files={}
for name in ['diagnostic-cpu-precheck.json','runtime-executable.json','driver-library.actual.json','numeric-cpu-plan.json','target-tests.actual.json','target-tests.actual.log','diagnostic-approval.json']:
 raw=(R/'logs'/name).read_bytes();files[name]=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),base64=base64.b64encode(raw).decode())
print(json.dumps(dict(root=str(R),files=files)))
