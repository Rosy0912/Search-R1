import json,hashlib,base64
from pathlib import Path
r=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-controller-initfix-20261006/logs')
f=r/'first5-v4-linux33-tests.actual.json';s=json.loads(f.read_text());b=(r/'first5-v4-linux33-tests.actual.log').read_bytes();assert hashlib.sha256(b).hexdigest()==s['raw_log_sha256'];assert s['passed']==33 and s['exit_code']==0
print(json.dumps(dict(record=s,record_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),raw_log_base64=base64.b64encode(b).decode())))
