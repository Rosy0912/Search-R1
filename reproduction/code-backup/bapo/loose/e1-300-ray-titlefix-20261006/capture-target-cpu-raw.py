import base64,hashlib,json
from pathlib import Path
r=Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-titlefix-20261006')
names=['logs/cpu-gate.log','logs/cpu-gate-durable.log','logs/cpu-gate-durable-exit.json','logs/runtime-executable.json','logs/runtime-capture.log','logs/tools-manifest.json','source-expected.txt']
files={}
for n in names:
 b=(r/n).read_bytes();files[n]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'base64':base64.b64encode(b).decode()}
print(json.dumps({'root':str(r),'files':files}))
