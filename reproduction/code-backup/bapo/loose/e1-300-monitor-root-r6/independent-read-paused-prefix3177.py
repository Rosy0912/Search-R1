import pathlib,json,base64,hashlib
r=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-monitor-r6-20261004/runs/bapo-instruct300-formal300-3177')
paths=[r/'checkpoints/full_state/global_step_5/manifest.json',r/'logs/independent-step5-extraction-firstactual-20261004/raw/step5-prefix.log']
for p in paths:
 b=p.read_bytes();print(json.dumps({'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'base64':base64.b64encode(b).decode()}))
