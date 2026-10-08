import datetime,hashlib,json,sys,os
from pathlib import Path
sys.path.insert(0,'/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct200-author-20261006/tools/final-acceptance-f5c72c942641')
import checkpoint_overlap as c
value=json.loads(sys.stdin.buffer.read());a=c.allocation();records={}
for p,h in value['dependencies'].items():
 raw=c.pinned_bytes(Path(p),h);q=Path(p);records[p]={'physical_path':str(q),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'uid':q.stat().st_uid}
target=c.ROOT/'tools/final-acceptance-90c2695a7f46'
assert not target.exists() and not target.is_symlink()
assert not (c.RUN/'evidence/checkpoint-cpu-75-launch.json').exists() and not (c.RUN/'evidence/checkpoint-75-cpu-inspection-once-v1').exists()
print(json.dumps({'scope':'FRESH_ACTUAL278_DEPENDENCIES_NOT_NEW_TOOL_DEPLOYMENT_OR75_PASS','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_commit':c.SOURCE,'source_clean':True,'new_directory_present':False,'cpu75_marker_present':False,'allocation':a,'verified_dependencies':records},sort_keys=True))
