
import datetime,hashlib,json,os,sys
from pathlib import Path
T=Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct200-author-20261006/tools/final-acceptance-f5c72c942641')
sys.path.insert(0,str(T));import checkpoint_overlap as c
c.tools('f5c72c9426413a18e465030efb714c9015fa84274baa3fc4e401383adc96b3fd');a=c.allocation()
p=Path(os.readlink('/proc/1533312/fd/1'));c.physical(p);raw=p.read_bytes();closed=raw[:raw.rfind(b'\n')+1]
rows=[]
for line in closed.splitlines():
 if b'OPSD_E1_OUTER_JSON=' in line:
  r=json.loads(line.split(b'OPSD_E1_OUTER_JSON=',1)[1]);assert r['source_commit']==c.SOURCE and r['config_sha256']=='73f6401e818dc93a11c705da2357d0f3d91b2a51609a51e78a599fb7cf416c60';rows.append(r)
assert [r['global_step'] for r in rows]==list(range(1,len(rows)+1))
saved=c.RUN/'full_state/global_step_75/driver.json'
print(json.dumps({'scope':'READ_ONLY_NEXT75_PRECHECK_NOT_NEW_FULL_PROGRESS_ACCEPTANCE','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'allocation':a,'driver_complete_rows':len(rows),'latest_outer':rows[-1],'driver_prefix':{'path':str(p),'bytes':len(closed),'sha256':hashlib.sha256(closed).hexdigest()},'checkpoint75_driver_metadata_present':saved.is_file(),'existing75_marker':(c.RUN/'evidence/checkpoint-cpu-75-launch.json').exists(),'resources_current_bytes':(c.RUN/'resources.ndjson').stat().st_size},sort_keys=True))
