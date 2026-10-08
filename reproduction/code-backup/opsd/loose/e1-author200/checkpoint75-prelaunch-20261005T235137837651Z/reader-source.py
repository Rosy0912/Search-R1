
import datetime,hashlib,json,os,sys
from pathlib import Path
T=Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct200-author-20261006/tools/final-acceptance-f5c72c942641');sys.path.insert(0,str(T));import checkpoint_overlap as c
PIN='f5c72c9426413a18e465030efb714c9015fa84274baa3fc4e401383adc96b3fd'
c.tools(PIN);a=c.allocation();result={'scope':'READ_ONLY_ACTUAL75_PRELAUNCH_PREFIXES_NOT_CONTENT_ACCEPTANCE','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_commit':c.SOURCE,'tools_manifest_sha256':PIN,'allocation':a,'log_prefixes':{},'complete_counts':{}}
fields=('st_dev','st_ino','st_uid')
for name,p in [('driver',Path(os.readlink('/proc/1533312/fd/1'))),('training',c.RUN/'train.log')]:
 c.physical(p);before=p.stat()
 fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
 with os.fdopen(fd,'rb') as handle:
  opened=os.fstat(handle.fileno());assert tuple(getattr(before,k) for k in fields)==tuple(getattr(opened,k) for k in fields);raw=handle.read(before.st_size);opened_after=os.fstat(handle.fileno())
 after=p.stat();assert tuple(getattr(before,k) for k in fields)==tuple(getattr(opened_after,k) for k in fields)==tuple(getattr(after,k) for k in fields)
 closed=raw[:raw.rfind(b'\n')+1];rows=[]
 for line in closed.splitlines():
  if b'OPSD_E1_OUTER_JSON=' in line:
   row=json.loads(line.split(b'OPSD_E1_OUTER_JSON=',1)[1]);assert row['source_commit']==c.SOURCE and row['config_sha256']=='73f6401e818dc93a11c705da2357d0f3d91b2a51609a51e78a599fb7cf416c60';rows.append(row)
 assert [r['global_step'] for r in rows]==list(range(1,len(rows)+1)) and len(rows)<=200
 result['log_prefixes'][name]={'path':str(p),'bytes':len(closed),'sha256':hashlib.sha256(closed).hexdigest()};result['complete_counts'][name]=len(rows)
H=c.RUN/'checkpoints/actor/global_step_75';paths=[H/n for n in ('config.json','generation_config.json','model-00001-of-00003.safetensors','model-00002-of-00003.safetensors','model-00003-of-00003.safetensors','model.safetensors.index.json','tokenizer_config.json','special_tokens_map.json','tokenizer.json','vocab.json','merges.txt','added_tokens.json')]
paths += [H/'training_state'/('rank_'+str(r)+s) for r in range(8) for s in ('.pt','.json')]
paths += [c.RUN/'full_state/global_step_75'/n for n in ('driver.pt','driver.json')]+[c.RUN/'checkpoints/algorithm/global_step_75/state.json']
result['missing_current_members']=[str(p) for p in paths if not p.is_file()];result['expected_member_count']=len(paths)
result['existing_marker']=(c.RUN/'evidence/checkpoint-cpu-75-launch.json').exists();result['existing_output']=(c.RUN/'evidence/checkpoint-75-cpu-inspection-once-v1').exists()
result['ready_for_one_cpu_inspection']=min(result['complete_counts'].values())>=75 and not result['missing_current_members'] and not result['existing_marker'] and not result['existing_output']
print(json.dumps(result,sort_keys=True))
