import json,gzip,hashlib,collections,math,time
from pathlib import Path
r=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-recovery300-3234/output');out=[];evidence=[]
for step in [126,127,128,129]:
 p=r/f'trajectories/outer-{step:06d}/trajectories.jsonl.gz';raw=p.read_bytes();evidence.append(dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw)))
 c=collections.Counter();reward=0;actions=obs=0
 with gzip.open(p,'rt') as f:
  for line in f:
   x=json.loads(line);reward+=x['reward'];c['rows']+=1
   rec=x['metadata']['action_records'];c['leaf_done_any']+=int(any(a['done'] for a in rec));c['leaf_answer_tag_any']+=int(any('<answer>' in a['text'] for a in rec));c['leaf_search_tag_any']+=int(any('<search>' in a['text'] for a in rec))
   for a in rec:
    c['action_records']+=1;c['valid_action']+=a['valid_action'];c['valid_search']+=a['valid_search'];c['done']+=a['done'];c['answer_tag_records']+=int('<answer>' in a['text']);actions+=a['action_tokens'];obs+=a['observation_tokens']
 out.append(dict(step=step,mean_reward=reward/c['rows'],counts=dict(c),action_token_events=actions,observation_token_events=obs,scope='retained leaf duplicated action records; text is generated action text, not prompt; not unique engine/HTTP cost'))
ratios=[]
for step in [127,128]:
 items=[]
 for p in sorted((r/'tree-rank-audit/ratio-tail').glob(f'step-{step:06d}-rank-*-optimizer-*.json')):
  raw=p.read_bytes();q=json.loads(raw);evidence.append(dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw)))
  for i,m in enumerate(q['microbatches']):items.append(dict(rank=q['rank'],optimizer_call=q['optimizer_call'],micro_index_zero=i,preclip_norm=q['preclip_norm'],record=m,path=str(p)))
 ratios.append(dict(step=step,largest_negative_tail=max(items,key=lambda x:x['record']['negative_advantage_max_log_ratio'] if x['record']['negative_advantage_max_log_ratio'] is not None else -math.inf),largest_preclip=max(items,key=lambda x:x['preclip_norm']),most_negative_delta=min(items,key=lambda x:x['record']['min_log_ratio'])))
print(json.dumps(dict(epoch=time.time(),behavior=out,ratio_peaks=ratios,evidence=evidence)),flush=True)
