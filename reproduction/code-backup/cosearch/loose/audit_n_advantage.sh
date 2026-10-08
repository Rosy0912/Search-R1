CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 /dataset/shaoshitong/search-r1-rl/cosearch/env/bin/python - <<'PY'
import collections,json,hashlib,unicodedata
from pathlib import Path
import torch
import regex
torch.set_num_threads(1)
p=Path('/dataset/shaoshitong/search-r1-rl/cosearch/runs/cosearch-local-gpu-five-20261003-n/output/ranker-update-evidence/1791036275337325910-3066368.json')
d=json.loads(p.read_text());groups=collections.defaultdict(list)
for e in d['events']:
 if e['retained']:groups[e['group_key']].append(e)
counts=collections.Counter();minimum=1.;examples=[]
answer_cache={}
def contains(text,aliases):
 key=(text,tuple(aliases))
 if key in answer_cache:return answer_cache[key]
 def tokens(s):
  matches=regex.findall(r'([\p{L}\p{N}\p{M}]+)|([^\p{Z}\p{C}])',unicodedata.normalize('NFD',s),flags=regex.I|regex.U|regex.M)
  return [(x or y).lower() for x,y in matches]
 words=tokens(text);result=False
 for alias in aliases:
  target=tokens(alias)
  if target and any(words[i:i+len(target)]==target for i in range(len(words)-len(target)+1)):result=True;break
 answer_cache[key]=result
 return result
def relevance(e):
 aliases=e['targets']
 trivial=bool(aliases) and all(x.lower().strip() in ('yes','no','true','false') for x in aliases)
 if trivial or not any(contains(x,aliases) for x in e['candidates']):return 0.
 selected=[e['candidates'][i-1] for i in e['ranking']]
 return sum(any(contains(x,aliases) for x in selected[:k]) for k in (1,3,5))/3
for key,rows in groups.items():
 if len(rows)<3 or len({e['ranker_reward'] for e in rows})<=1:continue
 values=torch.tensor([e['ranker_reward'] for e in rows])
 advantages=(values-values.mean())/(values.std(correction=1)+1e-6)
 legal=[e for e in rows if e['ranking'] is not None]
 if len({e['ranker_reward'] for e in legal})>1:
  relevance_values=[relevance(e) for e in legal]
  counts['valid_difference_groups_with_relevance_difference']+=len(set(relevance_values))>1
  counts['valid_difference_groups_with_main_difference']+=len({e['main_reward'] for e in legal})>1
 for e,adv in zip(rows,advantages):
  if not e['chosen']:continue
  counts['chosen']+=1
  x=float(adv);assert torch.isfinite(adv)
  counts['exact_float32_nonzero']+=x!=0.
  counts['abs_advantage_above_1e_5']+=abs(x)>1e-5
  if 0<abs(x)<=1e-5:
   counts['nonzero_near_roundoff']+=1
   if len(examples)<8:examples.append(dict(group=key,trajectory=e['trajectory_uid'],reward=e['ranker_reward'],advantage=x))
  if x:minimum=min(minimum,abs(x))
assert counts['exact_float32_nonzero']==d['selection_metrics_before_gradients']['ranker/effective_nonzero_advantage_events']
print(json.dumps(dict(counts=dict(counts),minimum_nonzero_abs_advantage=minimum,near_roundoff_examples=examples)))
PY
