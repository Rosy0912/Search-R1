CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 /dataset/shaoshitong/search-r1-rl/cosearch/env/bin/python - <<'PY'
import collections,hashlib,json,math,os,re,string,unicodedata
from pathlib import Path
import regex
from transformers import AutoTokenizer
root=Path('/dataset/shaoshitong/search-r1-rl/cosearch')
run=root/'runs/cosearch-local-gpu-five-20261003-n'
names=['1791036275337325910-3066368.json','1791036358528192443-3066368.json']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
stat=Path('/proc/3066368/stat').read_text().rsplit(')',1)[1].split()
assert int(stat[19])==280919692
assert '3002' in Path('/proc/3066368/cgroup').read_text()
hashes={n:sha(run/'output/ranker-update-evidence'/n) for n in names}
docs=[json.loads((run/'output/ranker-update-evidence'/n).read_text()) for n in names]
a,b=docs
assert a['events']==b['events'] and a['trajectories']==b['trajectories']
for d in docs:
 p=d['provenance'];assert str(p['job'])=='3002' and p['pid']==3066368 and p['outer_step']==1
 assert p['output']==str(run/'output')
 for name,digest in p['source_sha256'].items():
  path=root/'repo'/name if '/' in name else root/'repo/search_r1/baselines'/name
  assert sha(path)==digest,(name,'source changed')
 for name,digest in p['tokenizer_sha256'].items():assert sha(Path(p['tokenizer_path'])/name)==digest
tokenizer=AutoTokenizer.from_pretrained(a['provenance']['tokenizer_path'],local_files_only=True)
def records_reason(records):
 if not records:return 'empty_records'
 for i,r in enumerate(records):
  text=r['text'];acts=list(re.finditer(r'<(search|answer)>(.*?)</\1>',text,re.S))
  if len(acts)!=1:return 'action_count'
  act=acts[0];prefix=text[:act.start()].strip()
  if prefix and (not re.fullmatch(r'<think>.*?</think>',prefix,re.S) or prefix.count('<think>')!=1 or prefix.count('</think>')!=1):return 'invalid_prefix'
  if act.group(1)=='answer' and i!=len(records)-1:return 'nonfinal_answer'
  if not act.group(2).strip():return 'empty_action'
  if text[act.end():].strip():return 'trailing_text'
 return 'legal' if acts[0].group(1)=='answer' else 'no_final_answer'
def raw_answer(ids,mask):
 spans=[];start=None
 for i,m in enumerate(mask):
  if m and start is None:start=i
  elif not m and start is not None:spans.append(tokenizer.decode(ids[start:i]));start=None
 if start is not None:spans.append(tokenizer.decode(ids[start:]))
 result=None
 for text in spans:
  matches=list(re.finditer(r'<answer>(.*?)</answer>',text,re.S))
  if matches:result=matches[-1].group(1).strip()
 return result
reasons=collections.Counter();qid_rows=collections.Counter();mismatch=[]
for uid,t in a['trajectories'].items():
 reason=records_reason(t['action_records']);legal=reason=='legal';raw=raw_answer(t['response_ids'],t['response_mask'])
 assert legal==t['records_legal'] and raw==t['raw_answer'],uid
 assert t['answer']==(raw if legal else None),uid
 assert len(t['response_ids'])==len(t['response_mask'])
 reasons[reason]+=1;qid_rows[str(t['question_uid'])]+=1
 assert all(x in (0,1) for x in t['response_mask'])
def norm(s):
 s=''.join(c for c in s.lower() if c not in string.punctuation)
 return ' '.join(re.sub(r'\b(a|an|the)\b',' ',s).split())
def f1(pred,aliases):
 if pred is None:return -.2
 if norm(pred) in ('i dont know','i do not know'):return 0.
 left=norm(pred).split();scores=[]
 for alias in aliases:
  right=norm(alias).split();n=sum((collections.Counter(left)&collections.Counter(right)).values())
  scores.append(2*n/(len(left)+len(right)) if left and right else 0.)
 return max(scores,default=0.)
def tokens(s):return regex.findall(r'([\p{L}\p{N}\p{M}]+)|([^\p{Z}\p{C}])',unicodedata.normalize('NFD',s),flags=regex.I|regex.U|regex.M)
def tok(s):return [(x or y).lower() for x,y in tokens(s)]
def contains(doc,aliases):
 text=tok(doc)
 for alias in aliases:
  target=tok(alias)
  if target and any(text[i:i+len(target)]==target for i in range(len(text)-len(target)+1)):return True
 return False
def reward(e):
 if e['ranking'] is None:return -.2
 aliases=e['targets'];trivial=bool(aliases) and all(x.lower().strip() in ('yes','no','true','false') for x in aliases)
 available=not trivial and any(contains(d,aliases) for d in e['candidates'])
 selected=[e['candidates'][i-1] for i in e['ranking']]
 relevance=sum(any(contains(d,aliases) for d in selected[:k]) for k in (1,3,5))/3 if available else 0.
 return relevance if available and relevance<=.5 else relevance+e['main_reward'] if available else e['main_reward']
def parse(ids):
 text=tokenizer.decode(ids,skip_special_tokens=True)
 matches=list(re.finditer(r'<ranking>\s*(\[[^\]]*\])\s*</ranking>',text,re.S))
 if len(matches)!=1:return None
 try:result=json.loads(matches[0].group(1))
 except (TypeError,ValueError):return None
 if not isinstance(result,list) or len(result)!=5 or any(type(x)is not int for x in result):return None
 if len(set(result))!=5 or any(x<1 or x>50 for x in result):return None
 return result
retained=[];main_delta=0.;reward_delta=0.;rank_types=collections.Counter()
for e in a['events']:
 t=a['trajectories'][e['trajectory_uid']]
 assert e['question_uid']==t['question_uid']
 assert len(e['candidates'])==50 and parse(e['ranking_response_ids'])==e['ranking']
 assert e['main_eligible']==(t['answer'] is not None) and e['retained']==e['main_eligible']
 assert len(e['ranking_response_ids'])==e['response_length']
 main_delta=max(main_delta,abs(e['main_reward']-f1(t['answer'],e['targets'])))
 assert main_delta<1e-6
 if e['retained']:
  reward_delta=max(reward_delta,abs(e['ranker_reward']-reward(e)));assert reward_delta<1e-6
  retained.append(e);rank_types['valid' if e['ranking'] is not None else 'fallback']+=1
def query_tokens(s):return re.sub(r'\b(a|an|the)\b',' ',s.lower())
def query_f1(x,y):
 x=' '.join(re.sub(r'[^\w\s]',' ',query_tokens(x)).split()).split()
 y=' '.join(re.sub(r'[^\w\s]',' ',query_tokens(y)).split()).split()
 return 2*sum((collections.Counter(x)&collections.Counter(y)).values())/(len(x)+len(y)) if x and y else 0.
buckets=collections.defaultdict(list)
for i,e in enumerate(retained):buckets[e['question_uid']].append(i)
keys=[None]*len(retained);nextkey=0
for indices in buckets.values():
 representatives=[]
 for i in indices:
  for j,key in representatives:
   if query_f1(retained[i]['query'],retained[j]['query'])>=.8:keys[i]=key;break
  else:keys[i]=nextkey;representatives.append((i,nextkey));nextkey+=1
assert keys==[e['group_key'] for e in retained]
groups=collections.defaultdict(list)
for e,k in zip(retained,keys):groups[k].append(e)
informative={k for k,g in groups.items() if len(g)>=3 and len({e['ranker_reward'] for e in g})>1}
chosen=[e for e in retained if e['group_key'] in informative and e['response_length']>0]
assert [e['chosen'] for e in retained]==[e['group_key'] in informative and e['response_length']>0 for e in retained]
categories=collections.Counter();legal_pair_examples=[]
for k in informative:
 g=groups[k];valid=[e for e in g if e['ranking'] is not None];invalid=[e for e in g if e['ranking'] is None]
 distinct_valid=len({e['ranker_reward'] for e in valid})>1
 if distinct_valid:
  categories['valid_valid_reward_difference']+=1
  if len(legal_pair_examples)<5:
   left=valid[0];right=next(e for e in valid if e['ranker_reward']!=left['ranker_reward'])
   legal_pair_examples.append(dict(group=k,question_uid=left['question_uid'],rows=len(g),query=left['query'],
    pair=[dict(trajectory=e['trajectory_uid'],ranking=e['ranking'],main_reward=e['main_reward'],ranker_reward=e['ranker_reward']) for e in (left,right)]))
 else:categories['only_valid_invalid_reward_difference']+=1
 if invalid:categories['informative_groups_with_fallback']+=1
 if len({tuple(e['ranking']) for e in valid})>1:categories['different_valid_rankings']+=1
 # Distinct-reward informative groups imply nonzero advantages except exact mean members.
 mean=sum(e['ranker_reward'] for e in g)/len(g)
 categories['chosen_nonzero_double_precision']+=sum(e['response_length']>0 and e['ranker_reward']!=mean for e in g)
summary=dict(files_sha256=hashes,job=3002,pid=3066368,startticks=280919692,
 exact_events_equal=True,exact_trajectories_equal=True,source_and_tokenizer_hashes_verified=True,
 provenance=[d['provenance'] for d in docs],all_events=len(a['events']),retained=len(retained),trajectories=len(a['trajectories']),
 original_question_count=len(qid_rows),trajectories_per_question_histogram=dict(collections.Counter(qid_rows.values())),
 legality_reasons=dict(reasons),retained_rank_types=dict(rank_types),main_reward_max_abs_error=main_delta,
 ranker_reward_max_abs_error=reward_delta,groups=len(groups),informative_groups=len(informative),chosen=len(chosen),
 group_size_histogram=dict(collections.Counter(map(len,groups.values()))),reward_difference_categories=dict(categories),
 valid_pair_examples=legal_pair_examples,selection_metrics=[d['selection_metrics_before_gradients'] for d in docs],
 disclaimer='Selection evidence only; does not prove backward, optimizer step, parameter delta or complete outer')
assert hashes=={n:sha(run/'output/ranker-update-evidence'/n) for n in names}
print(json.dumps(summary,ensure_ascii=False))
PY
