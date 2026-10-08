import json,random,re,hashlib,pathlib,itertools
base=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo/assets-root/assets');sources=['nq','triviaqa','popqa','hotpotqa','2wikimultihopqa','musique','bamboogle'];splits={k:[] for k in ['train','validation','stage','final']}
for source in sources:
 def rows(split):
  raw=[json.loads(l) for l in (base/source/(split+'.jsonl')).read_text().splitlines()]
  return [dict(question_id=source+':'+split+':'+str(i),question=r['question'].strip(),answers=r['golden_answers'],source=source,raw_id=r['id']) for i,r in enumerate(raw)]
 if source in ['nq','hotpotqa']:
  train=rows('train');random.Random(42).shuffle(train);splits['validation']+=train[:256];splits['stage']+=train[256:288];splits['train']+=train[288:]
 final='test' if source in ['nq','triviaqa','popqa','bamboogle'] else 'dev';splits['final']+=rows(final)
out={'counts':{k:len(v) for k,v in splits.items()},'source_counts':{k:{s:sum(r['source']==s for r in v) for s in sorted(set(r['source'] for r in v))} for k,v in splits.items()},'intersections':{},'duplicate_counts':{}}
for mode in ['exact','case_space_question_mark']:
 key=lambda r:r['question'] if mode=='exact' else ' '.join(r['question'].casefold().rstrip('?').split())
 index={k:{key(r):r for r in rows} for k,rows in splits.items()};out['duplicate_counts'][mode]={k:len(splits[k])-len(index[k]) for k in splits}
 out['intersections'][mode]={}
 for a,b in itertools.combinations(splits,2):
  common=set(index[a])&set(index[b]);out['intersections'][mode][a+'__'+b]={'count':len(common),'examples':[dict(question=q,left_id=index[a][q]['question_id'],right_id=index[b][q]['question_id']) for q in sorted(common)[:5]]}
print(json.dumps(out))
