import json,collections,pathlib,time,subprocess
r=pathlib.Path('/dataset/shaoshitong/search-r1-rl/cosearch/author-native/runs/author-native-phase-aware-warm-five-m')
c=collections.Counter(); temps=collections.Counter();groups=collections.defaultdict(list); ev=[];last=[]
for p in (r/'evidence/workers').glob('*.jsonl'):
 for line in p.read_text().splitlines():
  x=json.loads(line);k=x.get('kind');c[k]+=1
  if k=='ranker_inference_contract':temps[str(x['actual_sampling_params'])]+=1
  if k=='ranker_ranking_result':groups[(x['query'],tuple(x['actual_candidate_ids']))].append(tuple(x['actual_reranked_docs']))
  if k in ['optimizer_step','update_end']:last.append({a:b for a,b in x.items() if not isinstance(b,(dict,list))})
for p in (r/'evidence').glob('controller-*.jsonl'):
 for line in p.read_text().splitlines():
  x=json.loads(line);ev.append({a:b for a,b in x.items() if not isinstance(b,(dict,list))})
print(json.dumps(dict(checked_at=time.time(),queue=subprocess.check_output(['squeue','-h','-j','3106','-o','%i %T %N'],text=True),worker_kind_counts=dict(c),actual_sampling=dict(temps),query_candidate_groups=len(groups),repeated_same_input_groups=sum(len(v)>1 for v in groups.values()),different_physical_ranking_groups=sum(len(set(v))>1 for v in groups.values()),context_refusals=len(list((r/'evidence/context-refusals').glob('*'))),controller=ev,worker_updates=last[-16:],driver_exit=(r/'driver-exit.json').read_text() if (r/'driver-exit.json').exists() else None)))
