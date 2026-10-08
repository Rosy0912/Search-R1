
import json,collections,hashlib,os,statistics,re
p='/data/shaoshitong/search-e1-opsd-from-archive100-8gpu-20261007/runs/preparation-3-pairs.jsonl'
c=collections.Counter(); by=collections.defaultdict(collections.Counter); lengths=collections.defaultdict(list); spans=collections.Counter(); ids=set(); h=hashlib.sha256()
with open(p,'rb') as f:
 for line in f:
  h.update(line); r=json.loads(line); source=r['question_id'].split(':')[0]; ids.add(r['question_id']); key='student_reward_'+str(r['student_reward']); c[key]+=1; c['reference_reward_'+str(r['reference_reward'])]+=1; by[source][key]+=1; by[source]['pairs']+=1
  for role in ['reference','student']:
   lengths[role+'_chars'].append(len(r[role])); lengths[role+'_search_tags'].append(len(re.findall(r'<search>.*?</search>',r[role],re.S)))
  ps=r.get('student_policy_spans') or []; spans['policy_chars']+=sum(e-s for s,e in ps); spans['response_chars']+=len(r['student']); spans['empty_spans']+=not bool(ps)
  answer=list(re.finditer(r'<answer>.*?</answer>',r['student'],re.S))
  c['student_complete_answer_present']+=bool(answer)
  c['student_answer_fully_in_policy_span']+=bool(answer and any(s<=answer[-1].start() and answer[-1].end()<=e for s,e in ps))
def stats(xs):
 xs.sort(); return {'min':xs[0],'max':xs[-1],'mean':statistics.mean(xs),'p50':xs[len(xs)//2],'p90':xs[int(len(xs)*.9)],'p99':xs[int(len(xs)*.99)]}
print(json.dumps({'path':p,'sha256':h.hexdigest(),'bytes':os.stat(p).st_size,'pairs':sum(x['pairs'] for x in by.values()),'unique_question_ids':len(ids),'counts':dict(c),'by_source':dict(by),'lengths':{k:stats(v) for k,v in lengths.items()},'span_char_totals':dict(spans),'limitations':'search_tags counts syntactic text, not logged executed calls; character lengths are not token lengths; no questions or answers emitted'},indent=2))
