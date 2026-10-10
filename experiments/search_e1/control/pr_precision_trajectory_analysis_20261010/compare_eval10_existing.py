"""CPU-only existing R/PR10 native predictions. Print compact aggregates, never raw text."""
import hashlib,json,random,time
from collections import Counter
from pathlib import Path
R_SUMMARY=Path('/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008/eval-stats-v2-72e1e12-8b641cb51b4c478684f596f75c603851/R/runs/summary.json')
PR_SUMMARY=Path('/data/shaoshitong/search-e1-opsd-pr100-8gpu-20261010/periodic-recovery-e72eb3c/eval-010/runs/summary.json')
def load(path):
 raw=path.read_bytes(); return json.loads(raw),dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def existing(path,version):
 summary,proof=load(path)
 ref=summary['private_primary_records'];rows=[json.loads(line) for line in Path(ref['path']).read_text().splitlines() if line.strip()]
 rows=sorted([r for r in rows if r['version']==version],key=lambda r:r['validation_index'])
 assert len(rows)==512 and [r['validation_index'] for r in rows]==list(range(512))
 return summary,proof,rows

def ci(ds):
 n=len(ds);rng=random.Random(42);draw=sorted(sum(ds[rng.randrange(n)] for _ in range(n))/n*100 for _ in range(10000))
 def q(x):
  p=x*(len(draw)-1);i=int(p);return draw[i]+(draw[min(i+1,len(draw)-1)]-draw[i])*(p-i)
 return dict(question_units=n,replicates=10000,seed=42,lower_pp=q(.025),upper_pp=q(.975),mean_pp=sum(ds)/n*100,method='paired question-slot percentile bootstrap')
def aggregate(pairs):
 c=Counter((a['protocol_em'],b['protocol_em']) for a,b in pairs)
 subset={k:[(a,b) for a,b in pairs if (a['protocol_em'],b['protocol_em'])==kind] for k,kind in [('lost',(1,0)),('gained',(0,1)),('preserved',(1,1)),('still_wrong',(0,0))]}
 def counts(rows):
  return dict(questions=len(rows),hits=sum(r['protocol_em'] for r in rows),final_invalid=sum(r['failure_class']=='final_invalid' for r in rows),valid_gold_mismatch=sum(r['failure_class']=='valid_gold_mismatch' for r in rows),no_search=sum(r['search_calls']==0 for r in rows),search_calls=dict(Counter(r['search_calls'] for r in rows)),termination=dict(Counter(r['termination'] for r in rows)))
 return dict(questions=len(pairs),R=counts([a for a,b in pairs]),PR10=counts([b for a,b in pairs]),new_hits=c[0,1],lost_hits=c[1,0],preserved_hits=c[1,1],still_wrong=c[0,0],net_hits=c[0,1]-c[1,0],bootstrap=ci([b['protocol_em']-a['protocol_em'] for a,b in pairs]),search_transition=dict(Counter(str(a['search_calls'])+'->'+str(b['search_calls']) for a,b in pairs)),termination_transition=dict(Counter(a['termination']+'->'+b['termination'] for a,b in pairs)),by_flip={k:dict(count=len(v),PR_failure_class=dict(Counter(str(b['failure_class']) for a,b in v)),search_change=dict(Counter('fewer' if b['search_calls']<a['search_calls'] else 'more' if b['search_calls']>a['search_calls'] else 'same' for a,b in v)),search_transition=dict(Counter(str(a['search_calls'])+'->'+str(b['search_calls']) for a,b in v))) for k,v in subset.items()})
def typical(pairs,kind):
 candidates=[(a,b) for a,b in pairs if (a['protocol_em'],b['protocol_em'])==kind]
 selected=[]
 # Deterministic 3: smallest index in each dataset, then greatest search-count change.
 for dataset in ('nq','hotpotqa'):
  choices=[p for p in candidates if p[0]['dataset']==dataset]
  if choices:selected.append(min(choices,key=lambda p:p[0]['validation_index']))
 other=[p for p in candidates if p not in selected]
 if other:selected.append(min(other,key=lambda p:(-abs(p[0]['search_calls']-p[1]['search_calls']),p[0]['validation_index'])))
 def item(p):
  a,b=p
  return dict(validation_index=a['validation_index'],question_id_sha256=hashlib.sha256(a['question_id'].encode()).hexdigest(),dataset=a['dataset'],R=dict(em=a['protocol_em'],failure_class=a['failure_class'],search_calls=a['search_calls'],termination=a['termination'],remote_native_trace=a['trace']),PR10=dict(em=b['protocol_em'],failure_class=b['failure_class'],search_calls=b['search_calls'],termination=b['termination'],remote_native_trace=b['trace']))
 return [item(p) for p in selected]
a,ap,ar=existing(R_SUMMARY,'raw10_bf16');b,bp,br=existing(PR_SUMMARY,'raw_bf16')
assert a['identities']==b['identities']
pairs=list(zip(ar,br));assert all(x['question_id']==y['question_id'] and x['row_semantic_sha256']==y['row_semantic_sha256'] and x['dataset']==y['dataset'] for x,y in pairs)
assert sum(x['protocol_em'] for x in ar)==240 and sum(x['protocol_em'] for x in br)==232
out=dict(schema='R-vs-PR10-existing-native512-cpu-comparison-v1',accepted=True,observed_unix=time.time(),R_summary=ap,PR_summary=bp,R_existing_records=a['private_primary_records'],PR_existing_records=b['private_primary_records'],identities=a['identities'],overall=aggregate(pairs),datasets={d:aggregate([p for p in pairs if p[0]['dataset']==d]) for d in ('nq','hotpotqa')},typical_examples=dict(selection='smallest validation index in NQ and Hotpot, then maximum absolute search-count difference with index tie-break; descriptive examples, not population causal evidence',lost=typical(pairs,(1,0)),gained=typical(pairs,(0,1))),scope='Existing complete raw10 R versus existing PR10 BF16 unmerged native512. No inference, rescoring, Tensor/Torch, GPU, model state, HTTP, locking or remote writes. Metadata/paths only; raw trace/question/answer text stays HKUST. Cannot isolate precision from changed online history.')
print(json.dumps(out,ensure_ascii=False))
