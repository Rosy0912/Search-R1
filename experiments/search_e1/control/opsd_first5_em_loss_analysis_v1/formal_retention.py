from pathlib import Path
from collections import Counter
import json,hashlib,datetime
O=Path('/Users/shaoshitong/Documents/Codex/2026-10-03/xia/work/search-e1-analysis-output/opsd-ablation-execution-20261008/em-loss-relation-20261009/formal512-existing-evidence-20261009-v2')
p=O/'formal512-paired-outcomes.private.jsonl';b=p.read_bytes(); rows=[json.loads(l) for l in b.splitlines() if l.strip()]
assert len(rows)==512 and hashlib.sha256(b).hexdigest()=='9246bfac17ffc5bd9414a7c69b0adf8b617aa6db0edbff6363a0261006b41ee1'
labels=['bf16_zero','C0','C1','M']
is_c0lost=lambda x:x['models']['bf16_zero']['protocol_em']==1 and x['models']['C0']['protocol_em']==0
sets={
 'baseline_correct_C0_lost_all':[r for r in rows if is_c0lost(r)],
 'baseline_correct_C0_lost_C1_correct':[r for r in rows if is_c0lost(r) and r['models']['C1']['protocol_em']==1],
 'baseline_correct_C0_lost_M_correct':[r for r in rows if is_c0lost(r) and r['models']['M']['protocol_em']==1],
 'baseline_correct_C0_lost_both_C1_M_correct':[r for r in rows if is_c0lost(r) and all(r['models'][n]['protocol_em']==1 for n in ['C1','M'])],
 'baseline_correct_C0_lost_both_C1_M_wrong':[r for r in rows if is_c0lost(r) and all(r['models'][n]['protocol_em']==0 for n in ['C1','M'])],
 'baseline_correct_C0_lost_only_C1_correct':[r for r in rows if is_c0lost(r) and r['models']['C1']['protocol_em']==1 and r['models']['M']['protocol_em']==0],
 'baseline_correct_C0_lost_only_M_correct':[r for r in rows if is_c0lost(r) and r['models']['C1']['protocol_em']==0 and r['models']['M']['protocol_em']==1]}
def actionclass(x):
 return 'strict_valid_final_answer' if x['protocol_valid_final_action'] else 'final_invalid'
def aggregates(rr):
 x={'questions':len(rr),'datasets':dict(Counter(r['dataset'] for r in rr)),'models':{},'search_change_relative_C0':{}}
 for n in labels:
  m=[r['models'][n] for r in rr]
  x['models'][n]={'strict_hits':sum(y['protocol_em'] for y in m),'search_calls_histogram':dict(Counter(str(y['search_calls']) for y in m)),'mean_search_calls':sum(y['search_calls'] for y in m)/len(m) if m else None,'final_action_classes':dict(Counter(actionclass(y) for y in m)),'termination_histogram':dict(Counter(y['termination'] for y in m)),'accepted_failure_classes':dict(Counter('null' if y['failure_class'] is None else y['failure_class'] for y in m))}
 for n in ['bf16_zero','C1','M']:
  d=[r['models'][n]['search_calls']-r['models']['C0']['search_calls'] for r in rr]
  x['search_change_relative_C0'][n]={'more':sum(v>0 for v in d),'same':sum(v==0 for v in d),'less':sum(v<0 for v in d),'sum_delta':sum(d),'mean_delta':sum(d)/len(d) if d else None}
 return x
pub={name:aggregates(rr) for name,rr in sets.items()}
private={'schema':'formal512-C0-loss-C1-M-retention-cohorts-v1','accepted':True,'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source':{'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()},'scope':'Exhaustive matching over all512 formal greedy questions, original accepted post10outer endpoints. Conditions defined by native strict EM, no rescoring, raw text or new inference. Action classes are final answer validity only, not intermediate generated action types or semantic reasoning validity. No TRAIN loss merged into these question outcomes.','aggregate_counts':pub,'private_slots_by_cohort':{n:[{'validation_index':r['validation_index'],'question_id':r['question_id'],'row_semantic_sha256':r['row_semantic_sha256']} for r in rr] for n,rr in sets.items()},'interpretation_limits':['Conditioning on C0 lost correctness selects32 baseline-correct questions after observing outcomes; descriptions are exploratory post-outcome subsets.','Search count changes co-occur with model/answer changes and cannot prove search depth caused correctness or training loss changes.','All final-valid wrong cases do not exclude intermediate malformed tags or poor retrieval/action decisions.','C1/M hold in this comparison means correct at a different fixed endpoint on same question, not recovery from C0 training or a temporal causal intervention.']}
q=O/'C0-lost-C1-M-retained.private.json';q.write_text(json.dumps(private,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
safe={k:v for k,v in private.items() if k not in ['source','private_slots_by_cohort']}
safe['source_table_content_sha256']=hashlib.sha256(b).hexdigest()
safe['private_derivation_sha256']=hashlib.sha256(q.read_bytes()).hexdigest()
s=O/'C0-lost-C1-M-retained.aggregate.json';s.write_text(json.dumps(safe,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
print(json.dumps({'private_path':str(q),'private_sha256':hashlib.sha256(q.read_bytes()).hexdigest(),'safe_path':str(s),'safe_sha256':hashlib.sha256(s.read_bytes()).hexdigest(),'cohorts':pub},ensure_ascii=False))
