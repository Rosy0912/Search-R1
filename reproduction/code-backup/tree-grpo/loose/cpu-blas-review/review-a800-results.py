from pathlib import Path
import json,math,hashlib
chat=Path(__file__).resolve().parents[2];ev=chat/'outputs/evidence';p=ev/'a800-fp32-index-difference.json';d=json.loads(p.read_text());base=json.loads((ev/'retrieval-cpu-2854.json').read_text());summaries=[];query_rows={i+1:{'query':q,'modes':{}} for i,q in enumerate(base['queries'])}
assert d['gpu_uuid']=='GPU-c63282c6-1d05-224e-234b-2a8c794b1f04' and d['index_float16'] is False and d['scratch_bytes']==512*1024**2
assert d['asset_identity']['e5_Flat.index']['sha256']==base['asset_identity']['index']['sha256'] and d['asset_identity']['wiki-18.jsonl']['sha256']==base['asset_identity']['corpus']['sha256']
assert [(r['queries'],r['k']) for r in d['results']]==[(14,3),(14,50),(512,3),(512,50)]
for block in d['results']:
 n,k=block['queries'],block['k'];assert len(block['details'])==n;deltas=[];exact=0
 for i,row in enumerate(block['details']):
  assert row['row']==i and row['query_index']==i%14
  old=base['results'][i%14][str(k)];wanted=old['ids'];got=row['ids'];assert len(got)==len(set(got))==k and set(wanted)==set(got);assert len(row['scores'])==k and all(math.isfinite(v) for v in row['scores'])
  lm=dict(zip(wanted,old['scores']));rm=dict(zip(got,row['scores']));delta=max(abs(lm[v]-rm[v]) for v in lm);assert delta==row['max_shared_score_delta'] and row['exact_rank']==(wanted==got) and row['overlap']==k
  exact+=wanted==got;deltas.append(delta);key=f'n{n}_k{k}';q=query_rows[i%14+1];stats=q['modes'].setdefault(key,{'exact_rows':0,'rows':0,'orders':set(),'max_delta':0.,'changed_positions':[]})
  stats['exact_rows']+=wanted==got;stats['rows']+=1;stats['orders'].add(tuple(got));stats['max_delta']=max(stats['max_delta'],delta)
  moved=[j+1 for j,(x,y) in enumerate(zip(wanted,got)) if x!=y]
  assert not stats['changed_positions'] or stats['changed_positions']==moved
  stats['changed_positions']=moved
  if n==14 and moved:stats['changed_entries']=[{'id':wanted[j-1],'cpu_rank':j,'gpu_rank':got.index(wanted[j-1])+1,'cpu_score':lm[wanted[j-1]],'gpu_score':rm[wanted[j-1]]} for j in moved]
 assert exact==block['exact_rank_rows'];summaries.append({'n':n,'k':k,'exact_rows':exact,'sets_equal':n,'max_delta':max(deltas),'wall_seconds':block['wall_seconds'],'process_cpu_seconds':block['process_cpu_seconds']})
for q in query_rows.values():
 for stats in q['modes'].values():stats['distinct_repeated_query_orders']=len(stats.pop('orders'))
report={'status':'INDEPENDENT_DIFFERENCE_REVIEW','candidate_next_gate':'CPU_ENCODER_AND_HTTP_GATE_SUPPORTED_PENDING_COST_RESOURCE_REVIEW','production_accepted':False,'queries':query_rows,'summary':summaries,'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest(),'retrieval-cpu-2854.json':hashlib.sha256((ev/'retrieval-cpu-2854.json').read_bytes()).hexdigest()}}
(chat/'work/cpu-blas-review/a800-independent-differences.json').write_text(json.dumps(report,indent=2));print(json.dumps(summaries,indent=2))
