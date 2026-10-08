import pathlib,json,hashlib,math,importlib.util
r=pathlib.Path(__file__).resolve().parents[2];ep=r/'outputs/evidence';bp=ep/'retrieval-cpu-2854.json';hp=ep/'hybrid-http-difference-2876.json';ip=ep/'hybrid-binary-index-difference.json';base=json.loads(bp.read_text());http=json.loads(hp.read_text());binary=json.loads(ip.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();summaries=[];changes=[]
spec=importlib.util.spec_from_file_location('hybrid_checker',r/'work/check_hybrid_retriever.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.validate_rows(binary,'binary');m.validate_rows(http,'http')
assert http['provenance']['index_instance']==binary['instance'] and http['baseline_sha256']==binary['baseline_sha256']==sha(bp)
assert http['provenance']['source_sha256']==sha(r/'work/hybrid_retrieval_server.py') and binary['instance']['source_sha256']==sha(r/'work/index_gpu_server.py')
assert http['provenance']['asset_identity']==base['asset_identity'];assert http['provenance']['model_sha256']=={k:v for k,v in base['asset_identity']['model'].items() if k not in m.IGNORED}
for kind,report in [('binary',binary),('http',http)]:
 for row in report['rows']:
  n=row.get('query_count',row.get('queries'));k=row['k'];reps=row['repeats'];assert reps[0]==reps[1]
  exact=sets=content=0;maximum=0
  for entry in reps[0]:
   expected=base['results'][entry['query_index']][str(k)];ids=entry['ids'];scores=entry['scores'];byid=dict(zip(ids,scores));common=set(ids)&set(expected['ids']);exact+=ids==expected['ids'];sets+=len(common)==k
   delta=max(abs(byid[i]-expected['scores'][expected['ids'].index(i)]) for i in common);maximum=max(maximum,delta)
   assert entry['exact_rank']==(ids==expected['ids']) and entry['overlap']==len(common) and entry['max_shared_score_delta']==delta
   if kind=='http':
    content+=entry['contents_sha256']==expected['contents_sha256'];assert entry['exact_content_rank']==(entry['contents_sha256']==expected['contents_sha256'])
    wanted_hash=dict(zip(expected['ids'],expected['contents_sha256']));current_hash=dict(zip(ids,entry['contents_sha256']));assert all(wanted_hash[i]==current_hash[i] for i in common)
   if n==14 and ids!=expected['ids']:
    positions=[j for j in range(k) if ids[j]!=expected['ids'][j]];changes.append({'kind':kind,'query_index':entry['query_index'],'query':base['queries'][entry['query_index']],'k':k,'rank_positions_1based':[p+1 for p in positions],'baseline_ids':[expected['ids'][p] for p in positions],'actual_ids':[ids[p] for p in positions],'baseline_scores':[expected['scores'][p] for p in positions],'actual_scores':[scores[p] for p in positions],'baseline_contents_sha256':[expected.get('contents_sha256',[])[p] for p in positions],'actual_contents_sha256':[entry.get('contents_sha256',['not-loaded']*k)[p] for p in positions]})
  summaries.append({'kind':kind,'n':n,'unique_queries':14,'k':k,'actual_exact_rank':exact,'sets_equal':sets,'content_rank_equal':content if kind=='http' else 'not read by index-only API','max_matched_score_delta':maximum,'repeat_stable':True,'wall_seconds':row.get('timings_seconds',row.get('times_seconds'))})
encoder=[json.loads(l) for l in (ep/'hybrid-http-encoder-2876.jsonl').read_text().splitlines()];cost_records=[json.loads(l) for l in (ep/'hybrid-http-cost-index-2876.jsonl').read_text().splitlines()];cost=cost_records[0];proxy=cost_records[1:];assert len(encoder)==len(proxy)==6
for e,p in zip(encoder,proxy):assert e['queries']==p['queries'] and e['vectors_sha256']==p['vectors_sha256']
assert all(e['vectors_sha256']==base['embeddings_sha256'] for e in encoder if e['queries']==14)
assert {e['vectors_sha256'] for e in encoder if e['queries']==512}=={'c7bc1b4a084a169e6b6f197958d462094a743d34347842bd57eb249c308da7de'}
assert cost['allocated_gpus']==0 and cost['cpus']==8 and cost['exit_code']==0
# Bound the repeated512 vector and canonical frame against saved baseline vectors, not a self-declared hash.
import numpy as np,io
payload=np.load(ep/'retrieval-cpu-embeddings-2854.npz',allow_pickle=False) if (ep/'retrieval-cpu-embeddings-2854.npz').exists() else None
vector_proof='raw14 SHA equals independently saved2854; repeated512 hash agrees between real encoder and proxy; saved NPZ local absent'
if payload is not None:
 emb=np.ascontiguousarray(payload['embeddings'],dtype=np.float32);repeat=np.ascontiguousarray(emb[np.arange(512)%14]);assert hashlib.sha256(repeat.tobytes()).hexdigest()==encoder[4]['vectors_sha256'];vector_proof='recomputed from saved NPZ'
result={'scope':'real reports independently recomputed, no new hardware run or automatic acceptance','evidence_sha256':{str(p):sha(p) for p in [bp,hp,ip,ep/'hybrid-http-encoder-2876.jsonl',ep/'hybrid-http-cost-index-2876.jsonl']},'summaries':summaries,'changes':changes,'encoding_costs':encoder,'index_roundtrip_costs':proxy,'cpu_gate_cost':cost,'vector_proof':vector_proof,'conclusion':'supports limited Tree top3 retrieval candidate acceptance for independently reviewed two-update smoke; not production500/effect/global-query/unique512 throughput acceptance'}
(r/'work/aux-gate-review/hybrid-real-independent-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'summaries':summaries,'cpu_gate_cost':cost,'vector_proof':vector_proof},indent=2))
