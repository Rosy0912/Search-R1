import pathlib,json,hashlib,importlib.util,math
r=pathlib.Path(__file__).resolve().parents[2];d=r/'work/aux-gate-review';e=r/'outputs/evidence';base=json.loads((e/'retrieval-cpu-2854.json').read_text());binary=json.loads((e/'hybrid-binary-index-difference.json').read_text());http=json.loads((e/'hybrid-http-difference-2878.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();proof=json.loads((d/'saved-2854-vector-hash-proof.json').read_text());spec=importlib.util.spec_from_file_location('checker',r/'work/check_hybrid_retriever.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.validate_rows(http,'http');m.validate_rows(binary,'binary')
assert http['provenance']['index_instance']==binary['instance'] and http['provenance']['index_instance']['pid']==774407
assert http['baseline_sha256']==binary['baseline_sha256']==sha(e/'retrieval-cpu-2854.json')
assert http['provenance']['source_sha256']==sha(r/'work/hybrid_retrieval_server.py')=='b057a34876036c1e7bc8005ccc23e0aa10c170d0dca77c2f71782c4b7c9b4429'
assert http['provenance']['model_sha256']=={k:v for k,v in base['asset_identity']['model'].items() if k not in m.IGNORED} and http['provenance']['asset_identity']==base['asset_identity']
summary=[];changes=[]
for row in http['rows']:
 n,k=row['query_count'],row['k'];assert row['repeats'][0]==row['repeats'][1];exact=content=sets=0;maximum=0
 for x in row['repeats'][0]:
  wanted=base['results'][x['query_index']][str(k)];ids=x['ids'];scores=x['scores'];idsame=ids==wanted['ids'];contentsame=x['contents_sha256']==wanted['contents_sha256'];got=dict(zip(ids,scores));common=set(ids)&set(wanted['ids']);delta=max(abs(got[i]-wanted['scores'][wanted['ids'].index(i)]) for i in common)
  assert idsame==x['exact_rank'] and contentsame==x['exact_content_rank'] and len(common)==x['overlap'] and delta==x['max_shared_score_delta'];exact+=idsame;content+=contentsame;sets+=len(common)==k;maximum=max(maximum,delta)
  reference=dict(zip(wanted['ids'],wanted['contents_sha256']));current=dict(zip(ids,x['contents_sha256']));assert all(reference[i]==current[i] for i in common)
  if n==14 and not idsame:changes.append({'query_index':x['query_index'],'query':base['queries'][x['query_index']],'k':k,'changed_ranks_1based':[j+1 for j in range(k) if ids[j]!=wanted['ids'][j]],'content_rank_equal':contentsame})
 summary.append({'n':n,'k':k,'exact_ids':exact,'exact_content':content,'same_sets':sets,'repeats_stable':True,'max_matched_score_delta':maximum,'wall_seconds':row['timings_seconds']})
encoder=[json.loads(l) for l in (e/'hybrid-http-encoder-2878.jsonl').read_text().splitlines()];assert len(encoder)>=6;fixed=encoder[:6];hashes={g['n']:g['vectors_sha256'] for g in proof['groups']};assert [x['queries'] for x in fixed]==[14,14,14,14,512,512] and all(x['vectors_sha256']==hashes[x['queries']] for x in fixed)
result={'status':'DRAFT_FULL_HTTP_REVIEW_WAITING_BUSY_AND_CPU_JOB_EXIT','not_acceptance':True,'summary':summary,'top50_changes':changes,'encoder_parity_costs':fixed,'frozen_binary_instance':binary['instance'],'evidence_sha256':{str(p):sha(p) for p in [e/'retrieval-cpu-2854.json',e/'hybrid-http-difference-2878.json',e/'hybrid-http-encoder-2878.jsonl',e/'hybrid-binary-index-difference.json']},'remaining':['actual successful busy-health report and matching rawsamples','actual CPU job exit0/cost','final versioned async launcher paths/source']}
(d/'async-real-http-partial-2878.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'summary':summary,'top50_changes':changes,'remaining':result['remaining']},indent=2))
