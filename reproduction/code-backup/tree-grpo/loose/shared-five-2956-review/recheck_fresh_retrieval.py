import json,pathlib,hashlib
r=pathlib.Path(__file__).resolve().parents[2];b=json.loads((r/'outputs/evidence/shared-original-gpu-2854.json').read_text());p=r/'outputs/evidence/shared-five-2956/retrieval-gate.json';x=json.loads(p.read_text());assert x['asset_identity_verified'] and x['baseline_sha256']==hashlib.sha256((r/'outputs/evidence/shared-original-gpu-2854.json').read_bytes()).hexdigest() and x['source_sha256']==hashlib.sha256((r/'work/probe_tree_shared_retrieval.py').read_bytes()).hexdigest()
by={};deltas=[]
for m in x['measurements']:
 key=(m.get('label','fixed14'),m['queries'],m['k']);prev=by.setdefault(key,m['results']);assert prev==m['results']
 if m['queries']==14:
  for i,a in enumerate(m['results']):
   ref=b['results'][i][str(m['k'])];assert a['corpus_ids']==ref['corpus_ids'] and a['contents_sha256']==ref['contents_sha256'] and a['scores']==ref['scores']
 elif 'capacity' not in m.get('label','').lower():
  for i,a in enumerate(m['results']):
   ref=b['results'][i%14][str(m['k'])];assert a['corpus_ids']==ref['corpus_ids'] and a['contents_sha256']==ref['contents_sha256'];deltas += [abs(v-w) for v,w in zip(a['scores'],ref['scores'])]
assert x['over_512_rows']==513
out=dict(status='RAW_FRESH_RETRIEVAL_RECHECK_PASS',fixed14_exact_ids_scores_contents=True,repeat_results_all_exact=True,repeated512_ID_content_exact=True,repeated512_max_score_delta=max(deltas),distinct_measurement_groups=list(map(str,by)),empty_and_513_endpoint_scope='probe source binds existing schema; over512 correctly processed513rows, not falsely rejected')
(r/'work/shared-five-2956-review/fresh-retrieval-recheck.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
