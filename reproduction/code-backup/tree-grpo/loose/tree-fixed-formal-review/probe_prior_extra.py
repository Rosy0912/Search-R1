from pathlib import Path
s=(Path(__file__).parent/'probe_formal_gates.py').read_text();exec(s[:s.index("(OUT/'FIVE_GATE_CASES.json')")])
extra=[]
for name in ['outside_proof','outside_review','empty_proof','empty_reviews','wrong_receipt_sha']:
 with tempfile.TemporaryDirectory() as d:
  x=copy.deepcopy(receipt)
  if name=='outside_proof':x['evidence'][0]['path']=str(REMOTE/'runs/not-owned/file.json')
  if name=='outside_review':x['small_review_bundle_mapping'][0]['path']=str(REMOTE/'logs/not-owned/file.md')
  if name=='empty_proof':x['evidence']=[]
  if name=='empty_reviews':x['small_review_bundle_mapping']=[]
  p=pathlib.Path(d)/'receipt.json';p.write_text(json.dumps(x));a=types.SimpleNamespace(actual_five_review=p,actual_five_review_sha='0'*64 if name=='wrong_receipt_sha' else sha(p));err=None
  try:exec(code,dict(a=a,ROOT=REMOTE,COMMIT=m['code_commit'],Path=pathlib.Path,sha=proof_sha,require=require,json=json))
  except (AssertionError,FileNotFoundError) as e:err=str(e)
  assert err is not None,(name,err);extra.append({'case':name,'rejected':True,'error':err})
assert results[0]['error'] is None and all(x['error'] is not None for x in results[1:])
(OUT/'FIVE_GATE_EXTRA_CASES.json').write_text(json.dumps({'source_sha256':sha(TOOLS/'supervisor.py'),'status':'PASS_LOCAL_SEMANTIC_NEGATIVES_ONLY','cases':extra,'prior_evidence_actual_paths_all_runs':all('/runs/' in x['path'] for x in receipt['evidence']),'scope':'Same exact block, fixture digest recomputed; actual wrapper immutable4dac receipt pin remains'},indent=2));print('17 prior cases plus5 extra cases passed; actualreceipt129 paths allruns')
