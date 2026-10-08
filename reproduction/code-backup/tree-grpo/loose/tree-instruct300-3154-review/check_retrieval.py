import json,hashlib,math
from pathlib import Path
O=Path(__file__).resolve().parent;ROOT=O.parents[1];P=O/'actual3154-files/runs/tree-instruct300-3154';j=json.loads((P/'retrieval-gate.json').read_text());refp=ROOT/'outputs/evidence/shared-original-gpu-2854.json';ref=json.loads(refp.read_text());assert hashlib.sha256(refp.read_bytes()).hexdigest()==j['baseline_sha256']=='854bef8268f8ec5cbe327719ea4b43de34d1097cd683d751045eeb0f99a7cabb';assert j['asset_identity_verified'] and j['source_sha256']=='335a3410bdc19fac01d5d29aa7eea3832c031fae5093f1f9f2d65f76374157b5';assert j['actual_asset_identity']['index']['sha256']=='69c98463fdb41fc08737d88513c597725f311c44f7ba6dca4b05d8c7c658d166' and j['actual_asset_identity']['corpus']['sha256']=='85a787a692e73fdd657e86411bfd1ac810ed193b1eea4a478de449d4871c06b9'
ms=j['measurements'];assert len(ms)==10;rows=[]
for m in ms:
 assert len(m['results'])==m['queries'] and math.isfinite(m['wall_seconds']) and m['wall_seconds']>0
 for r in m['results']:assert len(r['corpus_ids'])==len(r['scores'])==len(r['contents_sha256'])==m['k'] and all(math.isfinite(x) for x in r['scores'])
 if m['queries']==14:
  for i,r in enumerate(m['results']):
   b=ref['results'][i][str(m['k'])];assert r['corpus_ids']==b['corpus_ids'] and r['contents_sha256']==b['contents_sha256'] and r['scores']==b['scores']
 elif m['label'] in ['repeated_512','repeated_512_top50']:
  for i,r in enumerate(m['results']):
   b=ref['results'][i%14][str(m['k'])];assert r['corpus_ids']==b['corpus_ids'] and r['contents_sha256']==b['contents_sha256']
  assert all(h['instance_equal'] for h in m['concurrent_health_samples'])
 else:assert m['label']=='capacity_512_max256' and m['is_synthetic_capacity_probe'] and all(h['instance_equal'] for h in m['concurrent_health_samples'])
 rows.append({'queries':m['queries'],'label':m.get('label','fixed14'),'k':m['k'],'repeat':m['repeat'],'wall_seconds':m['wall_seconds']})
for m in ms:
 if m['repeat']==0:
  other=next(x for x in ms if x['repeat']==1 and x['k']==m['k'] and x.get('label')==m.get('label') and x['queries']==m['queries']);assert m['results']==other['results']
assert j['over_512_rows']==513
(O/'RETRIEVAL_PREFIX_RECOMPUTED.json').write_text(json.dumps({'status':'PASS_ACTUAL_LIMITED_RETRIEVAL_ONLY','job_id':'3154','fixed14_k3_k50_exact_ID_content_score':True,'repeated512_ID_content_exact_to_original14':True,'two_repeats_exact':True,'measurements':rows,'raw_sha256':hashlib.sha256((P/'retrieval-gate.json').read_bytes()).hexdigest(),'limits':['512 repeats and max256 synthetic are not512 unique natural throughput','14-query exact parity does not prove allqueries exact']},indent=2));print('all10 measurement shapes/finite, original14 exact, repeated512 IDs/content and exact repeat verified')
