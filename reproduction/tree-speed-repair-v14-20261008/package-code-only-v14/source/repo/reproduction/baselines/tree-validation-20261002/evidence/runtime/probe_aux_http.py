"""Real HTTP fixed-query and batching difference gate; never accepts training."""
import argparse,pathlib,json,hashlib,time,math,urllib.request
p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:28082');p.add_argument('--reference',type=pathlib.Path,default=pathlib.Path('outputs/evidence/retrieval-cpu-2854.json'));p.add_argument('--output',required=True,type=pathlib.Path);a=p.parse_args()
assert not a.output.exists()
base=json.loads(a.reference.read_text());rows=[]
with urllib.request.urlopen(a.url+'/provenance',timeout=180) as response:provenance=json.load(response)
for count,k in [(14,3),(14,50),(512,3)]:
 queries=[base['queries'][i%14] for i in range(count)]
 payload=json.dumps({'queries':queries,'topk':k,'return_scores':True}).encode();repeated=[];timings=[]
 for repeat in range(2):
  t=time.perf_counter();req=urllib.request.Request(a.url+'/retrieve',data=payload,headers={'Content-Type':'application/json'},method='POST')
  with urllib.request.urlopen(req,timeout=300) as response:out=json.load(response)
  timings.append(time.perf_counter()-t);assert set(out)=={'result'} and len(out['result'])==count
  data=[]
  for i,documents in enumerate(out['result']):
   assert len(documents)==k
   ids=[];scores=[];hashes=[]
   for item in documents:
    assert set(item)=={'document','score'} and math.isfinite(item['score'])
    doc=item['document'];ids.append(int(doc['id']));scores.append(item['score']);hashes.append(hashlib.sha256(doc['contents'].encode()).hexdigest())
   assert len(set(ids))==k
   wanted=base['results'][i%14][str(k)];current=dict(zip(ids,scores));common=set(ids)&set(wanted['ids'])
   data.append({'row':i,'query_index':i%14,'ids':ids,'scores':scores,'contents_sha256':hashes,'exact_rank':ids==wanted['ids'],'exact_content_rank':hashes==wanted['contents_sha256'],'overlap':len(common),'max_shared_score_delta':max((abs(current[v]-wanted['scores'][wanted['ids'].index(v)]) for v in common),default=None)})
  repeated.append(data)
 repeat_equal=repeated[0]==repeated[1]
 data=repeated[0]
 rows.append({'query_count':count,'unique_queries':14,'k':k,'timings_seconds':timings,'repeat_equal':repeat_equal,'repeats':repeated,'exact_rank_rows':sum(v['exact_rank'] for v in data),'exact_content_rank_rows':sum(v['exact_content_rank'] for v in data),'details':data})
with urllib.request.urlopen(a.url+'/provenance',timeout=180) as response:assert json.load(response)==provenance
report={'status':'DIFFERENCE_REPORT_ONLY','scope':'actual HTTP and CPU encoding, 14 fixed real questions repeated to512; no training/effect/global equivalence assertion','url':a.url,'provenance':provenance,'baseline_sha256':hashlib.sha256(a.reference.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'queries_sha256':base['queries_sha256'],'rows':rows}
with a.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps({'status':report['status'],'rows':[{k:v for k,v in r.items() if k not in ['details','repeats']} for r in rows]}))
