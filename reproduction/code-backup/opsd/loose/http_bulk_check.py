import hashlib,json,math,time,requests
from pathlib import Path
out=Path('/mnt/data/shaoshitong/search-r1-rl/opsd/runs/opsd-acceptance-2875/evidence')
c=json.loads((out/'retrieval-comparison.json').read_text());q=c['queries']*3
records={}
for k in (3,50):
 start=time.monotonic();r=requests.post('http://127.0.0.1:18084/retrieve',json={'queries':q,'topk':k,'return_scores':True},timeout=180);r.raise_for_status();rows=r.json()['result'];assert len(rows)==24
 hashes=[[hashlib.sha256(x['document']['contents'].encode()).hexdigest() for x in row] for row in rows]
 scores=[[float(x['score']) for x in row] for row in rows]
 records[str(k)]={'seconds':time.monotonic()-start,'result':rows,'finite':all(math.isfinite(s) for row in scores for s in row),'ordered_content_equals_small_cpu':hashes==c['content'][str(k)]['cpu']*3,'small_cpu_score_max_abs_delta':max(abs(s-c['cpu_scores'][i%8][j]) for i,row in enumerate(scores) for j,s in enumerate(row))}
(out/'retrieval-http-bulk24.json').write_text(json.dumps({'job_id':'2875','queries':q,'diagnostic_only':True,'results':records},indent=2))
print(json.dumps({k:{a:b for a,b in v.items() if a!='result'} for k,v in records.items()},indent=2))
