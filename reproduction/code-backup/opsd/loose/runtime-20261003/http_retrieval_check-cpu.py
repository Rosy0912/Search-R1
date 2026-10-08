import hashlib,json,math,os,time
from pathlib import Path
import requests
out=Path(os.environ['OPSD_EVIDENCE']); comparison=json.loads((out/'retrieval-comparison.json').read_text()); queries=comparison['queries']
for attempt in range(240):
    try:
        response=requests.post('http://127.0.0.1:18084/retrieve',json={'queries':queries[:1],'topk':3,'return_scores':True},timeout=30);response.raise_for_status();break
    except requests.RequestException: time.sleep(5)
else: raise RuntimeError('CPU retrieval HTTP service failed to become ready')
results={}
for k in (3,50):
    start=time.perf_counter();r=requests.post('http://127.0.0.1:18084/retrieve',json={'queries':queries,'topk':k,'return_scores':True},timeout=180);r.raise_for_status();data=r.json()['result']
    hashes=[[hashlib.sha256(x['document']['contents'].encode()).hexdigest() for x in row] for row in data]
    assert hashes==comparison['content'][str(k)]['cpu'], 'Live service differs from CPU comparison'
    scores=[[float(x['score']) for x in row] for row in data]
    assert all(math.isfinite(v) for row in scores for v in row)
    delta=max(abs(v-comparison['cpu_scores'][i][j]) for i,row in enumerate(scores) for j,v in enumerate(row))
    assert delta<1e-5, 'HTTP numerical score differs from in-process CPU reference'
    results[str(k)]=dict(seconds=time.perf_counter()-start,result=data,ordered_content_equals_cpu=True, score_max_abs_delta=delta)
(out/'retrieval-http.json').write_text(json.dumps(results,indent=2));print('REAL_HTTP_CPU_RETRIEVAL_TOP3_TOP50_OK',flush=True)
