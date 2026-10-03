"""Fixed24 diagnostic query entries (8 queries repeated3); no training trajectory count."""
import hashlib,json,math,os,time
from pathlib import Path
import requests
out=Path(os.environ['OPSD_EVIDENCE'])
comparison=json.loads((out/'retrieval-comparison.json').read_text())
queries=comparison['queries']*3
start=time.perf_counter()
r=requests.post('http://127.0.0.1:18084/retrieve',json={'queries':queries,'topk':3,'return_scores':True},timeout=180)
r.raise_for_status(); results=r.json()['result']
assert len(results)==24 and all(len(x)==3 for x in results)
reference=json.loads((out/'retrieval-http.json').read_text())['3']['result']*3
max_delta=0.0
for row, expected in zip(results,reference):
    for item, ref in zip(row,expected):
        assert item['document']==ref['document'], 'Bulk24 ordered ID/content differs from GPU reference'
        value=float(item['score']); baseline=float(ref['score'])
        assert math.isfinite(value) and math.isfinite(baseline)
        max_delta=max(max_delta,abs(value-baseline))
assert max_delta<1e-5, 'Bulk24 numerical scores differ from live GPU reference'
record=dict(job_id=os.environ['SLURM_JOB_ID'],query_entries=24,unique_queries=8,topk=3,
 seconds=time.perf_counter()-start,results=results,ordered_documents_equal_gpu_reference=True,score_max_abs_delta=max_delta,scope='Isolated live same-node GPU HTTP retrieval diagnostic; not a train outer or generation throughput benchmark')
(out/'gpu-retrieval-bulk24.json').write_text(json.dumps(record,indent=2))
print('GPU_RETRIEVAL_BULK24_SECONDS',record['seconds'],flush=True)
