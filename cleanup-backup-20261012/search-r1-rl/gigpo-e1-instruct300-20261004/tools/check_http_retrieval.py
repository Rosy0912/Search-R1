"""Compare own local HTTP results with the accepted original 8-GPU FP16 reference."""
import hashlib,json,math,os,time
from pathlib import Path
import requests
root=Path(os.environ['AGENT_ROOT'])
reference=json.loads((root/'logs/cpu-gpu-retrieval-comparison.json').read_text())
session=requests.Session();session.trust_env=False
records=[]
for k in (3,50):
 start=time.perf_counter()
 response=session.post('http://127.0.0.1:18081/retrieve',
     json={'queries':reference['queries'],'topk':k,'return_scores':True},timeout=(5,300))
 response.raise_for_status()
 actual=response.json()['result']
 expected=reference['gpu_top3_documents' if k==3 else 'gpu_top50_documents']
 scores=reference['gpu_top3_scores' if k==3 else 'gpu_scores']
 assert len(actual)==len(expected)
 maximum=0.
 for row,docs,values in zip(actual,expected,scores):
  assert len(row)==len(docs)==len(values)==k
  for item,doc,value in zip(row,docs,values):
   assert item['document']==doc,('HTTP document/order mismatch',k)
   actual_score,expected_score=float(item['score']),float(value)
   assert math.isfinite(actual_score) and math.isfinite(expected_score)
   delta=abs(actual_score-expected_score)
   assert math.isfinite(delta)
   maximum=max(maximum,delta)
 assert maximum<1e-5,maximum
 records.append({'topk':k,'queries':len(actual),'document_and_order_equal':True,
                 'maximum_score_delta':maximum,'wall_seconds':time.perf_counter()-start})
record={'job_id':os.environ['SLURM_JOB_ID'],'host':os.uname().nodename,
    'url':'http://127.0.0.1:18081/retrieve','proxy_used':False,'records':records,
    'reference_sha256':hashlib.sha256((root/'logs/cpu-gpu-retrieval-comparison.json').read_bytes()).hexdigest()}
Path(os.environ.get('SEARCH_R1_HTTP_RECEIPT',str(root/'logs/local-http-retrieval-verification.json'))).write_text(json.dumps(record,indent=2))
print('REAL_LOCAL_HTTP_TOP3_TOP50_REFERENCE_PASS',flush=True)
