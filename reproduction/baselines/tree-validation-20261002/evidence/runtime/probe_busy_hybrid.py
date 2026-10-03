"""Real max-length workload and concurrent health; no answer/parity/effect claim."""
import argparse,pathlib,json,time,concurrent.futures,hashlib,requests,math
p=argparse.ArgumentParser();p.add_argument('--baseline',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args();assert not a.output.exists()
base=json.loads(a.baseline.read_text());url='http://127.0.0.1:18082'
r=requests.get(url+'/provenance',timeout=10);r.raise_for_status();expected=r.json()
# Deliberately repeat question text to exercise the unchanged256-token encoder
# cap. These are synthetic long inputs, not training/evaluation examples.
queries=[' '.join([base['queries'][i%14]]*128) for i in range(512)]
def retrieve():
 t=time.perf_counter();r=requests.post(url+'/retrieve',json={'queries':queries,'topk':3,'return_scores':True},timeout=900);r.raise_for_status();out=r.json()
 assert set(out)=={'result'} and len(out['result'])==512
 for docs in out['result']:
  assert len(docs)==3 and len({int(v['document']['id']) for v in docs})==3 and all(math.isfinite(v['score']) for v in docs)
 return {'wall_seconds':time.perf_counter()-t,'response_sha256':hashlib.sha256(r.content).hexdigest(),'rows':512,'topk':3}
samples=[]
sample_path=a.output.with_suffix('.samples.jsonl');assert not sample_path.exists()
sample_path.touch()
with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
 future=pool.submit(retrieve)
 phase='concurrent-health'
 try:
  while not future.done():
   t=time.perf_counter();r=requests.get(url+'/provenance',timeout=10);r.raise_for_status();assert r.json()==expected
   sample={'epoch':time.time(),'health_wall_seconds':time.perf_counter()-t};samples.append(sample)
   with sample_path.open('a') as f:f.write(json.dumps(sample)+'\n')
   time.sleep(1)
  phase='retrieve-result';result=future.result()
 except BaseException as error:
  with a.output.open('x') as f:json.dump({'status':'FAILED_BUSY_HEALTH_MEASUREMENT','phase':phase,'failure':repr(error),'epoch':time.time(),'provenance':expected,'samples':samples,'source_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'scope':'Failed measurement, not acceptance. HTTP worker may still run until900s transport deadline/own Slurm cleanup.'},f,indent=2);f.write('\n')
  raise
assert len(samples)>=2,'No actual concurrent busy health samples'
r=requests.get(url+'/provenance',timeout=10);r.raise_for_status();assert r.json()==expected
report={'status':'BUSY_HEALTH_MEASUREMENT_ONLY','scope':'512 synthetic long inputs repeated from14 fixed questions; unchanged256-token E5 cap; not training/effect/CPU-index numerical parity proof','provenance':expected,'source_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'baseline_sha256':hashlib.sha256(a.baseline.read_bytes()).hexdigest(),'request_queries_sha256':hashlib.sha256(json.dumps(queries,ensure_ascii=False).encode()).hexdigest(),'samples':samples,'result':result}
with a.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps({'status':report['status'],'result':result,'health_samples':len(samples),'maximum_health_wall_seconds':max(v['health_wall_seconds'] for v in samples)}))
