"""Read-only, real HTTP retrieval gate; no training/gold answers or GPU use here."""
import argparse, concurrent.futures, hashlib, json, math, pathlib, time, urllib.request
DIRECT=urllib.request.build_opener(urllib.request.ProxyHandler({}))
p=argparse.ArgumentParser(); p.add_argument('--baseline',type=pathlib.Path,default=pathlib.Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/retrieval-2854/gpu.json')); p.add_argument('--output',type=pathlib.Path,required=True); a=p.parse_args()
url='http://127.0.0.1:18082'
assert not a.output.exists()
def get():
    with DIRECT.open(url+'/openapi.json',timeout=10) as f:return json.load(f)
def post(queries,k):
    data=json.dumps({'queries':queries,'topk':k,'return_scores':True}).encode()
    t=time.perf_counter()
    with DIRECT.open(urllib.request.Request(url+'/retrieve',data=data,headers={'Content-Type':'application/json'}),timeout=600) as f: r=json.load(f)
    assert set(r)=={'result'} and len(r['result'])==len(queries)
    rows=[]
    for row in r['result']:
        assert len(row)==k and all(set(v)=={'document','score'} for v in row)
        ids=[str(v['document']['id']) for v in row]; scores=[v['score'] for v in row]
        assert len(set(ids))==k and all(type(s) in (int,float) and math.isfinite(s) for s in scores)
        rows.append({'corpus_ids':ids,'scores':scores,'contents_sha256':[hashlib.sha256(v['document']['contents'].encode()).hexdigest() for v in row]})
    return rows,time.perf_counter()-t
baseline=json.loads(a.baseline.read_text()); instance=get()
assert baseline['asset_identity_verified'] and baseline['mode']=='gpu'
assert baseline['index_metadata']['ntotal']==21015324 and baseline['index_metadata']['gpu_shard'] and baseline['index_metadata']['gpu_use_float16']
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
assets={}
for kind,name in [('index','e5_Flat.index'),('corpus','wiki-18.jsonl')]:
    file=root/'assets-root/assets/wiki'/name
    assets[kind]={'size':file.stat().st_size,'sha256':sha(file)}
    assert assets[kind]==baseline['asset_identity'][kind], 'Actual asset bytes changed'
for name,expected in baseline['asset_identity']['model'].items():
    if 'partial' not in name and not name.startswith('.'): assert sha(root/'assets-root/assets/e5-base-v2'/name)==expected
assert sha(root/'repo/search_r1/search/retrieval_server.py')==baseline['source']['retrieval_server_sha256']

assert '/retrieve' in instance['paths']
measurements=[]; differences=[]; fixed={}
for k in (3,50):
    first=None
    for repeat in range(2):
        rows,seconds=post(baseline['queries'],k)
        if first is None:first=rows
        else:assert rows==first,'Repeated fixed query result changed'
        measurements.append({'queries':14,'unique_queries':14,'k':k,'repeat':repeat,'wall_seconds':seconds,'results':rows})
    fixed[k]=first
    for i,row in enumerate(first):
        old=baseline['results'][i][str(k)]; left=dict(zip(old['corpus_ids'],zip(old['scores'],old['contents_sha256']))); right=dict(zip(row['corpus_ids'],zip(row['scores'],row['contents_sha256']))); shared=set(left)&set(right)
        differences.append({'query_index':i,'k':k,'exact_rank':row['corpus_ids']==old['corpus_ids'],'overlap':len(shared),'all_shared_content_equal':all(left[x][1]==right[x][1] for x in shared),'max_matched_score_delta':max((abs(left[x][0]-right[x][0]) for x in shared),default=None),'only_reference':sorted(set(left)-set(right)),'only_gpu':sorted(set(right)-set(left))})
executor=concurrent.futures.ThreadPoolExecutor(max_workers=1)
try:
    for label,queries,k in [('repeated_512',[baseline['queries'][i%14] for i in range(512)],3),('repeated_512_top50',[baseline['queries'][i%14] for i in range(512)],50),('capacity_512_max256',[' '.join([baseline['queries'][i%14]]*64) for i in range(512)],3)]:
        first=None
        for repeat in range(2):
            future=executor.submit(post,queries,k); health=[]
            while not future.done():
                t=time.perf_counter(); assert get()==instance; health.append({'epoch':time.time(),'wall_seconds':time.perf_counter()-t,'instance_equal':True}); time.sleep(.2)
            rows,seconds=future.result()
            if first is None:first=rows
            else:assert rows==first,'Repeated 512 result changed'
            compared=[]
            if not label.startswith('capacity'):
                for i,row in enumerate(rows):
                    ref=fixed[k][i%14]; old=baseline['results'][i%14][str(k)]
                    compared.append({'row':i,'source_row':i%14,'exact_rank_vs_same_gpu14':row['corpus_ids']==ref['corpus_ids'],'exact_rank_vs_reference14':row['corpus_ids']==old['corpus_ids'],'content_rank_equal_vs_same_gpu14':row['contents_sha256']==ref['contents_sha256'],'max_position_score_delta_vs_same_gpu14':max(abs(x-y) for x,y in zip(row['scores'],ref['scores']))})
            measurements.append({'label':label,'repeat':repeat,'queries':512,'unique_source_queries':14,'is_synthetic_capacity_probe':label.startswith('capacity'),'k':k,'wall_seconds':seconds,'concurrent_health_samples':health,'results':rows,'compared_to_14':compared,'results_sha256':hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()})
except BaseException as err:
    with a.output.with_suffix(a.output.suffix+'.failed').open('x') as f:json.dump({'status':'PROBE_FAILED','error':repr(err),'instance':instance,'completed_measurements':measurements},f,indent=2)
    raise
finally:executor.shutdown()
assert get()==instance
empty,empty_seconds=post([],3)
assert empty==[]
large,large_seconds=post([baseline['queries'][i%14] for i in range(513)],3)
assert len(large)==513
report={'asset_identity_verified':True,'actual_asset_identity':assets,'empty_query_seconds':empty_seconds,'over_512_query_seconds':large_seconds,'over_512_rows':len(large),'status':'DIFFERENCE_REPORT_ONLY','scope':'14 fixed real queries, two exact repeats; 512 repeated and max256 capacity probes are not natural unique-query throughput','instance':instance,'measurements':measurements,'differences':differences,'source_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'baseline_sha256':hashlib.sha256(a.baseline.read_bytes()).hexdigest(),'reference_mode':'original 8GPU FP16 sharded index, GPU E5; same bytes and retrieval source'}
with a.output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
print(json.dumps({'status':report['status'],'output':str(a.output),'top3_exact_rank_queries':sum(r['exact_rank'] for r in differences if r['k']==3),'top50_exact_rank_queries':sum(r['exact_rank'] for r in differences if r['k']==50),'measurements':[{k:v for k,v in r.items() if k not in ('results','compared_to_14','concurrent_health_samples')} for r in measurements]}),flush=True)
