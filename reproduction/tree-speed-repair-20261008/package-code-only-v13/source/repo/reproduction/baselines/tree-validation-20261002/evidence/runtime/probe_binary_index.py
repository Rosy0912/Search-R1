"""Real binary vector transport/index difference measurement; no auto acceptance."""
import argparse,pathlib,json,time,hashlib,io,requests,numpy as np
p=argparse.ArgumentParser();p.add_argument('--url',required=True);p.add_argument('--reference',type=pathlib.Path,required=True);p.add_argument('--embeddings',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--instance-out',type=pathlib.Path,required=True);a=p.parse_args()
assert not a.output.exists() and not a.instance_out.exists()
base=json.loads(a.reference.read_text());payload=np.load(a.embeddings,allow_pickle=False);emb=np.ascontiguousarray(payload['embeddings'],dtype=np.float32)
assert emb.shape==(14,768) and np.isfinite(emb).all() and str(payload['queries_sha256'].item())==base['queries_sha256'] and hashlib.sha256(emb.tobytes()).hexdigest()==base['embeddings_sha256']
r=requests.get(a.url+'/provenance',timeout=30);r.raise_for_status();instance=r.json();assert instance['mode']=='fp32-index-only' and instance['index_float16'] is False and instance['index_sha256']==base['asset_identity']['index']['sha256']
rows=[]
for n in [14,512]:
 for k in [3,50]:
  q=np.ascontiguousarray(emb[np.arange(n)%14]);frame=io.BytesIO();np.save(frame,q,allow_pickle=False);raw=frame.getvalue();repeats=[];times=[]
  for repeat in range(2):
   t=time.perf_counter();r=requests.post(a.url+'/index_search',params={'k':k},data=raw,headers={'Content-Type':'application/x-npy'},timeout=120);r.raise_for_status();out=r.json();times.append(time.perf_counter()-t)
   assert out['instance']==instance and out['k']==k and out['vectors_sha256']==hashlib.sha256(q.tobytes()).hexdigest() and out['request_sha256']==hashlib.sha256(raw).hexdigest()
   ids=np.asarray(out['ids']);scores=np.asarray(out['scores']);assert ids.shape==scores.shape==(n,k) and ids.dtype.kind in 'iu' and np.isfinite(scores).all() and (ids>=0).all() and (ids<instance['ntotal']).all() and all(len(set(v.tolist()))==k for v in ids)
   details=[]
   for i in range(n):
    got=ids[i].tolist();wanted=base['results'][i%14][str(k)];byid=dict(zip(got,scores[i].tolist()));common=set(got)&set(wanted['ids'])
    details.append({'row':i,'query_index':i%14,'exact_rank':got==wanted['ids'],'overlap':len(common),'ids':got,'scores':scores[i].tolist(),'max_shared_score_delta':max((abs(byid[v]-wanted['scores'][wanted['ids'].index(v)]) for v in common),default=None)})
   repeats.append(details)
  rows.append({'queries':n,'unique_queries':14,'k':k,'times_seconds':times,'repeat_equal':repeats[0]==repeats[1],'repeats':repeats,'exact_rank_rows':sum(v['exact_rank'] for v in repeats[0])})
r=requests.get(a.url+'/provenance',timeout=30);r.raise_for_status();assert r.json()==instance
report={'status':'DIFFERENCE_REPORT_ONLY','scope':'CPU2854 saved vectors, actual binary transport and exact FP32 GPU index; repeated512 is not unique query throughput','instance':instance,'baseline_sha256':hashlib.sha256(a.reference.read_bytes()).hexdigest(),'embedding_file_sha256':hashlib.sha256(a.embeddings.read_bytes()).hexdigest(),'vectors_sha256':base['embeddings_sha256'],'source_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'rows':rows}
with a.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
with a.instance_out.open('x') as f:json.dump(instance,f,indent=2);f.write('\n')
print(json.dumps({'status':report['status'],'summary':[{key:value for key,value in row.items() if key!='repeats'} for row in rows]}))
