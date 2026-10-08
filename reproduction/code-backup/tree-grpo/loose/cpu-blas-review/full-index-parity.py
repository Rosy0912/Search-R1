"""CPU-only full-index measurement tool. Never called by production launcher.
Repeat-512 rows cover batching/BLAS behavior, not 512 independent queries.
"""
import argparse,ctypes,hashlib,json,math,os,pathlib,resource,time
os.environ['CUDA_VISIBLE_DEVICES']=''
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def blas_identity(faiss):
 suffix='avx512' if 'AVX512' in faiss.get_compile_options() else 'avx2' if 'AVX2' in faiss.get_compile_options() else ''
 library=next(pathlib.Path(faiss.__file__).parent.glob('_swigfaiss'+('_'+suffix if suffix else '')+'.*.so'))
 handle=ctypes.CDLL(str(library));meta={'faiss_extension':str(library),'ld_preload':os.environ.get('LD_PRELOAD')}
 class DlInfo(ctypes.Structure):_fields_=[('file',ctypes.c_char_p),('base',ctypes.c_void_p),('symbol',ctypes.c_char_p),('address',ctypes.c_void_p)]
 dl=ctypes.CDLL('libdl.so.2');dl.dladdr.argtypes=[ctypes.c_void_p,ctypes.POINTER(DlInfo)]
 for name in ['sgemm_','openblas_get_config','openblas_get_corename','openblas_get_num_threads']:
  f=getattr(handle,name,None)
  if f:
   info=DlInfo();dl.dladdr(ctypes.cast(f,ctypes.c_void_p),ctypes.byref(info));meta[name+'_symbol_lookup_library']=info.file.decode() if info.file else None
   if name!='sgemm_':
    f.restype=ctypes.c_int if name.endswith('threads') else ctypes.c_char_p;v=f();meta[name]=v.decode() if isinstance(v,bytes) else v
 global_handle=ctypes.CDLL(None);meta['global_openblas_identity']={}
 for name in ['openblas_get_config','openblas_get_corename','openblas_get_num_threads']:
  f=getattr(global_handle,name,None)
  if f:
   f.restype=ctypes.c_int if name.endswith('threads') else ctypes.c_char_p;v=f();meta['global_openblas_identity'][name]=v.decode() if isinstance(v,bytes) else v
 meta['note']='dladdr shows symbol lookup, use LD_DEBUG=bindings trace to prove actual FAISS SGEMM relocation'
 return meta

def measure(a):
 assert a.allow_large_index,'Explicit --allow-large-index required; schedule a dedicated CPU task first'
 assert not a.output.exists(),'Refuse overwrite'
 total_start=time.perf_counter();total_cpu=time.process_time()
 import faiss,numpy as np
 faiss.omp_set_num_threads(a.threads)
 assert a.expected_index_sha256 and sha(a.index)==a.expected_index_sha256,'Index identity mismatch'
 original=json.loads(a.reference_cpu.read_text());assert original['mode']=='cpu' and original['asset_identity_verified']
 assert original['asset_identity']['index']['sha256']==a.expected_index_sha256,'Index differs from reviewed CPU measurement'
 payload=np.load(a.embeddings,allow_pickle=False);emb=np.ascontiguousarray(payload['embeddings'],dtype=np.float32)
 assert emb.shape==(len(original['queries']),768) and len(emb)==14 and np.isfinite(emb).all()
 assert str(payload['queries_sha256'].item())==original['queries_sha256'] and hashlib.sha256(emb.tobytes()).hexdigest()==original['embeddings_sha256'],'CPU embedding identity mismatch'
 t=time.perf_counter();cpu=time.process_time();index=faiss.read_index(str(a.index));load={'wall_seconds':time.perf_counter()-t,'process_cpu_seconds':time.process_time()-cpu}
 assert index.d==768 and index.ntotal==21015324 and type(index).__name__=='IndexFlatIP','Wrong production index'
 rows=[];threshold=int(faiss.cvar.distance_compute_blas_threshold)
 assert 512>=threshold,'512 does not cover current FAISS BLAS threshold'
 for count in [14,512]:
  source_rows=np.arange(count)%14;queries=np.ascontiguousarray(emb[source_rows]);saved={};timings=[]
  for repeat in range(a.repeats):
   start=time.perf_counter();process=time.process_time();current={}
   for k in [3,50]:
    scores,ids=index.search(queries,k);assert (ids>=0).all() and np.isfinite(scores).all()
    current[str(k)]={'ids':ids.tolist(),'scores':scores.tolist()}
   timings.append({'wall_seconds':time.perf_counter()-start,'process_cpu_seconds':time.process_time()-process})
   if repeat==0:saved=current
   else:assert current==saved,'Repeated results changed'
  rows.append({'query_count':count,'unique_original_queries':14,'source_row_indices':source_rows.tolist(),'is_repeated_batch':count!=14,'results':saved,'timings':timings})
 report={'status':'MEASUREMENT_ONLY','mode':a.label,'scope':'CPU-only fixed embeddings; repeated 512 is not real unique-query throughput','index_sha256':a.expected_index_sha256,'embeddings_sha256':original['embeddings_sha256'],'embedding_file_sha256':sha(a.embeddings),'query_sha256':original['queries_sha256'],'reference_cpu_sha256':sha(a.reference_cpu),'blas':blas_identity(faiss),'threads':a.threads,'faiss_blas_threshold':threshold,'index_load':load,'total_wall_seconds':time.perf_counter()-total_start,'total_process_cpu_seconds':time.process_time()-total_cpu,'total_cost_scope':'tool body including asset hash/index load/search; excludes task setup','faiss_path':faiss.__file__,'faiss_version':getattr(faiss,'__version__',None),'rows':rows,'peak_self_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'process_threads':len(os.listdir('/proc/self/task')),'environment':{k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','LD_PRELOAD']}}
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)

def compare(a):
 assert not a.output.exists()
 left=json.loads(a.default.read_text());right=json.loads(a.preload.read_text());report={'status':'DIFFERENCE_REPORT_ONLY','scope':'No automatic acceptance or production switch','comparisons':[]}
 for key in ('index_sha256','embeddings_sha256','query_sha256','reference_cpu_sha256'):assert left[key]==right[key],'Identity differs '+key
 assert [r['query_count'] for r in left['rows']]==[14,512] and [r['query_count'] for r in right['rows']]==[14,512],'Missing measurement batches'
 for x,y in zip(left['rows'],right['rows']):
  assert x['query_count']==y['query_count'] and x['source_row_indices']==y['source_row_indices']
  for k in ('3','50'):
   count=x['query_count'];assert all(len(r[k][field])==count for r in [x['results'],y['results']] for field in ['ids','scores']),'Missing query results'
   for i,(ids_a,ids_b,scores_a,scores_b) in enumerate(zip(x['results'][k]['ids'],y['results'][k]['ids'],x['results'][k]['scores'],y['results'][k]['scores'])):
    assert len(ids_a)==len(ids_b)==len(scores_a)==len(scores_b)==int(k) and len(set(ids_a))==len(set(ids_b))==int(k),'Invalid topk dimensions'
    assert all(math.isfinite(v) for v in scores_a+scores_b),'Nonfinite scores'
    lm=dict(zip(ids_a,scores_a));rm=dict(zip(ids_b,scores_b));shared=set(lm)&set(rm)
    report['comparisons'].append({'query_count':x['query_count'],'row':i,'original_query_row':x['source_row_indices'][i],'k':int(k),'exact_rank':ids_a==ids_b,'overlap_count':len(shared),'only_default':sorted(set(lm)-set(rm)),'only_preload':sorted(set(rm)-set(lm)),'max_matched_id_score_abs_delta':max((abs(lm[v]-rm[v]) for v in shared),default=None),'ids_default':ids_a,'ids_preload':ids_b})
 report['input_sha256']={str(a.default):sha(a.default),str(a.preload):sha(a.preload)}
 with a.output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)

def main():
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True);m=sub.add_parser('measure');m.add_argument('--allow-large-index',action='store_true');m.add_argument('--index',type=pathlib.Path,required=True);m.add_argument('--expected-index-sha256',required=True);m.add_argument('--embeddings',type=pathlib.Path,required=True);m.add_argument('--reference-cpu',type=pathlib.Path,required=True);m.add_argument('--output',type=pathlib.Path,required=True);m.add_argument('--label',choices=['default','preload'],required=True);m.add_argument('--threads',type=int,default=8);m.add_argument('--repeats',type=int,default=2);c=sub.add_parser('compare');c.add_argument('--default',type=pathlib.Path,required=True);c.add_argument('--preload',type=pathlib.Path,required=True);c.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
 if a.command=='measure':assert a.threads>0 and a.repeats>=2;measure(a)
 else:compare(a)
if __name__=='__main__':main()
