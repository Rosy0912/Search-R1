import ctypes,json,time,os,resource,pathlib
import faiss,numpy as np
faiss.omp_set_num_threads(8)
suffix='avx512' if 'AVX512' in faiss.get_compile_options() else 'avx2' if 'AVX2' in faiss.get_compile_options() else ''
pattern='_swigfaiss'+('_'+suffix if suffix else '')+'.*.so'
lib=ctypes.CDLL(str(next(pathlib.Path(faiss.__file__).parent.glob(pattern))))
meta={}
for name in ['openblas_get_config','openblas_get_corename','openblas_get_num_threads']:
 f=getattr(lib,name,None)
 if f is not None:
  f.restype=ctypes.c_int if name.endswith('threads') else ctypes.c_char_p
  x=f();meta[name]=x.decode() if isinstance(x,bytes) else x
rng=np.random.default_rng(42);db=rng.standard_normal((50000,768),dtype=np.float32);query=rng.standard_normal((512,768),dtype=np.float32);index=faiss.IndexFlatIP(768);index.add(db);rows=[]
faiss.cvar.distance_compute_blas_threshold=1000000000
for n in [14,128,512]:
 start=time.perf_counter();cpu=time.process_time();scores,ids=index.search(query[:n],3);rows.append({'queries':n,'wall_seconds':time.perf_counter()-start,'process_cpu_seconds':time.process_time()-cpu,'finite':bool(np.isfinite(scores).all())})
print(json.dumps({'scope':'synthetic small CPU index timing and BLAS identity only; not full production throughput or retrieval acceptance','distance_compute_blas_threshold':faiss.cvar.distance_compute_blas_threshold,'blas':meta,'faiss_compile_options':faiss.get_compile_options(),'omp_threads':faiss.omp_get_max_threads(),'database_vectors':50000,'dimension':768,'rows':rows,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}))
