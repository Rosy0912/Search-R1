import ctypes,json,time,os,resource,pathlib
import faiss,numpy as np
faiss.omp_set_num_threads(8)
suffix='avx512' if 'AVX512' in faiss.get_compile_options() else 'avx2' if 'AVX2' in faiss.get_compile_options() else ''
pattern='_swigfaiss'+('_'+suffix if suffix else '')+'.*.so'
lib=ctypes.CDLL(str(next(pathlib.Path(faiss.__file__).parent.glob(pattern))))
class DlInfo(ctypes.Structure):
 _fields_=[('filename',ctypes.c_char_p),('base',ctypes.c_void_p),('symbol',ctypes.c_char_p),('address',ctypes.c_void_p)]
dl=ctypes.CDLL('libdl.so.2');dl.dladdr.argtypes=[ctypes.c_void_p,ctypes.POINTER(DlInfo)];dl.dladdr.restype=ctypes.c_int
bindings={}
for scope,handle in [('faiss_handle',lib),('global',ctypes.CDLL(None))]:
 for name in ['sgemm_','openblas_get_config','openblas_get_corename','openblas_get_num_threads']:
  f=getattr(handle,name,None)
  if f is not None:
   info=DlInfo();dl.dladdr(ctypes.cast(f,ctypes.c_void_p),ctypes.byref(info));bindings[scope+':'+name]=info.filename.decode() if info.filename else None
meta={}
for name in ['openblas_get_config','openblas_get_corename','openblas_get_num_threads']:
 f=getattr(lib,name,None)
 if f is not None:
  f.restype=ctypes.c_int if name.endswith('threads') else ctypes.c_char_p
  x=f();meta[name]=x.decode() if isinstance(x,bytes) else x

identity={}
for scope,handle in [('faiss_handle',lib),('global',ctypes.CDLL(None))]:
 row={}
 for name in ['openblas_get_config','openblas_get_corename','openblas_get_num_threads']:
  f=getattr(handle,name,None)
  if f:
   f.restype=ctypes.c_int if name.endswith('threads') else ctypes.c_char_p
   v=f();row[name]=v.decode() if isinstance(v,bytes) else v
 identity[scope]=row
print(json.dumps({'identity':identity,'bindings':bindings,'omp_threads':faiss.omp_get_max_threads(),'environment':{k:os.environ.get(k) for k in ['LD_PRELOAD','CUDA_VISIBLE_DEVICES','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']}}))
