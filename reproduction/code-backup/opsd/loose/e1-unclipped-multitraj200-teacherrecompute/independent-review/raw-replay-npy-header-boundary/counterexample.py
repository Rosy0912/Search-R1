import hashlib,io,json,os,sys,types,zipfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
H=Path(__file__).resolve().parents[2]
P=H/'final200-raw-replay-consumer-candidate/cpu-originals/20261006T180647165485Z/replay_final200_raw.py.input-original'
raw=P.read_bytes();assert hashlib.sha256(raw).hexdigest()=='c5a4497347bfa2ac23ecb725d4ae0dc1d4b0489fe8fd2805416604c691a7374e'
m=types.ModuleType('exact_c5a_boundary');m.__file__=str(P);exec(compile(raw,str(P),'exec'),m.__dict__)
OUT=Path(__file__).resolve().parent
npz=OUT/'small-malformed-original.npz'
with zipfile.ZipFile(npz,'x',compression=zipfile.ZIP_DEFLATED) as archive:
 for name in sorted(m.SCORING_KEYS):
  stream=io.BytesIO();shape=(64,100000000) if name=='responses' else (1,1)
  np.lib.format.write_array_header_1_0(stream,dict(descr=np.dtype('int64').str,fortran_order=False,shape=shape))
  if name!='responses':stream.write(b'\0'*8)
  archive.writestr(name+'.npy',stream.getvalue())
seen=[];saved=np.ndarray
class HugeArrayAllocationIntercepted(RuntimeError):pass
def safe_array(shape,*args,**kwargs):
 count=int(np.prod(shape)) if not isinstance(shape,int) else shape
 if count>1000000:
  seen.append(dict(elements=count,dtype=str(kwargs.get('dtype')),would_allocate_bytes=count*np.dtype(kwargs['dtype']).itemsize))
  raise HugeArrayAllocationIntercepted('Blocked giant native array allocation before any memory was allocated')
 return saved(shape,*args,**kwargs)
reader=m.Originals(os.getuid());pin=hashlib.sha256(npz.read_bytes()).hexdigest()
with patch.object(np,'ndarray',safe_array):
 try:m.npz_arrays(reader,str(npz),pin)
 except HugeArrayAllocationIntercepted as error:message=str(error)
 else:raise AssertionError('Expected old consumer to reach giant allocation despite tinyZIP')
assert seen and seen[0]['elements']==6400000000
result=dict(kind='SAFE_LOCAL_NPY_HEADER_BOUNDED_MEMORY_COUNTEREXAMPLE',consumer_sha256=hashlib.sha256(raw).hexdigest(),physical_source=str(P),original_npz_sha256=pin,original_npz_bytes=npz.stat().st_size,allocation_intercepted=seen,no_actual_giant_allocation=True,production_main_executed=False,gpu_executed=False,remote_executed=False,error=message)
(OUT/'counterexample.original.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n');print(json.dumps(result))
