import ast,pathlib,tempfile,types,io,json,hashlib,time,asyncio,copy
import numpy as np
root=pathlib.Path(__file__).resolve().parents[2];server=root/'work/index_gpu_server.py';proxy=root/'work/hybrid_retrieval_server.py';rows=[]
class HTTPException(Exception):
 def __init__(self,status_code,detail):self.status_code=status_code;super().__init__(detail)
class Request:
 def __init__(self,raw):self.raw=raw
 async def body(self):return self.raw
 async def stream(self):
  for i in range(0,len(self.raw),65536):yield self.raw[i:i+65536]
class ToyIndex:
 ntotal=21015324;d=768
 def search(self,q,k):
  return np.tile(np.arange(k,dtype=np.float32),(len(q),1))+q[:,0:1],np.tile(np.arange(k,dtype=np.int64),(len(q),1))
with tempfile.TemporaryDirectory() as tmp:
 run=pathlib.Path(tmp);instance={'gpu_uuid':'GPU-c63282c6-1d05-224e-234b-2a8c794b1f04','pid':1234,'run':'own-fixture','mode':'fp32-index-only','index_float16':False,'source_sha256':'a'*64,'index_sha256':'b'*64,'dimension':768,'ntotal':21015324};idx=ToyIndex()
 sg={'np':np,'io':io,'hashlib':hashlib,'json':json,'time':time,'healthy':lambda:None,'check_assets':lambda:None,'torch':types.SimpleNamespace(cuda=types.SimpleNamespace(synchronize=lambda:None)),'index':idx,'instance':instance,'run':run,'Request':Request,'HTTPException':HTTPException}
 node=next(n for n in ast.walk(ast.parse(server.read_text())) if isinstance(n,ast.AsyncFunctionDef) and n.name=='index_search');node.decorator_list=[];exec(compile(ast.Module(body=[node],type_ignores=[]),str(server),'exec'),sg)
 mutation=[None];seen=[]
 class Response:
  def __init__(self,data):self.data=data
  def raise_for_status(self):pass
  def json(self):return self.data
 def post(url,params,data,headers,timeout):
  seen.append({'url':url,'k':params['k'],'content_type':headers['Content-Type'],'timeout':timeout,'sha':hashlib.sha256(data).hexdigest()})
  if mutation[0]=='timeout':raise TimeoutError('own fixture socket timeout')
  payload=asyncio.run(sg['index_search'](Request(data),k=params['k']));payload=copy.deepcopy(payload)
  if callable(mutation[0]):mutation[0](payload)
  return Response(payload)
 pg={'np':np,'io':io,'hashlib':hashlib,'json':json,'time':time,'expected':instance,'a':types.SimpleNamespace(run=run,index_url='http://127.0.0.1:18083'),'requests':types.SimpleNamespace(post=post),'poison':[]}
 klass=next(n for n in ast.walk(ast.parse(proxy.read_text())) if isinstance(n,ast.ClassDef) and n.name=='IndexProxy');exec(compile(ast.Module(body=[klass],type_ignores=[]),str(proxy),'exec'),pg);obj=pg['IndexProxy']()
 rng=np.random.RandomState(17);q=rng.normal(size=(14,768)).astype(np.float32)
 def test(name,vector=q,k=3,mut=None,should_pass=False):
  mutation[0]=mut;pg['poison'].clear()
  try:
   scores,ids=obj.search(vector,k);passed=True;err=None
   if should_pass:
    want_s,want_i=idx.search(vector,k);assert np.array_equal(scores,want_s) and np.array_equal(ids,want_i)
  except Exception as e:passed=False;err=repr(e)
  rows.append({'case':name,'accepted':passed,'expected_accepted':should_pass,'matches_requirement':passed==should_pass,'error':err})
  if passed:assert not pg['poison']
  else:
   assert pg['poison'] and (run/'index-failure.json').exists()
   count=len(seen)
   try:obj.search(q,3);raise AssertionError('poison allowed next search')
   except RuntimeError:pass
   assert len(seen)==count, 'poison caused another RPC'
   import sys
   sys.modules['fastapi']=types.SimpleNamespace(HTTPException=HTTPException)
   pn=next(n for n in ast.walk(ast.parse(proxy.read_text())) if isinstance(n,ast.AsyncFunctionDef) and n.name=='provenance');pn.decorator_list=[]
   ng={'poison':pg['poison'],'metadata':instance};exec(compile(ast.Module(body=[pn],type_ignores=[]),str(proxy),'exec'),ng)
   try:asyncio.run(ng['provenance']());raise AssertionError('poison metadata returned healthy')
   except HTTPException as e:assert e.status_code==503

 for count in (1,14,512):
  v=q[np.arange(count)%14];test('roundtrip_'+str(count)+'_k3',v,should_pass=True)
 test('roundtrip_14_k50',k=50,should_pass=True);test('fortran_input',np.asfortranarray(q),should_pass=True)
 for name,v in [('float64',q.astype(np.float64)),('float16',q.astype(np.float16)),('wrong_dimension',q[:,:767]),('empty',q[:0]),('oversized',np.zeros((513,768),np.float32)),('nonfinite',q.copy())]:
  if name=='nonfinite':v[0,0]=np.nan
  test(name,v)
 test('transport_timeout',mut='timeout')
 for name,mut in [('instance_changed',lambda p:p['instance'].update(pid=999)),('vector_sha_changed',lambda p:p.update(vectors_sha256='x')),('frame_sha_changed',lambda p:p.update(request_sha256='x')),('k_changed',lambda p:p.update(k=50)),('duplicate_ids',lambda p:p['ids'][0].__setitem__(1,p['ids'][0][0])),('id_range',lambda p:p['ids'][0].__setitem__(0,21015324)),('id_float',lambda p:p['ids'][0].__setitem__(0,0.5)),('score_nan',lambda p:p['scores'][0].__setitem__(0,float('nan'))),('score_string',lambda p:p['scores'][0].__setitem__(0,'1.25')),('score_boolean',lambda p:p['scores'][0].__setitem__(0,True)),('score_non_fp32_precision',lambda p:p['scores'][0].__setitem__(0,1/3))]:test(name,mut=mut)
 # Exact server parsing negative cases, without real HTTP/server/GPU.
 for name,value,k in [('server_float64',q.astype(np.float64),3),('server_float16',q.astype(np.float16),3),('server_bad_k',q,4),('server_object',np.array([['x']]*768,dtype=object),3)]:
  f=io.BytesIO();np.save(f,value,allow_pickle=True)
  try:asyncio.run(sg['index_search'](Request(f.getvalue()),k));passed=True;err=None
  except Exception as e:passed=False;err=repr(e)
  rows.append({'case':name,'accepted':passed,'expected_accepted':False,'matches_requirement':not passed,'error':err})
 # Header-bound negatives must reject before np.load is called.
 header=io.BytesIO();np.lib.format.write_array_header_1_0(header,{'descr':'<f4','fortran_order':False,'shape':(10**9,768)})
 f=io.BytesIO();np.save(f,q,allow_pickle=False);valid=f.getvalue()
 for name,raw,code in [('huge_header_no_allocation',header.getvalue(),400),('truncated_frame',valid[:-1],400),('trailing_bytes',valid+b'x',400),('stream_over_2MiB',b'x'*(2*1024*1024+1),413)]:
  original_load=np.load;called=[]
  def forbidden_load(*args,**kwargs):called.append(True);raise AssertionError('np.load before budget/header check')
  np.load=forbidden_load
  try:
   try:asyncio.run(sg['index_search'](Request(raw),3));raise AssertionError('bad frame accepted')
   except HTTPException as error:assert error.status_code==code
   assert not called
  finally:np.load=original_load
  rows.append({'case':name,'accepted':False,'expected_accepted':False,'matches_requirement':True,'rejected_before_np_load':True,'http_status':code})
 # Reviewed NPY v2 headers and Fortran order are supported without numerical change.
 h=io.BytesIO();np.lib.format.write_array_header_2_0(h,{'descr':'<f4','fortran_order':False,'shape':q.shape});raw=h.getvalue()+q.tobytes()
 output=asyncio.run(sg['index_search'](Request(raw),3));assert output['vectors_sha256']==hashlib.sha256(q.tobytes()).hexdigest()
 rows.append({'case':'server_npy_v2_positive','accepted':True,'expected_accepted':True,'matches_requirement':True})
 result={'scope':'AST-isolated actual source functions/classes; toy numeric index, mocked torch/NVML/network; no service/whole index/GPU execution','sources_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (server,proxy)},'rows':rows,'mismatches':[r['case'] for r in rows if not r['matches_requirement']],'request_count':len(seen),'request_frames':seen}
 (root/'work/aux-gate-review/index-proxy-fixtures.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'cases':len(rows),'mismatches':result['mismatches']},indent=2))
