"""Independent actual-source CPU probes; no service/GPU/Slurm operations."""
import ast,asyncio,concurrent.futures,json,pathlib,sys,types,time
from unittest import mock
import pytest
ROOT=pathlib.Path(__file__).parent

def endpoint_namespace():
 tree=ast.parse((ROOT/'h100_gpu_retrieval_service.py').read_text());node=next(n for n in ast.walk(tree) if isinstance(n,ast.AsyncFunctionDef) and n.name=='retrieve');node.decorator_list=[]
 from fastapi import HTTPException
 calls=[]
 async def original(request):calls.append((len(request.queries),request.return_scores));return {'result':[[] for _ in request.queries]}
 executor=concurrent.futures.ThreadPoolExecutor(max_workers=1)
 ns={'rs':types.SimpleNamespace(QueryRequest=object,retrieve_endpoint=original),'healthy':lambda:None,'HTTPException':HTTPException,'asyncio':asyncio,'executor':executor,'poison':[],'run':None,'time':time,'json':json}
 exec(compile(ast.Module(body=[node],type_ignores=[]),'actual-h100-service-endpoint','exec'),ns)
 return ns,calls,executor

@pytest.mark.parametrize('rows,scores',[(0,True),(513,True),(2,False)])
def test_original_api_empty_large_and_scoreless_requests_preserved(rows,scores):
 ns,calls,pool=endpoint_namespace()
 try:
  class Query(types.SimpleNamespace):
   def copy(self,update):return Query(**dict(vars(self),**update))
  request=Query(queries=['q']*rows,topk=3,return_scores=scores)
  result=asyncio.run(ns['retrieve'](request));assert result=={'result':[[] for _ in range(rows)]} and calls==[(rows,True)]
 finally:pool.shutdown()

@pytest.mark.parametrize('kind,expected_row', [('small-foreign',True),('duplicate-records',True),('high-memory',False),('high-sm',False),('unknown-memory',False),('unknown-active-pid',False),('wrong-cgroup',False),('stop-log-failure',False),('unsupported-nvml',False)])
def test_actual_guard_background_and_fail_closed(tmp_path,monkeypatch,kind,expected_row):
 src=ROOT/'h100_retrieval_resource_guard.py';code=src.read_text();proc=tmp_path/'proc';(proc/'self').mkdir(parents=True);(proc/'self/cgroup').write_text('0::/slurm/job_123' if kind=='wrong-cgroup' else '0::/slurm/job_12/step_0')
 (proc/'42').mkdir();(proc/'42/cgroup').write_text('0::/foreign')
 output=tmp_path/'samples.jsonl';cancel=[];NoSamples=type('NoSamples',(Exception,),{})
 mem=types.SimpleNamespace(total=1000,used=100);records=[types.SimpleNamespace(pid=42,usedGpuMemory=160 if kind in ('high-memory','stop-log-failure') else None if kind=='unknown-memory' else 100)]
 if kind=='unknown-active-pid':(proc/'42/cgroup').unlink()
 gpu=types.SimpleNamespace(NVMLError_NotFound=NoSamples,nvmlInit=lambda:None,nvmlDeviceGetCount=lambda:1,nvmlDeviceGetHandleByIndex=lambda i:'h',nvmlDeviceGetUUID=lambda h:'GPU-own',nvmlDeviceGetMemoryInfo=lambda h:mem,nvmlDeviceGetUtilizationRates=lambda h:types.SimpleNamespace(gpu=0),nvmlDeviceGetComputeRunningProcesses=lambda h:records,nvmlDeviceGetGraphicsRunningProcesses=lambda h:records if kind=='duplicate-records' else [])
 def utilization(h,timestamp):
  if kind=='unsupported-nvml':raise RuntimeError('NVML NotSupported')
  if kind=='high-sm':return [types.SimpleNamespace(pid=42,timeStamp=time.time()*1e6,smUtil=16)]
  raise NoSamples()
 gpu.nvmlDeviceGetProcessUtilization=utilization
 original_dumps=json.dumps
 def dumps(value,*a,**kw):
  if kind=='stop-log-failure' and isinstance(value,dict) and value.get('event')=='STOP_OWN_JOB':raise OSError('guard log disk failure')
  return original_dumps(value,*a,**kw)
 code=code.replace("'/proc/self/cgroup'",repr(str(proc/'self/cgroup'))).replace("'/proc'",repr(str(proc)))
 with mock.patch('json.dumps',side_effect=dumps),mock.patch.dict(sys.modules,{'pynvml':gpu}),mock.patch.dict('os.environ',{'SLURM_JOB_ID':'12','SLURM_JOB_NODELIST':'h100-11','SEARCH_R1_GPU_LEASES':'[{"uuid":"GPU-own"}]'}),mock.patch.object(sys,'argv',[str(src),'--output',str(output)]),mock.patch('subprocess.run',side_effect=lambda cmd,**kw:cancel.append(cmd)),mock.patch('time.sleep',side_effect=RuntimeError('end independent fixture')):
  with pytest.raises(BaseException):exec(compile(code,str(src),'exec'),{})
 rows=[json.loads(x) for x in output.read_text().splitlines()] if output.exists() else []
 assert any('gpus' in r for r in rows)==expected_row
 if expected_row:assert rows[0]['gpus'][0]['foreign_memory_fraction']==.1
 if kind!='wrong-cgroup':assert cancel==[['scancel','--signal=TERM','12']]
 else:assert cancel==[]

def test_actual_endpoint_single_worker_serial_health_and_original_document_shape(tmp_path):
 ns,calls,pool=endpoint_namespace();threads=[];active=[0];peak=[0]
 import threading
 async def original(request):
  threads.append(threading.get_ident());active[0]+=1;peak[0]=max(peak[0],active[0]);time.sleep(.07);active[0]-=1
  return {'result':[[{'document':{'id':i,'contents':'fixed'},'score':.5}] for i in range(len(request.queries))]}
 ns['rs'].retrieve_endpoint=original
 class Query(types.SimpleNamespace):
  def copy(self,update):return Query(**dict(vars(self),**update))
 async def run():
  one=asyncio.create_task(ns['retrieve'](Query(queries=['q'],topk=3,return_scores=True)));two=asyncio.create_task(ns['retrieve'](Query(queries=['q'],topk=50,return_scores=False)));health=[]
  while not one.done() or not two.done():
   if active[0]:
    t=time.perf_counter();ns['healthy']();health.append(time.perf_counter()-t)
   await asyncio.sleep(.002)
  a,b=await asyncio.gather(one,two);assert a=={'result':[[{'document':{'id':0,'contents':'fixed'},'score':.5}]]} and b=={'result':[[{'id':0,'contents':'fixed'}]]}
  assert health and max(health)<.01
 try:asyncio.run(run());assert peak[0]==1 and len(set(threads))==1
 finally:pool.shutdown()

def test_actual_endpoint_backend_exception_poison_and_persist(tmp_path):
 ns,calls,pool=endpoint_namespace();ns['run']=tmp_path;error=RuntimeError('original backend failure')
 async def fail(request):raise error
 ns['rs'].retrieve_endpoint=fail
 class Query(types.SimpleNamespace):
  def copy(self,update):return Query(**dict(vars(self),**update))
 try:
  with pytest.raises(RuntimeError) as out:asyncio.run(ns['retrieve'](Query(queries=['q'],topk=3,return_scores=True)))
  assert out.value is error and ns['poison'] and json.loads((tmp_path/'service-failure.json').read_text())['error']==ns['poison'][0]
 finally:pool.shutdown()
