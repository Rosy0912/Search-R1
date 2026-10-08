"""Actual CUDA entropy function proof only; no model, actor, teacher or capacity claim."""
import enum,ast,contextlib,hashlib,inspect,json,os,pathlib,sys,time,types
import torch
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd')
SOURCE=ROOT/'tools/search-r1-32b-grpo100-source-v5'
MANIFEST='72c5fd163b9cbe6bfb502d098b7bc14e7dfa386e5c008ffd7a202eebd14b57b4'
ACTOR_SHA='a0569434298344cedbad7fe184995e732c20c4c32aa62f2fd284f1be36f64a44'
ENTROPY_SHA='e6f0578b0f52b530d52e9fc69968108d67badeb8374f18f39258e3550ac35454'
OPTIONS={'triton.autotune_pointwise':False,'max_autotune':False,'max_autotune_pointwise':False,'coordinate_descent_tuning':False,'dynamic_scale_rblock':False}
# Fixed before execution. BF16 eager has different reduction rounding from compiled;
# compiled-original comparison remains tight and eager differences are recorded separately.
TOLERANCES={'float32':{'value_atol':1e-4,'value_rtol':1e-5,'gradient_atol':1e-7,'gradient_rtol':1e-4},'bfloat16':{'value_atol':0.125,'value_rtol':0.016,'gradient_atol':0.0001,'gradient_rtol':0.032}}
def need(x,msg):
 if not x:raise RuntimeError(msg)
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def metadata_projection(value):
 # Preserve JSON primitives exactly, especially measured boolean flags.
 if isinstance(value,enum.Enum):return {'__actual_type__':type(value).__module__+'.'+type(value).__qualname__,'__actual_repr__':repr(value),'name':value.name}
 if value is None or isinstance(value,(bool,int,float,str)):return value
 if isinstance(value,dict):
  need(all(isinstance(k,str) for k in value),'Non-string actual metadata keys');return {k:metadata_projection(v) for k,v in value.items()}
 if isinstance(value,(list,tuple)):return [metadata_projection(v) for v in value]
 if isinstance(value,(set,frozenset)):return {'__actual_type__':type(value).__name__,'__actual_repr__':repr(value),'items':[metadata_projection(v) for v in sorted(value,key=repr)]}
 return {'__actual_type__':type(value).__module__+'.'+type(value).__qualname__,'__actual_repr__':repr(value)}
def publish(p,v):
 t=p.with_name(p.name+'.private')
 with t.open('x') as f:json.dump(v,f,indent=2);f.flush();os.fsync(f.fileno())
 os.link(t,p);t.unlink();fd=os.open(p.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
def source_functions():
 p=SOURCE/'verl/utils/torch_functional.py';need(sha(p)==ENTROPY_SHA,'Original entropy file drift')
 tree=ast.parse(p.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='entropy_from_logits');need(not node.decorator_list,'Unexpected original entropy decorator')
 namespace={'torch':torch};exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),'exec'),namespace);eager=namespace['entropy_from_logits']
 actor=SOURCE/'verl/workers/actor/dp_actor.py';need(sha(actor)==ACTOR_SHA,'Reviewed actor compile source drift')
 tree=ast.parse(actor.read_text());node=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='compute_entropy_from_logits' for t in n.targets));call=node.value
 need(ast.literal_eval(next(k.value for k in call.keywords if k.arg=='options'))==OPTIONS,'Exact five local options required')
 candidate=eval(compile(ast.Expression(body=call),str(actor),'eval'),{'torch':torch,'verl_F':types.SimpleNamespace(entropy_from_logits=eager)})
 # Original V4 actor expression, unchanged original entropy function.
 old=ROOT/'tools/search-r1-32b-grpo100-source-v4/verl/workers/actor/dp_actor.py';need(sha(old)=='26ab1b6ade6f1d385c3381941a4346ba8c136c3b9b65405d4c7d32242cbafcad','Original V4 actor full source drift');oldtree=ast.parse(old.read_text());oldnode=next(n for n in ast.walk(oldtree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='compute_entropy_from_logits' for t in n.targets))
 need(ast.unparse(oldnode.value)=='torch.compile(verl_F.entropy_from_logits, dynamic=True)','Original compiled reference expression drift')
 original=eval(compile(ast.Expression(body=oldnode.value),str(old),'eval'),{'torch':torch,'verl_F':types.SimpleNamespace(entropy_from_logits=eager)})
 return eager,original,candidate,dict(entropy_file_sha256=ENTROPY_SHA,entropy_body_ast_sha256=hashlib.sha256(ast.dump(next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='entropy_from_logits')).encode()).hexdigest(),actor_source_sha256=ACTOR_SHA,original_compile_expression=ast.unparse(oldnode.value),candidate_compile_expression=ast.unparse(call))
@contextlib.contextmanager
def telemetry(scope,events,ordinal=None):
 cache_directory=os.environ.get('TORCHINDUCTOR_CACHE_DIR')
 import triton.testing as testing
 import torch._inductor.runtime.runtime_utils as utils
 import torch._inductor.runtime.triton_heuristics as heuristics
 originals=[]
 compiler_cls=torch._TorchCompileInductorWrapper;old_backend=compiler_cls.__call__;originals.append((compiler_cls,'__call__',old_backend))
 def dispatched(self,*a,**kw):
  events.append(dict(kind='ACTUAL_BACKEND_DISPATCH',scope=scope,case_ordinal=ordinal,cache_directory=cache_directory,epoch=time.time(),config_patches=metadata_projection(dict(self.config)),dynamic=self.dynamic));return old_backend(self,*a,**kw)
 compiler_cls.__call__=dispatched
 def wrap_bench(module,name):
  if not hasattr(module,name):return
  old=getattr(module,name);originals.append((module,name,old))
  def wrapped(*a,**kw):
   events.append(dict(kind='DO_BENCH_CALL',scope=scope,entry=module.__name__+'.'+name,epoch=time.time()))
   need(scope!='candidate','Candidate invoked forbidden do_bench')
   return old(*a,**kw)
  setattr(module,name,wrapped)
 for m,n in [(testing,'do_bench'),(utils,'triton_do_bench'),(utils,'do_bench_gpu'),(heuristics,'do_bench_gpu')]:wrap_bench(m,n)
 cls=heuristics.CachingAutotuner
 for method in ('bench','autotune_to_one_config','coordinate_descent_tuning'):
  if callable(getattr(cls,method,None)):
   old_method=getattr(cls,method);originals.append((cls,method,old_method))
   def intercepted(self,*a,_method=method,_original=old_method,**kw):
    events.append(dict(kind='TUNING_ROUTE_CALL',scope=scope,entry=_method,epoch=time.time()));need(scope!='candidate','Candidate called forbidden tuning route '+_method);return _original(self,*a,**kw)
   setattr(cls,method,intercepted)
 old=cls.precompile;originals.append((cls,'precompile',old))
 def precompile(self,*a,**kw):
  rec=dict(kind='ACTUAL_PRECOMPILE',scope=scope,case_ordinal=ordinal,cache_directory=cache_directory,epoch=time.time(),inductor_meta=metadata_projection(dict(getattr(self,'inductor_meta',{}))),inductor_meta_original_repr=repr(getattr(self,'inductor_meta',{})),metadata_encoding='TYPE_TAGGED_JSON_PROJECTION_WITH_ORIGINAL_REPR',triton_meta=repr(getattr(self,'triton_meta',{})),configs_before=repr(getattr(self,'configs',None)),actual_kernel_source=getattr(getattr(self,'fn',None),'src',None))
  events.append(rec)
  result=old(self,*a,**kw)
  rec.update(launchers_after=len(getattr(self,'launchers',[])),configs_after=repr(getattr(self,'configs',None)))
  return result
 cls.precompile=precompile
 try:yield
 finally:
  for m,n,v in reversed(originals):setattr(m,n,v)
def candidate_frontcheck(events):
 pre=[e for e in events if e['kind']=='ACTUAL_PRECOMPILE' and e['scope']=='candidate']
 dispatch=[e for e in events if e['kind']=='ACTUAL_BACKEND_DISPATCH' and e['scope']=='candidate']
 need(dispatch and all(e['config_patches']==OPTIONS for e in dispatch),'Cold candidate actual backend missing or wrong options')
 need(pre and all(e['launchers_after']==1 for e in pre),'Cold candidate precompile absent or multiple launchers')
 need(all(all(e['inductor_meta'].get(k) is False for k in ('autotune_pointwise','max_autotune','max_autotune_pointwise','dynamic_scale_rblock')) and 'coordinate_descent_tuning' not in e['inductor_meta'] for e in pre),'Cold actual candidate metadata options not proved')
 need(not any(e['kind'] in ('DO_BENCH_CALL','TUNING_ROUTE_CALL') and e['scope']=='candidate' for e in events),'Cold candidate forbidden benchmark')
 return dict(actual_backend_dispatches=len(dispatch),actual_precompile_records=len(pre),candidate_bench_calls=0)
def cold_compiled(scope,folder,ordinal,events):
 # Only this dedicated proof worker owns these caches; no live training process.
 torch._dynamo.reset()
 cache=folder/(scope+'-cold-cache-'+str(ordinal));cache.mkdir(exist_ok=False)
 os.environ['TORCHINDUCTOR_CACHE_DIR']=str(cache)
 eager,original,candidate,identity=source_functions()
 fn=candidate if scope=='candidate' else original
 keys=list(OPTIONS);actual=fn.get_compiler_config()
 flags={key:actual[key] for key in keys}
 if scope=='candidate':need(flags==OPTIONS,'Actual cold candidate compiler configuration differs')
 events.append(dict(kind='COLD_COMPILE_CONFIGURATION',scope=scope,case_ordinal=ordinal,epoch=time.time(),cache_directory=str(cache),actual_options=flags,owned_dynamo_reset=True))
 return fn,identity
def evaluated(fn,seed,shape,dtype):
 # All three paths use identical seeded synthetic logits, no question/answer inputs.
 torch.manual_seed(seed);x=torch.randn(shape,device='cuda',dtype=dtype,requires_grad=True)
 torch.cuda.reset_peak_memory_stats();start=time.monotonic()
 with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=dtype==torch.bfloat16):
  y=fn(x);weights=torch.linspace(.5,1.5,shape[0],device='cuda',dtype=y.dtype);(y*weights).sum().backward()
 torch.cuda.synchronize()
 need(torch.isfinite(y).all().item() and torch.isfinite(x.grad).all().item(),'Nonfinite entropy/gradient')
 out=y.detach().cpu();grad=x.grad.detach().cpu();record=dict(seconds=time.monotonic()-start,output_dtype=str(y.dtype),gradient_dtype=str(x.grad.dtype),autocast_enabled=dtype==torch.bfloat16,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
 del weights,y,x;torch.cuda.empty_cache();return out,grad,record
def difference(a,b,atol,rtol):
 need(a.shape==b.shape and a.dtype==b.dtype,'Value/gradient identity mismatch')
 maximum=relative=0.;passed=True
 # Full tensors are compared, never sampled; CPU chunks limit temporary allocation.
 for x,y in zip(a.reshape(-1).split(1024*1024),b.reshape(-1).split(1024*1024)):
  x=x.float();y=y.float();d=(x-y).abs();maximum=max(maximum,d.max().item());relative=max(relative,(d/y.abs().clamp_min(1e-12)).max().item());passed=passed and bool(torch.allclose(x,y,atol=atol,rtol=rtol))
 return dict(max_absolute_delta=maximum,max_relative_delta=relative,atol=atol,rtol=rtol,all_elements_passed=passed,elements=a.numel())
def adjacent_difference(a,b):
 need(a.shape==b.shape and a.dtype==b.dtype==torch.bfloat16,'Actual BF16 adjacent comparison required')
 maximum=relative=max_ulp=0.;nonadjacent=0
 for ax,bx in zip(a.reshape(-1).split(1024*1024),b.reshape(-1).split(1024*1024)):
  up=torch.nextafter(bx,torch.full_like(bx,float('inf')));down=torch.nextafter(bx,torch.full_like(bx,float('-inf')))
  step=torch.maximum((up.float()-bx.float()).abs(),(bx.float()-down.float()).abs());d=(ax.float()-bx.float()).abs()
  maximum=max(maximum,d.max().item());relative=max(relative,(d/bx.float().abs().clamp_min(1e-30)).max().item());max_ulp=max(max_ulp,(d/step).max().item());nonadjacent+=int(((ax<down)|(ax>up)).sum().item())
 return dict(max_absolute_delta=maximum,max_relative_delta=relative,max_reference_adjacent_steps=max_ulp,nonadjacent_elements=nonadjacent,allowed_reference_adjacent_steps=1,all_elements_passed=nonadjacent==0,elements=a.numel(),criterion='ACTUAL_REFERENCE_BF16_NEXTAFTER_CLOSED_NEIGHBOR_INTERVAL_NO_EXTRA_ABSOLUTE_FLOOR')
def compiled_difference(a,b,gradient):
 if a.dtype==torch.bfloat16:return adjacent_difference(a,b)
 need(a.dtype==torch.float32,'Unexpected actual entropy tensor dtype')
 return difference(a,b,1e-7 if gradient else 1e-4,1e-4 if gradient else 1e-5)
def main():
 folder=pathlib.Path(os.environ['R1_DEVICE_PROOF_FOLDER']);output=folder/'worker-result.json';result=dict(status='FAIL_ENTROPY_DEVICE_CORRECTNESS_NOT_CAPACITY_NOT_TRAINING',job=os.environ['SLURM_JOB_ID'],pid=os.getpid(),source_manifest_sha256=MANIFEST,options=OPTIONS,tolerances=TOLERANCES,compiled_comparison_policy={'actual_float32_value_atol':1e-4,'actual_float32_value_rtol':1e-5,'actual_float32_gradient_atol':1e-7,'actual_float32_gradient_rtol':1e-4,'actual_bfloat16_max_reference_nextafter_step':1,'eager_comparison':'REPORT_ONLY_NOT_CORE_ACCEPTANCE'},cases=[],events=[])
 try:
  need(torch.__version__.split('+')[0]=='2.4.0','Exact native Torch2.4 required');need(os.environ.get('CUDA_VISIBLE_DEVICES','').startswith('GPU-'),'Own single selected leased UUID required')
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  eager,original,candidate,identity=source_functions();result['source_identity']=identity
  cases=[(6144,152064,'float32'),(6143,152064,'float32'),(6144,152064,'bfloat16'),(6143,152064,'bfloat16')]
  for ordinal,(length,vocab,dname) in enumerate(cases):
   dtype=getattr(torch,dname);tol=TOLERANCES[dname];seed=42+ordinal
   with telemetry('eager',result['events']):ey,eg,em=evaluated(eager,seed,(length,vocab),dtype)
   original,identity=cold_compiled('original_compiled',folder,ordinal,result['events'])
   # Dedicated proof compiler executes synchronously for direct backend/precompile evidence.
   with torch._inductor.config.patch({'compile_threads':1}),telemetry('original_compiled',result['events'],ordinal):oy,og,om=evaluated(original,seed,(length,vocab),dtype)
   candidate_start_index=len(result['events'])
   candidate,identity=cold_compiled('candidate',folder,ordinal,result['events'])
   with torch._inductor.config.patch({'compile_threads':1}),telemetry('candidate',result['events'],ordinal):cy,cg,cm=evaluated(candidate,seed,(length,vocab),dtype)
   frontcheck=candidate_frontcheck(result['events'][candidate_start_index:]);publish(folder/('candidate-frontcheck-%d.json'%ordinal),frontcheck)
   rec=dict(length=length,vocab=vocab,input_dtype=dname,seed=seed,eager=em,original_compiled=om,candidate=cm,candidate_vs_original_compiled_value=compiled_difference(cy,oy,False),candidate_vs_original_compiled_gradient=compiled_difference(cg,og,True),candidate_vs_eager_value=difference(cy,ey,tol['value_atol'],tol['value_rtol']),candidate_vs_eager_gradient=difference(cg,eg,tol['gradient_atol'],tol['gradient_rtol']))
   result['cases'].append(rec);publish(folder/('case-%d.json'%ordinal),rec)
   need(all(rec[k]['all_elements_passed'] for k in ('candidate_vs_original_compiled_value','candidate_vs_original_compiled_gradient')),'Predeclared compiled value/gradient tolerance failed')
   del ey,eg,oy,og,cy,cg;torch.cuda.empty_cache()
  pre=[x for x in result['events'] if x['kind']=='ACTUAL_PRECOMPILE' and x['scope']=='candidate'];need(pre,'No actual candidate generated/precompile metadata');need(all(x['launchers_after']==1 for x in pre),'Candidate produced multiple tuning launchers')
  need(all(all(x['inductor_meta'].get(key) is False for key in ('autotune_pointwise','max_autotune','max_autotune_pointwise','dynamic_scale_rblock')) for x in pre),'Candidate actual four explicit metadata flags not False')
  codegen=pathlib.Path(torch.__file__).parent/'_inductor/codegen/triton.py';need(sha(codegen)=='2f0105a59f6e341c2e8a695e02e192e2603a6ef31e6e914c8b6dd33b9a46235a','Actual codegen metadata source drift')
  need(all('coordinate_descent_tuning' not in x['inductor_meta'] for x in pre),'Unexpected enabled coordinate-descent metadata')
  result['source_identity']['codegen_source_sha256']=sha(codegen)
  result['coordinate_metadata_provenance']=dict(status='ABSENT_WHEN_DISABLED_BY_PINNED_CODEGEN',source_ref={'path':str(codegen),'sha256':sha(codegen)},actual_option=False,raw_field_absent=True)
  need(not any(x['kind'] in ('DO_BENCH_CALL','TUNING_ROUTE_CALL') and x['scope']=='candidate' for x in result['events']),'Candidate benchmark call detected')
  dispatch=[x for x in result['events'] if x['kind']=='ACTUAL_BACKEND_DISPATCH' and x['scope']=='candidate'];need(dispatch and all(x['config_patches']==OPTIONS for x in dispatch),'Actual candidate backend dispatch absent or wrong options')
  result['proof_compiler_execution']=dict(compile_threads=1,scope='DEDICATED_PROOF_OBSERVABILITY_NOT_PRODUCTION_CONCURRENCY_OR_CAPACITY')
  generated=[];saved=folder/'generated-candidate-kernels';saved.mkdir(exist_ok=False)
  for p in sorted(p for cache in folder.glob('candidate-cold-cache-*') for p in cache.rglob('*.py')):
   b=p.read_bytes();digest=hashlib.sha256(b).hexdigest();dest=saved/(digest+'.py');
   if not dest.exists():
    with dest.open('xb') as out:out.write(b);out.flush();os.fsync(out.fileno())
   else:need(dest.read_bytes()==b,'Generated source content collision')
   generated.append(dict(path=str(dest),original_cache_path=str(p),size=len(b),sha256=digest))
  need(generated,'No actual generated kernel source metadata');result['generated_files']=generated
  result['torch_version']=torch.__version__;result['actual_device']=dict(uuid=os.environ['CUDA_VISIBLE_DEVICES'],name=torch.cuda.get_device_name(0));result['status']='ENTROPY_DEVICE_CORRECTNESS_AND_BENCH_SUPPRESSION_PASS_NOT_CAPACITY_NOT_TRAINING'
 except BaseException as e:result['error_type']=type(e).__name__;result['error']=str(e);raise
 finally:
  import gc;gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache();result['allocator_after_release']=dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved());
  if result['allocator_after_release']!=dict(allocated=0,reserved=0):result['status']='FAIL_ENTROPY_DEVICE_CORRECTNESS_NOT_CAPACITY_NOT_TRAINING';result['release_error']='CUDA allocator nonzero'
  publish(output,result)
 need(result['allocator_after_release']==dict(allocated=0,reserved=0),'Actual CUDA allocator not released')
if __name__=='__main__':main()
