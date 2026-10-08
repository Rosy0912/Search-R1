"""Independent held-FD CPU counterexamples; only generated tiny byte assets."""
import ast,hashlib,importlib.util,io,json,os,sys,tempfile
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from unittest.mock import patch
import torch
ROOT=Path.cwd();REPO=ROOT/'work/tree-probability-probe-runner-v2';HERE=ROOT/'work/tree-probability-probe-runner-review/v2/io-child';sys.path.insert(0,str(REPO))
from search_r1.baselines import probability_probe_io as I,probability_probe_loader as L
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(REPO/'search_r1/baselines/probability_probe_io.py')=='32f1bea8506bc6eebad6987f2602c97aa3c1872a7ff5c95d452b057d13c26c16'
assert sha(REPO/'docs/TREE_PROBABILITY_PROBE_RUNNER_V2_CPU_PROOF.json')=='59b6e8ff2ead58ec040730e947e4a1b700527ff35bcf284ebe554da0f8b46f10'
cases=[];observations=[]
def ok(n,b,d=None):
 assert b,(n,d);cases.append(dict(name=n,passed=True,detail=d))
def refused(n,call,classes=(ValueError,),phrase=None):
 try:call()
 except classes as e:ok(n,phrase is None or phrase in str(e),type(e).__name__+': '+str(e))
 else:raise AssertionError(n+' was unexpectedly accepted')
def payload(value):
 stream=io.BytesIO();torch.save(value,stream);return stream.getvalue()
def tinyfiles(tmp):
 a=Path(tmp)/'a.pt';b=Path(tmp)/'b.pt';a.write_bytes(payload({'value':torch.tensor([-1.])}));b.write_bytes(payload({'value':torch.tensor([-99.])}));return a,b,sha(a)
original_load=torch.load
with tempfile.TemporaryDirectory(prefix='held-fd-child-') as tmp:
 a,b,h=tinyfiles(tmp);live=[]
 def held_reader(stream,**kwargs):
  live.append(stream);ok('torch_parser_actual_input_is_held_stream',not isinstance(stream,(str,Path)) and stream.tell()==0 and kwargs==dict(map_location='cpu',weights_only=False));return original_load(stream,**kwargs)
 with patch.object(I.torch,'load',held_reader):loaded=I.load_tensor_verified(a,h)
 ok('actual_tiny_load_returns_expected_bytes',loaded['value'].tolist()==[-1.])
 ok('held_tensor_FD_closed_before_return',len(live)==1 and live[0].closed)
 for missinghash in [None,'', 'unknown', '0'*64]:
  with patch.object(I.torch,'load') as parser:
   refused('unknown_expected_SHA_'+repr(missinghash),lambda v=missinghash:I.load_tensor_verified(a,v),phrase='descriptor SHA')
   ok('unknown_expected_SHA_never_parsed_'+repr(missinghash),not parser.called)
 with patch.object(I.torch,'load') as parser:
  refused('missing_path_never_loads',lambda:I.load_tensor_verified(Path(tmp)/'missing.pt',h),classes=(FileNotFoundError,))
  ok('missing_path_parser_not_called',not parser.called)

# Path swap + restoration: real parser on stream must consume A, and no return is accepted.
with tempfile.TemporaryDirectory(prefix='held-fd-swap-') as tmp:
 a,b,h=tinyfiles(tmp);backup=Path(tmp)/'backup';consumed=[];streams=[]
 def swap_reader(stream,**kwargs):
  streams.append(stream);a.rename(backup);b.rename(a)
  try:value=original_load(stream,**kwargs);consumed.extend(value['value'].tolist());return value
  finally:a.rename(b);backup.rename(a)
 with patch.object(I.torch,'load',swap_reader):refused('path_swap_restore_same_FD_rejected',lambda:I.load_tensor_verified(a,h),phrase='descriptor consumption')
 ok('path_swap_real_parser_consumed_only_original_A',consumed==[-1.] and sha(a)==h)
 ok('path_swap_failed_context_closed_FD',len(streams)==1 and streams[0].closed)
 observations.append(dict(case='path_swap_restore',parser_consumed=consumed,path_final_SHA_matches=True,published_success=False))

# Inode mutation + restore, including restoring mtime: hashes before/after match
# yet ctime/stat refuses. Real parser may consume B; loader must never publish B.
with tempfile.TemporaryDirectory(prefix='held-fd-inplace-') as tmp:
 a,b,h=tinyfiles(tmp);original=a.read_bytes();st=a.stat();consumed=[];streams=[]
 def inplace_reader(stream,**kwargs):
  streams.append(stream);a.write_bytes(b.read_bytes())
  try:value=original_load(stream,**kwargs);consumed.extend(value['value'].tolist());return value
  finally:a.write_bytes(original);os.utime(a,ns=(st.st_atime_ns,st.st_mtime_ns))
 with patch.object(I.torch,'load',inplace_reader):refused('inplace_restore_and_mtime_restore_rejected',lambda:I.load_tensor_verified(a,h),phrase='descriptor consumption')
 ok('inplace_real_parser_B_cannot_escape_as_success',consumed==[-99.] and sha(a)==h and streams[0].closed)
 ok('ctime_detects_even_restored_mtime',a.stat().st_mtime_ns==st.st_mtime_ns and a.stat().st_ctime_ns!=st.st_ctime_ns)
 observations.append(dict(case='inplace_restore',parser_consumed=consumed,path_final_SHA_matches=True,mtime_restored=True,published_success=False))

# Exception before parser return closes FD and cannot expose a loaded-success value.
with tempfile.TemporaryDirectory(prefix='held-fd-error-') as tmp:
 a,b,h=tinyfiles(tmp);streams=[]
 def bad_reader(stream,**kwargs):streams.append(stream);raise RuntimeError('synthetic parser failed')
 with patch.object(I.torch,'load',bad_reader):refused('parser_exception_propagates',lambda:I.load_tensor_verified(a,h),classes=(RuntimeError,))
 ok('parser_exception_closes_FD',streams[0].closed)

# Synthetic safetensors reader actually consumes generated tensor bytes from the
# passed held FD. It is deliberately not a real safetensors parser/GPU proof.
reader_events=[]
@contextmanager
def synthetic_safe_open(descriptor_path,**kwargs):
 assert descriptor_path.startswith('/proc/self/fd/') and kwargs==dict(framework='pt',device='cpu')
 fd=int(descriptor_path.rsplit('/',1)[1]);held=os.fstat(fd)
 with os.fdopen(os.dup(fd),'rb') as stream:loaded=original_load(stream,map_location='cpu',weights_only=False)
 reader_events.append(dict(path=descriptor_path,fd=fd,dev=held.st_dev,ino=held.st_ino,keys=sorted(loaded)))
 class Shard:
  def keys(self):return list(loaded)
  def get_tensor(self,key):return loaded[key]
 yield Shard()
with tempfile.TemporaryDirectory(prefix='held-fd-safe-') as tmp:
 a,b,h=tinyfiles(tmp)
 with I.safetensor_verified(a,h,synthetic_safe_open) as shard:
  ok('synthetic_shard_reader_consumes_expected_A',shard.get_tensor('value').tolist()==[-1.])
  fd=reader_events[-1]['fd'];ok('safetensor_reader_FD_live_inside_context',os.fstat(fd).st_ino==a.stat().st_ino)
 try:os.fstat(fd)
 except OSError:ok('safetensor_reader_held_FD_closed_after_context',True)
 else:raise AssertionError('held FD leaked')
with tempfile.TemporaryDirectory(prefix='held-fd-safe-swap-') as tmp:
 a,b,h=tinyfiles(tmp);backup=Path(tmp)/'backup';consumed=[]
 @contextmanager
 def swapped_safe_open(path,**kwargs):
  a.rename(backup);b.rename(a)
  try:
   with synthetic_safe_open(path,**kwargs) as shard:consumed.extend(shard.get_tensor('value').tolist());yield shard
  finally:a.rename(b);backup.rename(a)
 def consume():
  with I.safetensor_verified(a,h,swapped_safe_open):pass
 refused('safetensor_path_swap_restore_refused',consume,phrase='descriptor consumption')
 ok('synthetic_safetensor_actual_parser_stays_on_A',consumed==[-1.] and sha(a)==h)

# Real loader inventory calls the new descriptor wrapper with actual expected
# file map; absent/unknown shard/hash must refuse without inventory receipt.
class Tiny(torch.nn.Module):
 def __init__(self):
  super().__init__();self.model=torch.nn.Module();self.model.embed_tokens=torch.nn.Embedding(2,2)
  self.lm_head=torch.nn.Linear(2,2,bias=False);self.lm_head.weight=self.model.embed_tokens.weight
m=Tiny();saved={n:p.detach().clone() for n,p in m.named_parameters(remove_duplicate=False)}
fake=ModuleType('safetensors');fake.safe_open=synthetic_safe_open
with tempfile.TemporaryDirectory(prefix='held-fd-loader-') as tmp,patch.dict(sys.modules,{'safetensors':fake}):
 p=Path(tmp)/'tiny.safetensors';p.write_bytes(payload(saved));h=sha(p);index={'weight_map':{k:p.name for k in saved}}
 targets,receipt=L.exhaustive_hf_cast_inventory(m,tmp,index,{p.name:h},2,1)
 ok('loader_full_tiny_inventory_uses_held_shard_and_returns_cast_receipt',receipt['serialized_count']==2 and receipt['unique_parameter_count']==1 and {t['logical_name'] for t in targets}==set(saved))
 for name,expected in [('missing_hash_map',{}),('unknown_shard_in_map',{'unknown.safetensors':h}),('unknown_hash_value',{p.name:None}),('changed_hash_value',{p.name:'0'*64})]:
  count=len(reader_events)
  refused('loader_'+name,lambda e=expected:L.exhaustive_hf_cast_inventory(m,tmp,index,e,2,1))
  ok('loader_'+name+'_parser_not_called',len(reader_events)==count)
 # Change file during actual synthetic parser callback and restore it: values
 # remain A via FD, but final identity must prevent even an apparently good receipt.
 fake.safe_open=synthetic_safe_open
 backup=Path(tmp)/'backup.safetensors';other=Path(tmp)/'other.bin';other.write_bytes(payload({k:v+3 for k,v in saved.items()}))
 @contextmanager
 def inventory_swapped_reader(path,**kwargs):
  p.rename(backup);other.rename(p)
  try:
   with synthetic_safe_open(path,**kwargs) as shard:yield shard
  finally:p.rename(other);backup.rename(p)
 fake.safe_open=inventory_swapped_reader
 refused('loader_path_swap_restore_no_cast_receipt_published',lambda:L.exhaustive_hf_cast_inventory(m,tmp,index,{p.name:h},2,1),phrase='descriptor consumption')
 ok('loader_path_swap_restored_frozen_bytes',sha(p)==h)

# Source wiring and exact small loader delta; no other loader behavior changed.
old=ast.parse((ROOT/'work/tree-probability-probe-runner/search_r1/baselines/probability_probe_loader.py').read_text());new=ast.parse((REPO/'search_r1/baselines/probability_probe_loader.py').read_text())
new.body=[n for n in new.body if not(isinstance(n,ast.ImportFrom) and n.module=='probability_probe_io')]
method=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=='exhaustive_hf_cast_inventory')
method.args.args=[a for a in method.args.args if a.arg!='expected_files']
loop=next(n for n in method.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='filename')
loop.body=[n for n in loop.body if not(isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and any(isinstance(a,ast.Compare) and any(isinstance(o,ast.In) for o in a.ops) for a in n.value.args))]
withnode=next(n for n in loop.body if isinstance(n,ast.With));withnode.items[0].context_expr=ast.parse("safe_open(str(path),framework='pt',device='cpu')",mode='eval').body
ok('loader_AST_only_expected_hash_and_descriptor_reader_delta',ast.dump(old,include_attributes=False)==ast.dump(new,include_attributes=False))
runner=ast.parse((REPO/'search_r1/baselines/probability_probe_runner.py').read_text())
run=next(n for n in runner.body if isinstance(n,ast.FunctionDef) and n.name=='run_probe');assets=next(n for n in run.body if isinstance(n,ast.FunctionDef) and n.name=='load_assets')
calls=[n for n in ast.walk(assets) if isinstance(n,ast.Call)]
ok('raw_and_companion_load_assets_both_verified_descriptor_calls',sum(isinstance(n.func,ast.Name) and n.func.id=='load_tensor_verified' for n in calls)==2 and not any(isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='torch' and n.func.attr=='load' for n in calls))
hfcalls=[n for n in ast.walk(run) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='exhaustive_hf_cast_inventory']
ok('runner_passes_bundle_HF_identity_files_expected_hashes',len(hfcalls)==1 and ast.dump(hfcalls[0].args[3],include_attributes=False)==ast.dump(ast.parse("bundle['hf_identity']['files']",mode='eval').body,include_attributes=False))
oldce=ROOT/'work/tree-probability-probe-runner-review/LOAD_ASSET_TOCTOU_COUNTEREXAMPLE.json'
ok('old_cfce_counterexample_preserved',sha(oldce)=='cfce3f3abc9be28b63b992e335337180a6722042bba0d1ad49680546469283e8')
result=dict(status='V2_IO_SAME_DESCRIPTOR_INDEPENDENT_TINY_CPU_CASES_PASSED',case_count=len(cases),cases=cases,parser_observations=observations,synthetic_safetensor_reader_calls=len(reader_events),actual_PT_HF_assets_loaded=False,actual_GPU_executed=False,actual_remote_calls=0,actual_model_forwards=0,actual_optimizer_writes=0,old_136_tests_reexecuted=False,scope='Tiny generated Torch archive bytes and explicit synthetic safetensors held-FD reader only; actual parser input bytes tracked. No actual HF/PT/GPU/remote and no actual IO/runtime release receipt.')
(HERE/'IO_CHILD_CPU_PROOF.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(status=result['status'],case_count=len(cases))))
