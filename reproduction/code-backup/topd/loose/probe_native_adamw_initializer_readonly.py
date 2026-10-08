import subprocess,pathlib,json
code=r'''
import os,sys,json,inspect,importlib.util,torch
assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not torch.cuda.is_initialized()
assert torch.__version__.split('+')[0]=='2.4.0'
def meta(fn):
 fn=getattr(fn,'__func__',fn);body=inspect.unwrap(fn)
 return {'module':fn.__module__,'name':fn.__name__,'code_file':fn.__code__.co_filename,'code_name':fn.__code__.co_name,'wrapped':hasattr(fn,'__wrapped__'),'dynamo_disable':getattr(fn,'_torchdynamo_disable',None),'orig_callable_is_unwrapped':getattr(fn,'_torchdynamo_orig_callable',None) is body,'unwrapped_file':body.__code__.co_filename,'unwrapped_name':body.__code__.co_name,'closure_names':list(fn.__code__.co_freevars),'unwrapped_torch_is_real':body.__globals__.get('torch') is torch}
rows={'scope':'READ_ONLY_NATIVE_CPU_INITIALIZER_METADATA_AND_SCOPED_SPY_NOT_GPU_PROOF','torch':torch.__version__,'before':meta(torch.optim.AdamW._init_group)}
p=torch.nn.Parameter(torch.ones(2,3,dtype=torch.float32));opt=torch.optim.AdamW([p],lr=.01,foreach=False);p.grad=torch.ones_like(p);rows['after_constructor']=meta(opt._init_group)
path='/dataset/shaoshitong/search-r1-rl/topd/tools/search-r1-32b-grpo100-source-v3/verl/workers/actor/streamed_adamw.py'
spec=importlib.util.spec_from_file_location('owned_cpu_initializer_probe',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
original=torch.zeros_like;calls=[]
def spy(value,*args,**kwargs):
 result=original(value,*args,**kwargs);calls.append({'input_device':str(value.device),'explicit_device':str(kwargs.get('device')),'output_device':str(result.device)});return result
try:
 torch.zeros_like=spy;m.step_cpu_offloaded_adamw(opt)
finally:torch.zeros_like=original
rows['after_step']=meta(opt._init_group);rows['actual_zero_like_calls']=calls;rows['state_devices']={k:str(v.device) for k,v in opt.state[p].items() if isinstance(v,torch.Tensor)};rows['cuda_initialized']=torch.cuda.is_initialized();assert not rows['cuda_initialized'];print(json.dumps(rows))
'''
r=subprocess.run(['python3','work/remote_topd.py'],input="CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /dataset/shaoshitong/search-r1-rl/topd/env/bin/python -B - <<'NATIVE_CPU_PROBE'\n"+code+"\nNATIVE_CPU_PROBE\n",text=True,capture_output=True,timeout=90)
pathlib.Path('outputs/standard3197-native-initializer-metadata-readonly-response.log').write_text(r.stdout+r.stderr)
print(r.stdout);print(r.stderr);raise SystemExit(r.returncode)
