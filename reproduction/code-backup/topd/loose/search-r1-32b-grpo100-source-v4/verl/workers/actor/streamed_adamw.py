"""Torch2.4 original AdamW, CPU state storage and one-parameter CUDA staging."""
import contextlib,hashlib,importlib,pathlib,threading,types,inspect
import torch
OPTIMIZER_SOURCE_SHA='46f911d59955d779b3323a61cbc35c33725da64edfff6b9b2fa7e59b732b1a5b'
SOURCE_SHA='fe1a551e4cadce2b0ecbdb74e2c7827fdc52c51bfb6fb0a11f0aee7e792596b6'
_LOCK=threading.RLock()
_ACTIVE=False
class _CPUInitTorch:
 def __getattr__(self,name):return getattr(torch,name)
 def zeros_like(self,p,**kwargs):
  if 'device' in kwargs:raise RuntimeError('Unexpected original state initializer device')
  return torch.zeros_like(p,device='cpu',**kwargs)
def _validate(optimizer,module):
 if type(optimizer) is not module.AdamW:raise RuntimeError('Only original Torch2.4 AdamW supported')
 for g in optimizer.param_groups:
  if g['foreach'] is not False or any(g.get(k,False) for k in ('fused','capturable','differentiable')):raise RuntimeError('Streaming requires foreachFalse/nonfused/noncapturable/nondifferentiable')
  for p in g['params']:
   if p.dtype!=torch.float32 or p.is_complex():raise RuntimeError('Only original FP32 real master parameters supported')
   for k,t in optimizer.state.get(p,{}).items():
    if isinstance(t,torch.Tensor) and t.device.type!='cpu':raise RuntimeError('Existing optimizer state must already be CPU offloaded')
 if getattr(optimizer,'grad_scale',None) is not None or getattr(optimizer,'found_inf',None) is not None:raise RuntimeError('AMP scaler state unsupported')
@contextlib.contextmanager
def _namespace(optimizer,module):
 global _ACTIVE
 with _LOCK:
  if _ACTIVE:raise RuntimeError('Nested streamed AdamW context unsupported')
  _validate(optimizer,module)
  original_single=module._single_tensor_adamw;original_init=optimizer._init_group;had_instance='_init_group' in optimizer.__dict__;saved_instance=optimizer.__dict__.get('_init_group')
  if original_init.__func__ is not module.AdamW._init_group:raise RuntimeError('Original AdamW initializer required; instance override unsupported')
  owned={id(p):p for g in optimizer.param_groups for p in g['params']};owner_thread=threading.get_ident()
  public_init=original_init.__func__
  body=inspect.unwrap(public_init)
  if body is not public_init:
   if not (getattr(public_init,'_torchdynamo_disable',False) is True and getattr(public_init,'_torchdynamo_orig_callable',None) is body and getattr(public_init,'__wrapped__',None) is body):raise RuntimeError('Unsupported original initializer wrapper chain')
  cpu_globals=dict(body.__globals__,torch=_CPUInitTorch())
  cpu_init=types.FunctionType(body.__code__,cpu_globals,body.__name__,body.__defaults__,body.__closure__)
  cpu_init.__kwdefaults__=body.__kwdefaults__
  # Rebuild the original public Dynamo-disable semantics around the cloned body.
  # Cloning the wrapper itself would retain its closure to the GPU initializer.
  if body is not public_init:cpu_init=torch._dynamo.disable(cpu_init)
  def streamed(params,grads,exp_avgs,exp_avg_sqs,max_exp_avg_sqs,state_steps,grad_scale,found_inf,**kwargs):
   membership=[id(p) in owned for p in params]
   if not any(membership):return original_single(params,grads,exp_avgs,exp_avg_sqs,max_exp_avg_sqs,state_steps,grad_scale,found_inf,**kwargs)
   if threading.get_ident()!=owner_thread:raise RuntimeError('Same optimizer concurrently stepped outside owned lock')
   if not all(membership):raise RuntimeError('Mixed optimizer parameter dispatch unsupported')
   # Validate all incoming references before any copy-back replaces list objects.
   for i,p in enumerate(params):
    state=optimizer.state.get(p,{})
    required=[('exp_avg',exp_avgs),('exp_avg_sq',exp_avg_sqs),('step',state_steps)]+([('max_exp_avg_sq',max_exp_avg_sqs)] if kwargs['amsgrad'] else [])
    if any(key not in state or values[i] is not state[key] for key,values in required):raise RuntimeError('Shared parameter dispatch from another optimizer unsupported')
   for i,p in enumerate(params):
    state=optimizer.state[p];moved={};primary=None
    try:
     for key,values in [('exp_avg',exp_avgs),('exp_avg_sq',exp_avg_sqs)]+([('max_exp_avg_sq',max_exp_avg_sqs)] if kwargs['amsgrad'] else []):
      moved[key]=state[key].to(device=p.device,non_blocking=False);values[i]=moved[key]
     # Original CPU step tensor is passed by reference; never copy/reset its counter.
     original_single([p],[grads[i]],[exp_avgs[i]],[exp_avg_sqs[i]],[max_exp_avg_sqs[i]] if kwargs['amsgrad'] else [],[state_steps[i]],grad_scale,found_inf,**kwargs)
    except BaseException as error:primary=error;raise
    finally:
     try:
      for key,values in [('exp_avg',exp_avgs),('exp_avg_sq',exp_avg_sqs)]+([('max_exp_avg_sq',max_exp_avg_sqs)] if kwargs['amsgrad'] else []):
       if key in moved:state[key]=moved[key].to(device='cpu',non_blocking=False)
       values[i]=state[key]
      moved.clear()
      if p.device.type=='cuda':torch.cuda.synchronize(p.device);torch.cuda.empty_cache()
     except BaseException as cleanup:
      if primary is not None:raise primary from cleanup
      raise
  try:
   _ACTIVE=True
   optimizer._init_group=types.MethodType(cpu_init,optimizer)
   module._single_tensor_adamw=streamed
   yield
  finally:
   module._single_tensor_adamw=original_single
   if had_instance:optimizer._init_group=saved_instance
   elif '_init_group' in optimizer.__dict__:del optimizer._init_group
   _ACTIVE=False

def step_cpu_offloaded_adamw(optimizer):
 module=importlib.import_module('torch.optim.adamw');optimizer_module=importlib.import_module('torch.optim.optimizer')
 if hashlib.sha256(pathlib.Path(optimizer_module.__file__).read_bytes()).hexdigest()!=OPTIMIZER_SOURCE_SHA:raise RuntimeError('Exact original Torch2.4 optimizer wrapper bytes required')
 if torch.__version__.split('+')[0]!='2.4.0' or hashlib.sha256(pathlib.Path(module.__file__).read_bytes()).hexdigest()!=SOURCE_SHA:raise RuntimeError('Exact original Torch2.4 AdamW bytes required')
 with _namespace(optimizer,module):return optimizer.step()
