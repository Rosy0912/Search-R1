"""External CPU diagnosis, no source edits, no GPU; compile backend=eager only."""
import os,sys,json,types,importlib.util
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
R=Path(__file__).resolve().parent.parent/'opsd-e1-300';sys.path.insert(0,str(R))
import torch
import verl.utils.torch_functional as f
p=R/'reproduction/baselines/e1_import_bootstrap.py';s=importlib.util.spec_from_file_location('diagnostic_source_checker',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
# Native library wrapper rows are included; no bypass introduced.
allowed={str(p.resolve()):m.compiled_codes(p)}
for name in m.LIBRARY_MODULES:
 path=m.source_path(R,name);allowed[str(path)]=m.compiled_codes(path)
path=Path(f.__file__).resolve();allowed[str(path)]=m.compiled_codes(path)
before=m.check_live_code(f,allowed);fn=torch.compile(f.entropy_from_logits,dynamic=True,backend='eager');v=fn(torch.arange(32,dtype=torch.float32).reshape(4,8));assert torch.isfinite(v).all() and not torch.cuda.is_initialized()
generated={k:{'filename':v.__code__.co_filename,'name':v.__code__.co_name,'firstlineno':v.__code__.co_firstlineno} for k,v in vars(f).items() if isinstance(v,types.FunctionType) and 'torch_dynamo_resume' in v.__qualname__ or (isinstance(v,types.FunctionType) and k.startswith('__resume_at_'))}
try:after=m.check_live_code(f,allowed)
except ValueError as e:error=str(e)
else:error=None
print(json.dumps({'scope':'CPU compile eager backend diagnosis only; not GPU/native2.4 or algorithm acceptance','torch':torch.__version__,'before_live_functions':before,'generated_functions':generated,'actual_guard_error':error,'cuda_is_initialized':torch.cuda.is_initialized()},sort_keys=True))
assert error is not None and 'torch_dynamo_resume' in error
