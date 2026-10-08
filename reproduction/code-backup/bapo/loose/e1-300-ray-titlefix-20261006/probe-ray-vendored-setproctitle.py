# coding: utf-8
import hashlib,importlib,importlib.util,inspect,json,pathlib,sys,os
names=['setproctitle','ray','ray._private.thirdparty.setproctitle','ray._private.thirdparty.setproctitle._setproctitle','ray._private.utils']
rows={}
for name in names:
 try:
  mod=importlib.import_module(name);p=pathlib.Path(mod.__file__).resolve();r=dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
  if hasattr(mod,'setproctitle'):
   f=mod.setproctitle;r['function']=dict(module=f.__module__,name=f.__name__,builtin=inspect.isbuiltin(f))
   try:r['function']['source']=inspect.getsource(f)
   except (OSError,TypeError):r['function']['source_unavailable_native']=True
  if name=='ray':r['version']=mod.__version__
  if name=='ray._private.utils':
   source=p.read_text();lines=source.splitlines();r['setproctitle_source_context']=[dict(line=i+1,text='\n'.join(lines[max(0,i-4):i+7]))for i,line in enumerate(lines)if'setproctitle'in line]
  rows[name]=r
 except Exception as e:rows[name]=dict(error=type(e).__name__,message=str(e))
print(json.dumps(dict(scope='READ_ONLY_EXISTING_DEPENDENCY_IMPORT_NO_TITLE_CHANGE_NO_SIGNAL_NO_GPU',sys_executable=sys.executable,sys_prefix=sys.prefix,hostname=os.uname().nodename,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),modules=rows),indent=2))
