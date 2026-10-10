"""Check audit failure cannot prevent native checkpoint preservation."""
import ast,functools,sys,tempfile,types
from pathlib import Path
source=ast.parse(Path(__file__).with_name('runtime_audit.py').read_text())
fn=next(n for n in ast.walk(source) if isinstance(n,ast.FunctionDef) and n.name=='save')
with tempfile.TemporaryDirectory() as tmp:
 calls=[]
 def native(worker,path,*args,**kwargs):
  calls.append('native_save');(Path(path)/'native-shard.pt').write_bytes(b'synthetic');return 'saved'
 def audit(worker):calls.append('audit');raise TypeError('Synthetic ListConfig failure')
 ns={'functools':functools,'original_save':native,'state':audit,'Path':Path,'record':lambda *a:calls.append('failure_record')}
 prior=sys.modules.get('torch');sys.modules['torch']=types.SimpleNamespace(distributed=types.SimpleNamespace(get_rank=lambda:0))
 try:
  exec(compile(ast.Module(body=[fn],type_ignores=[]),'actual-save-wrapper','exec'),ns)
  assert ns['save'](object(),tmp)=='saved'
 finally:
  if prior is None:sys.modules.pop('torch',None)
  else:sys.modules['torch']=prior
 assert calls==['native_save','audit','failure_record']
 assert (Path(tmp)/'native-shard.pt').read_bytes()==b'synthetic'
 assert not (Path(tmp)/'recovery-state-rank-0.json').exists()
print('NATIVE_SAVE_PRESERVED_AUDIT_FAILURE_REMAINS_UNACCEPTED_PASS')
