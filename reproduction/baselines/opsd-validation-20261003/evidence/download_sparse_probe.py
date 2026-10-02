import hashlib, importlib.util, json, subprocess, tempfile
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[4]
spec=importlib.util.spec_from_file_location('download_assets',ROOT/'reproduction/h20/download_assets.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
results={}
with tempfile.TemporaryDirectory() as td:
 m.BASE=Path(td)
 for name,size,sidecar,exitcode,good in [('sparse_interrupted',65*1024**2,True,1,False),('sparse_false_success',65*1024**2,True,0,False),('sparse_verified_success',65*1024**2,True,0,True),('boundary_curl',64*1024**2,False,1,False)]:
  path=m.BASE/'assets'/name; path.parent.mkdir(parents=True,exist_ok=True)
  tmp=path.with_name(path.name+'.partial')
  with tmp.open('wb') as f:f.truncate(size)
  state=tmp.with_name(tmp.name+'.aria2')
  if sidecar:state.write_bytes(b'bitmap fixture')
  expected=m.digest(tmp) if good else 'f'*64
  a={'path':name,'size':size,'sha256':expected,'kind':'dataset','repo':'test/source','revision':'fixed','file':name}
  before_blocks=tmp.stat().st_blocks
  with patch.object(m.subprocess,'run',return_value=subprocess.CompletedProcess([],exitcode,'','')) as run:
   ok=m.fetch(a)
  assert ok==good,(name,ok)
  assert path.exists()==good
  assert run.call_args.args[0][0]==('aria2c' if size>64*1024**2 else 'curl')
  if sidecar and not good:
   assert tmp.exists() and state.read_bytes()==b'bitmap fixture'
   assert not list(tmp.parent.glob(tmp.name+'.rejected.*'))
  results[name]={'ok':ok,'promoted':path.exists(),'sparse_allocated_bytes_before':before_blocks*512,'logical_bytes':size,'tool':run.call_args.args[0][0]}
print(json.dumps(results,indent=2))
