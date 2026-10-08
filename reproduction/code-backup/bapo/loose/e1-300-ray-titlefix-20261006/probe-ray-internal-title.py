import ray,ray._raylet as r,inspect,pathlib,hashlib,json,sys
base=pathlib.Path(ray.__file__).parent;rows=[]
for p in base.rglob('*.py'):
 source=p.read_text(errors='replace');lines=source.splitlines()
 for i,line in enumerate(lines):
  if 'setproctitle' in line or 'set_process_title' in line:
   rows.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),line=i+1,context='\n'.join(lines[max(0,i-5):i+10])))
attrs={}
for n in dir(r):
 if 'title' in n.lower() or 'proctitle' in n.lower():
  f=getattr(r,n);attrs[n]=dict(type=type(f).__name__,module=getattr(f,'__module__',None),doc=getattr(f,'__doc__',None))
print(json.dumps(dict(scope='READ_ONLY_INSTALLED_RAY_TITLE_SOURCE_NO_CALL',raylet_path=r.__file__,raylet_sha256=hashlib.sha256(pathlib.Path(r.__file__).read_bytes()).hexdigest(),raylet_attrs=attrs,python_source_matches=rows),indent=2))
