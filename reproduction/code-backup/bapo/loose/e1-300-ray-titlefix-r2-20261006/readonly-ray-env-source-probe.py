import ray,pathlib,hashlib,json
root=pathlib.Path(ray.__file__).parent;found=[]
for p in root.rglob('*.py'):
 text=p.read_text(errors='replace')
 lines=[{'line':i,'text':v.strip()}for i,v in enumerate(text.splitlines(),1)if 'SPT_NOENV' in v or 'setproctitle.setproctitle('in v]
 if lines:found.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'lines':lines})
print(json.dumps({'scope':'READONLY_INSTALLED_RAY_SOURCE_NO_RAY_INIT_OR_GPU','ray_version':ray.__version__,'found':found},indent=2))
