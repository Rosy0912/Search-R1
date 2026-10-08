"""Actual runtime helper/import/config binding; no GPU/model work."""
import hashlib,json,os,sys
from pathlib import Path

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def verify_runtime(bundle,current_file,config_path=None,modules=None):
 root=Path(bundle['directory']).resolve();current=Path(current_file).resolve()
 if not current.is_relative_to(root):raise ValueError('Actual executing file outside reviewed bundle directory')
 relative=str(current.relative_to(root));record=bundle['file_sources'].get(relative)
 if record!={'path':str(current),'sha256':digest(current)}:raise ValueError('Actual executing file path/SHA differs from reviewed bundle')
 modules=sys.modules if modules is None else modules
 module_roles={}
 for name in sorted(bundle['file_sources'],key=lambda n:(n.count('/'),n)):
  if name.endswith('.py'):module_roles.setdefault(Path(name).stem,name)
 for name,pin in bundle['file_sources'].items():
  expected=root/name
  if Path(pin['path']).resolve()!=expected or digest(expected)!=pin['sha256']:raise ValueError('Actual helper path/SHA outside exact bundle directory')
  if expected.suffix=='.py' and module_roles[expected.stem]==name:
   module=modules.get(expected.stem)
   if module is not None and (not getattr(module,'__file__',None) or Path(module.__file__).resolve()!=expected):raise ValueError('Actual imported helper outside reviewed bundle')
 cfg=bundle['config_binding']
 if set(cfg)!={'path','sha256'} or not Path(cfg['path']).is_absolute() or digest(cfg['path'])!=cfg['sha256']:raise ValueError('Actual externally pinned resolved config differs')
 if config_path is not None and Path(config_path).resolve()!=Path(cfg['path']).resolve():raise ValueError('Actual config argv path outside external binding')
 return {'directory':str(root),'executing_file':record,'config_binding':cfg}

def load_runtime(current_file,config_path=None):
 from source_bundle_check import verify_bundle
 path=Path(os.environ['CORRECTED_ACTIVATED_BUNDLE_MANIFEST'])
 if digest(path)!=os.environ['CORRECTED_ACTIVATED_BUNDLE_SHA256']:raise ValueError('External launcher bundle SHA differs')
 bundle=json.loads(path.read_text());root=Path(os.environ['CORRECTED_EVAL_BUNDLE']).resolve()
 if root!=Path(bundle['directory']).resolve():raise ValueError('Actual CORRECTED_EVAL_BUNDLE differs from verified source directory')
 verify_bundle(bundle,Path(os.environ['CORRECTED_CANONICAL_TEMPLATE']))
 if os.environ.get('CORRECTED_EVAL_CONFIG_SHA256')!=bundle['config_binding']['sha256']:raise ValueError('Actual external config environment SHA differs')
 return bundle,verify_runtime(bundle,current_file,config_path)
