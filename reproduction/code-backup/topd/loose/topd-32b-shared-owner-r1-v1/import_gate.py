"""Verify the complete external IPC package before importing any of its Python."""
import hashlib,importlib,json,os,sys
from pathlib import Path
ROOT=Path('/dataset/shaoshitong/search-r1-rl/topd')
PINS_SHA='d4914b8c46b12737a9e095bd27d5921baa644e6d412b6e2e7c6ef46a458dc7d5'
def require(value,message):
 if not value:raise RuntimeError(message)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pins():
 path=Path(__file__).with_name('ipc-source-pins.json');require(sha(path)==PINS_SHA,'IPC pin manifest drift');return json.loads(path.read_text())
def verify_package(directory,records):
 directory=Path(directory)
 require(directory.is_dir() and not directory.is_symlink(),'IPC package directory alias/missing')
 actual=set()
 for base,dirs,files in os.walk(directory,followlinks=False):
  for name in dirs+files:require(not Path(base,name).is_symlink(),'IPC source symlink rejected')
  actual.update(str(Path(base,name).relative_to(directory)) for name in files)
 require(actual==set(records),'IPC exact source file set drift')
 for name,digest in records.items():
  p=directory/name;require(p.is_file() and sha(p)==digest,'IPC source SHA drift: '+name)
 return directory.resolve()
def verified_ipc():
 require(os.environ.get('PYTHONDONTWRITEBYTECODE')=='1','IPC no source bytecode writes required')
 expected=ROOT/'tools/topd-32b-pipeline-ipc-r1-v1/topd32b_shared_pipeline'
 require(expected.resolve()==expected,'IPC canonical own directory required')
 package=verify_package(expected,pins())
 names={'topd32b_shared_pipeline':'__init__.py','topd32b_shared_pipeline.backend':'backend.py','topd32b_shared_pipeline.protocol':'protocol.py','topd32b_shared_pipeline.readiness':'readiness.py'}
 def origins():
  for name,module in list(sys.modules.items()):
   if name=='topd32b_shared_pipeline' or name.startswith('topd32b_shared_pipeline.'):
    require(name in names and Path(module.__file__).resolve()==package/names[name] and sha(module.__file__)==pins()[names[name]],'Cached/imported foreign IPC module: '+name)
 origins();sys.path.insert(0,str(package.parent))
 backend=importlib.import_module('topd32b_shared_pipeline.backend');importlib.import_module('topd32b_shared_pipeline.readiness');origins()
 return backend
