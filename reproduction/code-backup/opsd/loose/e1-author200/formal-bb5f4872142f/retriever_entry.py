"""Record actual FAISS runtime origin before unchanged frozen retrieval source."""
import hashlib,json,os,runpy,sys
from pathlib import Path
import torch
import faiss
repo=Path(os.environ['SEARCH_R1_REPO']);runtime=Path('/mnt/data/shaoshitong/search-r1-rl/opsd/retriever-gpu-lib')
p=Path(faiss.__file__).resolve();assert runtime.resolve()==runtime and runtime in p.parents
files={}
for name,module in list(sys.modules.items()):
 if name=='faiss' or name.startswith('faiss.'):
  path=Path(module.__file__).resolve();assert runtime in path.parents
  files[name]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
with (Path(os.environ['E1_FORMAL_RUN'])/'actual-faiss-runtime.json').open('x') as f:
 json.dump(dict(pid=os.getpid(),faiss_version=getattr(faiss,'__version__',None),runtime_root=str(runtime),modules=files,pythonpath=os.environ['PYTHONPATH']),f,sort_keys=True)
runpy.run_path(str(repo/'reproduction/h20/retriever_local.py'),run_name='__main__')
