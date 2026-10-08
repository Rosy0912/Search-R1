import sys
sys.dont_write_bytecode=True
import hashlib,json,os,pathlib
import torch,pytest
here=pathlib.Path(__file__).resolve().parent
source=pathlib.Path(os.environ['R1_ENTROPY_SOURCE']).resolve()
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd')
assert root in here.parents and root in source.parents
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and torch.__version__.split('+')[0]=='2.4.0'
for directory,name in [(here,'files.json'),(source,'r1-files.json')]:
 m=json.loads((directory/name).read_text());actual={str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}
 assert actual==set(m['files'])|{name},('Exact files differ',directory)
 for rel,rec in m['files'].items():
  p=directory/rel;assert not p.is_symlink() and p.stat().st_size==rec['size'] and hashlib.sha256(p.read_bytes()).hexdigest()==rec['sha256']
assert not torch.cuda.is_initialized()
out=root/'logs'/('r1-entropy-source-v5-cpu-'+os.environ['SLURM_JOB_ID']);out.mkdir(mode=0o700,exist_ok=False)
class Results:
 def __init__(self):self.count=0;self.skips=0;self.failures=0
 def pytest_runtest_logreport(self,report):
  if report.when=='call':self.count+=1;self.failures+=int(report.failed)
  self.skips+=int(report.skipped)
r=Results();rc=pytest.main([str(here/'test_entropy_compile.py'),'-q','-p','no:cacheprovider','--basetemp',str(out/'tmp')],plugins=[r])
receipt={'job':os.environ['SLURM_JOB_ID'],'tests':r.count,'skips':r.skips,'failures':r.failures,'exit_code':int(rc),'cuda_initialized':torch.cuda.is_initialized(),'source_manifest_sha256':hashlib.sha256((source/'r1-files.json').read_bytes()).hexdigest()}
with (out/'result.json').open('x') as f:json.dump(receipt,f,indent=2);f.flush();os.fsync(f.fileno())
assert r.count==4 and r.skips==0 and r.failures==0 and rc==0 and not torch.cuda.is_initialized(),receipt
print(json.dumps(receipt),flush=True)
