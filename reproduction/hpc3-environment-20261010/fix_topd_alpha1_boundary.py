"""Accept the exact alpha=1 limit in an isolated source snapshot."""
import pathlib,json,shutil,hashlib,sys,importlib.util
BASE=pathlib.Path('/data/user/jxu957/experiments/topd-20261008');ROOT=BASE/'runtime'
old=ROOT/'tools/topd-instruct300-source-v1';new=ROOT/'tools/topd-alpha1-source-20261012'
assert not new.exists(),'Inspect existing source before retry'
shutil.copytree(old,new,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
p=new/'search_r1/baselines/advantages.py';s=p.read_text()
s=s.replace('if not 0 < alpha < 1:', 'if not 0 < alpha <= 1:').replace("TOP-D requires 0 < alpha < 1", "TOP-D requires 0 < alpha <= 1")
s=s.replace('reward = torch.logaddexp(d + math.log(alpha), d.new_full((), math.log1p(-alpha)))','reward = d if alpha == 1 else torch.logaddexp(d + math.log(alpha), d.new_full((), math.log1p(-alpha)))')
p.write_text(s)
import torch
spec=importlib.util.spec_from_file_location('test_advantage',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
t=torch.tensor([[-3.,-1.,float('inf')],[-2.,-5.,float('inf')]]);o=torch.tensor([[-4.,-2.,float('inf')],[-4.,-4.,float('inf')]]);mask=torch.tensor([[1.,1.,0.],[1.,1.,0.]])
a,r,d=m.topd(t,o,mask,['same','same'],1.)
expect=torch.tensor([[1.,1.,0.],[2.,-1.,0.]])
assert torch.equal(d,expect) and torch.equal(r,torch.tensor([[2.,1.,0.],[1.,-1.,0.]])) and torch.isfinite(a).all()
values=torch.tensor([2.,1.,1.,-1.]);expected=(values-values.mean())/(values.std(correction=1)+1e-6)
assert torch.allclose(a[mask.bool()],expected)
for alpha in [.1,.5,.999999,1.]:
 aa,rr,dd=m.topd(t,o,mask,['same','same'],alpha);assert torch.isfinite(aa).all() and torch.isfinite(rr).all() and torch.isfinite(dd).all()
assert torch.allclose(m.topd(t,o,mask,['same','same'],.999999)[2],d,atol=2e-5)
for alpha in [0.,1.1,float('nan')]:
 try:m.topd(t,o,mask,['same','same'],alpha)
 except ValueError:pass
 else:raise AssertionError('Invalid alpha accepted')
launch=ROOT/'tools/topd-alpha1-launch-20261012'
cpath=BASE/'outputs/training-preparation/hpc3-topd200-alpha1.json';c=json.loads(cpath.read_text());c=json.loads(json.dumps(c).replace(str(old),str(new)))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for key in ['source_manifest','tools_manifest','launcher_manifest']:
 p=pathlib.Path(c[key]['path']);record=json.loads(p.read_text())
 record['files']={str(f.relative_to(p.parent)):{'size':f.stat().st_size,'sha256':sha(f)} for f in p.parent.rglob('*') if f.is_file() and f!=p and '__pycache__' not in f.parts}
 p.write_text(json.dumps(record,sort_keys=True,indent=2));c[key]['sha256']=sha(p)
cpath.write_text(json.dumps(c,indent=2))
sys.path.insert(0,str(launch/'runtime'))
from supervise import verify_recipe
from student_binding import check
assert verify_recipe(c)==200;check(c)
atomic=BASE/'outputs/training-preparation/alpha1-boundary-test.json';atomic.write_text(json.dumps({'status':'EXACT_ALPHA1_LIMIT_AND_MASKED_INFINITY_NUMERICAL_TEST_PASS','formula':'reward=teacher_logp-old_logp at alpha=1','source_sha256':sha(new/'search_r1/baselines/advantages.py'),'source_manifest':c['source_manifest']},indent=2))
print(atomic.read_text())
