"""Isolated alpha=1 arm; never mutate the active alpha=.1 runtime."""
import pathlib,json,shutil,hashlib,sys
BASE=pathlib.Path('/data/user/jxu957/experiments/topd-20261008')
old=BASE/'runtime/tools/topd-instruct300-launch-v6'
new=BASE/'runtime/tools/topd-alpha1-launch-20261012'
assert not new.exists(),'Inspect existing arm before retrying'
shutil.copytree(old,new,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for p in new.rglob('*'):
 if p.is_file() and p.suffix in ['.py','.json','.sbatch','.sh']:
  s=p.read_text().replace(str(old),str(new))
  if p.name=='supervise.py':
   assert s.count("'baselines.topd.alpha':'0.1'")==1
   s=s.replace("'baselines.topd.alpha':'0.1'","'baselines.topd.alpha':'1'")
  if p.name=='launch.sbatch':s=s.replace('--job-name=topd-instruct200','--job-name=topd-alpha1-200')
  p.write_text(s)
c=json.loads((BASE/'outputs/training-preparation/hpc3-topd200-protocol-draft.json').read_text())
c=json.loads(json.dumps(c).replace(str(old),str(new)))
assert c['training_overrides'].count('baselines.topd.alpha=0.1')==1
c['training_overrides']=[x.replace('baselines.topd.alpha=0.1','baselines.topd.alpha=1') for x in c['training_overrides']]
c['experiment_arm']='TOPD_ALPHA_1_FRESH_OFFICIAL_INITIALIZATION'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for key in ['tools_manifest','launcher_manifest']:
 p=pathlib.Path(c[key]['path']);d=json.loads(p.read_text())
 d['files']={str(f.relative_to(p.parent)):{'size':f.stat().st_size,'sha256':sha(f)} for f in p.parent.rglob('*') if f.is_file() and f!=p and '__pycache__' not in f.parts}
 p.write_text(json.dumps(d,sort_keys=True,indent=2));c[key]['sha256']=sha(p)
out=BASE/'outputs/training-preparation/hpc3-topd200-alpha1.json';out.write_text(json.dumps(c,indent=2))
sys.path.insert(0,str(new/'runtime'))
from supervise import verify_recipe
from student_binding import check
assert verify_recipe(c)==200
check(c)
print('ISOLATED_ALPHA1_CONFIG_AND_SOURCE_BINDINGS_PASS',str(out))
