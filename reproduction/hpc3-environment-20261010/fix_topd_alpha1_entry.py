import pathlib,json,hashlib,sys
base=pathlib.Path('/data/user/jxu957/experiments/topd-20261008')
root=base/'runtime/tools/topd-alpha1-launch-20261012'
p=root/'launch.sbatch'
s=p.read_text().replace('$TASK_ROOT/tools/topd-instruct300-launch-v6','$TASK_ROOT/tools/topd-alpha1-launch-20261012')
assert 'topd-instruct300-launch-v6' not in s
p.write_text(s)
cpath=base/'outputs/training-preparation/hpc3-topd200-alpha1.json'
c=json.loads(cpath.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for key in ['tools_manifest','launcher_manifest']:
 p=pathlib.Path(c[key]['path']);d=json.loads(p.read_text())
 d['files']={str(f.relative_to(p.parent)):{'size':f.stat().st_size,'sha256':sha(f)} for f in p.parent.rglob('*') if f.is_file() and f!=p and '__pycache__' not in f.parts}
 p.write_text(json.dumps(d,sort_keys=True,indent=2));c[key]['sha256']=sha(p)
cpath.write_text(json.dumps(c,indent=2))
sys.path.insert(0,str(root/'runtime'))
from supervise import verify_recipe
from student_binding import check
assert verify_recipe(c)==200
check(c)
print('ALPHA1_ENTRY_AND_BINDINGS_PASS',sha(root/'launch.sbatch'))
