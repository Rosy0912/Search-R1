"""Build source-only native225 package; no remote operations or submissions."""
from pathlib import Path
import shutil,hashlib,json,tarfile
W=Path('work');D=W/'standard-identity-v5-native-cpu'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
shutil.copytree(W/'standard-identity-v4-native-cpu-r2',D)
(D/'native-files.json').unlink()
# Exact source snapshots needed by frozen V5 real chain contracts.
for name,rel in [('search-r1-32b-grpo100-source-v2','r1-files.json'),('topd-32b-native-score-gate-r1-v4','worker.py'),('qwen25-32b-trained-eval64-r1-v4','eval64.py')]:
 p=D/name/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(W/name/rel,p)
shutil.copytree(W/'r1-32b-hf-semantic-review-v4',D/'r1-32b-hf-semantic-review-v4',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
p=D/'r1-32b-hf-semantic-review-v4/test_review.py';original=p.read_text();p.unlink()
fixed=original.replace("assert m.job_id(actual['job'])==m.job_id(3163)","assert m.job_id(actual['job'])==int(actual['job'])").replace("assert m.job_id('3164')!=m.job_id(actual['job'])","assert m.job_id(str(int(actual['job'])+1))!=m.job_id(actual['job'])")
(D/'r1-32b-hf-semantic-review-v4/test_review_native.py').write_text(fixed)
(D/'hf-native-fixture-binding.json').write_text(json.dumps({'scope':'ONLY_NATIVE_SLURM_JOB_EXPECTATION_FIX_NOT_SCAN','original_test_sha256':sha(W/'r1-32b-hf-semantic-review-v4/test_review.py'),'native_test_sha256':sha(D/'r1-32b-hf-semantic-review-v4/test_review_native.py'),'original_hf_manifest_sha256':sha(W/'r1-32b-hf-semantic-review-v4/files.json')},indent=2)+'\n')
p=D/'native_cpu_runner.py';s=p.read_text().replace('identity-v4','identity-v5').replace('r1-v4/','r1-v5/').replace("'chain':(34,","'chain':(36,")
s=s.replace("'supervisor':(62,['topd-shared32b-supervisor-r1-v5/test_contracts.py'])}","'supervisor':(62,['topd-shared32b-supervisor-r1-v5/test_contracts.py']),\n'hf':(34,['r1-32b-hf-semantic-review-v4/test_review_native.py'])}")
s=s.replace("[sys.executable,'-m','pytest'","[sys.executable,'-B','-m','pytest'").replace('len(rows)==5','len(rows)==6').replace('passed=189','passed=225')
s=s.replace("for manifest in (", "assert os.environ['PYTHONDONTWRITEBYTECODE']=='1' and sys.dont_write_bytecode\nfor manifest in (")
# Complete frozen HF runtime helpers are verified separately (native test name differs intentionally).
s=s.replace("assert not list(P.rglob('__pycache__'))", "for n,v in json.loads((P/'r1-32b-hf-semantic-review-v4/files.json').read_text())['files'].items():\n if n!='test_review.py':assert sha(P/'r1-32b-hf-semantic-review-v4'/n)==v['sha256']\nassert not list(P.rglob('__pycache__'))",1)
p.write_text(s)
p=D/'native-cpu.sbatch';s=p.read_text().replace('identity-v4-r2','identity-v5').replace('identity-v4-native-r2','identity-v5-native').replace('"$PACKAGE/native_cpu_runner.py"','-B "$PACKAGE/native_cpu_runner.py"');p.write_text(s)
files={str(p.relative_to(D)):{'size':p.stat().st_size,'sha256':sha(p)} for p in sorted(D.rglob('*')) if p.is_file()}
(D/'native-files.json').write_text(json.dumps({'scope':'SOURCE_DEPENDENCY_SNAPSHOTS_NATIVE225_NO_GPU_NO_MODEL','files':files},indent=2)+'\n')
a=W/'standard-identity-v5-native-cpu.tar.gz'
with tarfile.open(a,'w:gz') as tf:
 for p in sorted(D.rglob('*')):
  if p.is_file():tf.add(p,arcname=str(p.relative_to(D)),recursive=False)
with tarfile.open(a) as tf:
 ms=tf.getmembers();assert len({m.name for m in ms})==len(ms)
 for m in ms:assert m.isfile() and tf.extractfile(m).read()==(D/m.name).read_bytes()
print(sha(a),sha(D/'native-files.json'),len(files))
