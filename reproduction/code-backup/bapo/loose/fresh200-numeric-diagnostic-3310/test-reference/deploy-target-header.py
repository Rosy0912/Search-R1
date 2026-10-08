from pathlib import Path
import os,sys,base64,hashlib,io,tarfile,subprocess,json
if sys.flags.optimize:raise RuntimeError("Unoptimized deployment interpreter required")
S=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-behavior-20261006');R=S/'numeric-diagnostic-3310-20261006'
assert os.uname().nodename=='h100-5' and (S/'source-expected.txt').read_text().strip()=='90755a561abd239fbb23882939410e643bb913ae'
blob=base64.b64decode(BLOB)
assert hashlib.sha256(blob).hexdigest()==SHA
R.mkdir(exist_ok=False)
for n in ['logs','tools','runs','cache','scheduler-head','tests','test-reference']:(R/n).mkdir()
with tarfile.open(fileobj=io.BytesIO(blob),mode='r:gz') as t:
 for m in t.getmembers():
  p=Path(m.name);assert m.isfile() and not p.is_absolute() and '..' not in p.parts
  q=R/p;assert not q.exists();q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(t.extractfile(m).read())
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(R/'tools')+':'+str(S/'repo'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1');env.pop('PYTHONOPTIMIZE',None)
for script in ['run-target-tests.py','capture-cpu-runtime.py']:
 result=subprocess.run([str(S/'env/bin/python'),'-B',str(R/'tools'/script)],env=env,cwd=S/'repo')
 assert result.returncode==0,(script,result.returncode)
print(json.dumps({'scope':'NEW_DIAGNOSTIC_TARGET_CPU_DEPLOYMENT_NOT_GPU','root':str(R),'deployment_sha256':SHA,'gpu_job_submitted':False}))
