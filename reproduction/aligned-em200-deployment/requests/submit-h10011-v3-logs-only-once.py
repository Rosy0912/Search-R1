"""Run the installed single-use node11 submission; never replay an unknown intent."""
import os,subprocess,json,datetime
from pathlib import Path
assert os.uname().nodename=='h100-11'
r=Path('/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v3-20261009')

old=Path('/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v2-20261009')
receipt=json.loads((old/'scheduler-head/formal-submit-receipt.json').read_text());assert receipt['job']=='3352'
run=Path(receipt['run']);assert (run/'original-exit-code.txt').read_text().strip()=='1' and (run/'cleanup-exit-code.txt').read_text().strip()=='0'
q=subprocess.run(['squeue','-h','-j','3352','-o','%i|%T'],capture_output=True,text=True);assert not q.stdout.strip() and (q.returncode==0 or 'Invalid job id' in q.stderr)
import tarfile,hashlib
archive=old/'evidence/formal3352-failure-logs.tar';archive.parent.mkdir(exist_ok=True)
if not archive.exists():
 with tarfile.open(archive,'x')as tf:
  for p in run.rglob('*'):
   if p.is_file() and not p.is_symlink() and p.suffix!='.pt' and p.stat().st_size<5000000:tf.add(p,arcname=str(p.relative_to(run)),recursive=False)
terminal=subprocess.run(['scontrol','show','job','3352','-o'],capture_output=True,text=True)
print(json.dumps({'old3352_terminal':terminal.stdout,'archive':str(archive),'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'old_exit':1,'old_cleanup':0}),flush=True)

assert not (r/'scheduler-head/formal-submit-intent.json').exists()
e=dict(os.environ,PYTHONPATH=str(r/'tools')+':'+str(r/'retriever-gpu-lib')+':'+str(r/'repo'),CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1');e.pop('LD_LIBRARY_PATH',None)
p=subprocess.run([str(r/'env/bin/python'),'-B',str(r/'tools/deployment/cohost-node11-v3/submit_formal.py')],env=e,capture_output=True,text=True,timeout=65)
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'scope':'ACTUAL_NODE11_SINGLE_SUBMISSION_RESPONSE_NOT_TRAINING'}));raise SystemExit(p.returncode)
