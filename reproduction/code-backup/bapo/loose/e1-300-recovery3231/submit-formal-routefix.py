"""New head-only formal controller: scope imports to the fixed new source per call."""
import pathlib,json,hashlib,subprocess,fcntl,shlex,sys,os
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-recovery-20261005')
BASE=ROOT/'scheduler-head';D=ROOT/'scheduler-formal-routefix'
SOURCE='7de6494bff08f35ffeefbf23ed9a9e1b73e5612e'
assert BASE==BASE.resolve()and D==D.resolve()and not BASE.is_symlink()and not D.is_symlink()
lock=(BASE/'submit-global.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if (D/'formal300-job-id.txt').exists():
 job=(D/'formal300-job-id.txt').read_text().strip();assert job.isascii()and job.isdigit()and int(job)>0;print(job);sys.exit(0)
assert not (D/'formal300-submit-intent.txt').exists(),'Unknown outcome; reconcile this intent, never resubmit'
assert not (BASE/'formal300-job-id.txt').exists()and not (BASE/'formal300-submit-intent.txt').exists(),'Original formal outcome exists; reconcile'
m=json.loads((D/'approved-formal-routefix-hashes.json').read_text())
for p,key in [(D/'submit-formal-routefix.py','controller'),(BASE/'formal300.sbatch','sbatch'),(D/'FORMAL-ROUTEFIX-INDEPENDENT-REVIEW.md','independent_review')]:assert hashlib.sha256(p.read_bytes()).hexdigest()==m[key]
worker_prefix='cd '+shlex.quote(str(ROOT/'repo'))+' && env PYTHONPATH='+shlex.quote(str(ROOT/'repo'))+' '+shlex.quote(str(ROOT/'env/bin/python'))+' -B '
def worker(command):
 subprocess.run(['env','-u','LD_LIBRARY_PATH','ssh','-n','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','h100-5',worker_prefix+command],check=True)
origin="import pathlib,json,hashlib; import search_r1.baselines.recovery as r; p=pathlib.Path(r.__file__).resolve(); assert str(p)=="+repr(str(ROOT/'repo/search_r1/baselines/recovery.py'))+"; receipt=pathlib.Path("+repr(str(ROOT/'logs/hardware-acceptance.json'))+"); assert hashlib.sha256(receipt.read_bytes()).hexdigest()=="+repr(m['hardware_receipt'])+"; print(json.dumps({'status':'ACTUAL_NEW_SOURCE_RECOVERY_IMPORT_OK','path':str(p)}),flush=True)"
worker('-c '+shlex.quote(origin))
worker(shlex.quote(str(ROOT/'tools/verify_launch.py'))+' formal300')
worker(shlex.quote(str(ROOT/'tools/verify_hardware_receipt.py')))
q=subprocess.check_output(['squeue','-h','-w','h100-5','-o','%i|%j'],text=True)
for line in q.splitlines():
 row=line.split('|');assert len(row)==2,'Malformed queue';job,name=(x.strip()for x in row)
 assert job not in ('3231','3237','3239')and 'bapo'not in name.lower(),'BAPO active; no duplicate'
with (D/'formal300-submit-intent.txt').open('x')as f:f.write('formal300-source-routefix\n');f.flush();os.fsync(f.fileno())
worker(shlex.quote(str(ROOT/'tools/submission-worker-intent.py'))+' formal300')
result=subprocess.run(['sbatch','--parsable',str(BASE/'formal300.sbatch')],text=True,capture_output=True)
if result.returncode:raise RuntimeError('Unknown sbatch outcome; preserve intent: '+result.stderr)
job=result.stdout.strip();assert job.isascii()and job.isdigit()and int(job)>0,'Unknown scheduler output; preserve intent'
with (D/'formal300-job-id.txt').open('x')as f:f.write(job+'\n');f.flush();os.fsync(f.fileno())
subprocess.run(['scontrol','show','job',job],check=True)
