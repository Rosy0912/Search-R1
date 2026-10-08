"""One fresh formal run after confirmed accidental cancellation; preserve all old artifacts."""
import pathlib,subprocess,json,hashlib,fcntl,datetime,os,re,sys
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-stabilityfix-r2-20261005')
D=ROOT/'scheduler-retry-after3222'
OLD=3222
SOURCE='39cdcdd1404213b897bb0de99cc52d72707682f0'
def call(args):
 p=subprocess.run(args,text=True,capture_output=True)
 if p.returncode:raise RuntimeError((args,p.returncode,p.stdout,p.stderr))
 return p.stdout
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def worker(script):return call(['env','-u','LD_LIBRARY_PATH','ssh','-n','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','h100-5',script])
def reject_active_bapo():
 rows=call(['squeue','-h','-o','%i|%j']).splitlines()
 for row in rows:
  fields=row.strip().split('|');assert len(fields)==2,row
  job,name=(x.strip() for x in fields)
  assert job!=str(OLD) and not name.lower().startswith('bapo'),row
assert D.is_dir() and not D.is_symlink() and D.resolve()==D
with (ROOT/'scheduler-head/submit-global.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 # Idempotence: a known submission is returned; an uncertain outcome is never retried.
 if (D/'formal300-job-id.txt').exists():
  value=(D/'formal300-job-id.txt').read_text().strip();assert value.isdecimal();print('EXISTING_JOB',value);sys.exit(0)
 assert not (D/'formal300-submit-intent.json').exists(),'Unknown prior outcome: reconcile without submission'
 approval=json.loads((D/'retry-approval.json').read_text())
 assert approval['status']=='PASS_INDEPENDENT_RETRY_REVIEW' and approval['source_commit']==SOURCE and approval['old_job']==OLD
 assert approval['root']==str(ROOT) and approval['retry_dir']==str(D)
 assert approval['script_sha256']==sha(pathlib.Path(__file__))
 report=D/'INDEPENDENT-REVIEW.md';assert sha(report)==approval['report_sha256']
 archived=worker(f'cat {ROOT}/logs/user-stop3222-evidence.json')
 assert hashlib.sha256(archived.rstrip('\n').encode()).hexdigest()==approval['old_stop_record_sha256']
 record=json.loads(archived);before=record['after']['stdout'];assert record['after']['returncode']==0
 current=subprocess.run(['scontrol','show','job',str(OLD)],text=True,capture_output=True)
 if current.returncode==0:assert re.search(r'\bJobId=3222\b',current.stdout) and re.search(r'\bJobState=CANCELLED(?:\s|$)',current.stdout)
 else:assert current.returncode==1 and current.stdout=='' and current.stderr=='slurm_load_jobs error: Invalid job id specified\n'
 reject_active_bapo()
 assert re.search(r'JobId=3222\b',before) and 'JobState=CANCELLED' in before and 'Restarts=0' in before
 assert f'WorkDir={ROOT}\n' in before and 'JobName=bapo-instruct300-formal300' in before
 assert 'NodeList=h100-5\n' in before
 m=json.loads((ROOT/'scheduler-head/approved-launch-hashes.json').read_text())
 original=ROOT/'scheduler-head/formal300.sbatch'
 assert not original.is_symlink() and sha(original)==m['formal300.sbatch']==approval['original_sbatch_sha256']
 # Original launch/source/Linux/asset and exact same-root hardware evidence gates remain mandatory.
 print(worker(f'{ROOT}/env/bin/python {ROOT}/tools/verify_launch.py formal300'),flush=True)
 print(worker(f'{ROOT}/env/bin/python {ROOT}/tools/verify_hardware_receipt.py'),flush=True)
 # Read /proc only; reject remaining old marked processes, preserve all foreign processes.
 probe="import pathlib,json; marker=b'BAPO_RUN_ID="+str(ROOT)+"/runs/bapo-instruct300-formal300-3222'; found=[]\nfor p in pathlib.Path('/proc').iterdir():\n if not p.name.isdigit(): continue\n try:\n  if marker in (p/'environ').read_bytes().split(b'\\0'): found.append(int(p.name))\n except (FileNotFoundError,ProcessLookupError): pass\nassert not found,found; print('OLD_OWNED_PROCESSES_EMPTY')"
 import shlex
 print(worker(f'{ROOT}/env/bin/python -c '+shlex.quote(probe)),flush=True)
 reject_active_bapo()
 (D/'previous-job-terminal.txt').write_text(before)
 intent={'old_job':OLD,'source_commit':SOURCE,'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'starts_from':'official fresh Qwen2.5-3B-Instruct','approval_sha256':sha(D/'retry-approval.json')}
 with (D/'formal300-submit-intent.json').open('x') as f:json.dump(intent,f);f.flush();os.fsync(f.fileno())
 worker_intent=ROOT/'logs/submission-intent-formal300-retry-after3222.json'
 payload=json.dumps(intent)
 print(worker(f'{ROOT}/env/bin/python -c '+shlex.quote("import pathlib,os; p=pathlib.Path("+repr(str(worker_intent))+"); f=p.open('x'); f.write("+repr(payload)+"); f.flush(); os.fsync(f.fileno()); f.close(); print('FRESH_DURABLE_WORKER_INTENT')")),flush=True)
 # Submit unchanged, approved sbatch; stage creates a brand-new job-ID-specific run and official initial model.
 value=call(['sbatch','--parsable',str(original)]).strip();assert value.isdecimal(),value
 with (D/'formal300-job-id.txt').open('x') as f:f.write(value+'\n');f.flush();os.fsync(f.fileno())
 print('NEW_JOB_ID='+value,flush=True);print(call(['scontrol','show','job',value]),flush=True)
