"""One reviewed real64 submit; uncertain submission is never blindly retried."""
import hashlib,json,pathlib,subprocess,sys
review=pathlib.Path(sys.argv[1]);review_sha=hashlib.sha256(review.read_bytes()).hexdigest()
remote=r'''python3 - <<'INNER'
import fcntl,hashlib,json,os,pathlib,subprocess,time
r=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd');tools=r/'tools/qwen25-32b-v6-burst-probe'
entry=tools/'probe.sbatch';binding=r/'logs/qwen25-32b-terminal-fixed413-binding.json'
assert hashlib.sha256(entry.read_bytes()).hexdigest()=='f824fcb6ae44aa621668dfacee04347212f85bdad4c7e16c4c344d965717eca8'
assert hashlib.sha256(binding.read_bytes()).hexdigest()=='7ded91b8c81a18bd56d6123b1fb5c1dc6426368766a34b2e29383388fa1304a2'
for n,h in json.loads((tools/'files.json').read_text()).items():assert hashlib.sha256((tools/n).read_bytes()).hexdigest()==h
cpu_record=json.loads((r/'logs/topd-v7-burst-cpu-submission.json').read_text());assert cpu_record['status']=='SUBMITTED'
cpu=r/'logs'/('topd-v7-burst-cpu-'+str(cpu_record['job_id'])+'.log');b=cpu.read_bytes();text=b.decode()
assert '51 passed' in text,text
old=r/'logs/qwen25-32b-terminal-fixed413-eval64-3036';failed=json.loads((old/'supervisor-result.json').read_text())
assert failed['status']=='FAIL_NO_TRAINING' and not failed['cleanup']['remaining'] and failed['cleanup']['actual_cgroup_empty'] and not failed['foreign_baseline_missing']
pids={int(s['pid']) for s in failed['cleanup']['signals']}
assert all(not pathlib.Path('/proc',str(p)).exists() for p in pids),pids
cg=pathlib.Path('/sys/fs/cgroup/freezer'+failed['cgroup'])
if cg.exists():assert all(not p.read_text().strip() for p in cg.rglob('cgroup.procs'))
old_queue=subprocess.run(['squeue','-h','-j','3036'],capture_output=True,text=True)
assert not old_queue.stdout.strip()
if old_queue.returncode:assert 'Invalid job id' in old_queue.stderr
node_queue=subprocess.check_output(['squeue','-h','-w','h100-11','-o','%i|%j'],text=True)
assert not any(row.split('|',1)[0]=='3036' for row in node_queue.splitlines())
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True)
assert len(gpu.splitlines())==8
for row in gpu.splitlines():
 uid,used,total,sm=[x.strip() for x in row.split(',')];assert float(used)/float(total)<=.15 and float(sm)<=15
record=r/'logs/topd-v7-telemetry-burst-submission.json'
def persist(obj):
 temp=record.with_suffix('.tmp')
 with temp.open('w') as f:json.dump(obj,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(temp,record)
 fd=os.open(str(record.parent),os.O_RDONLY);os.fsync(fd);os.close(fd)
with (r/'logs/topd-v7-telemetry-burst-submission.lock').open('a') as lock:
 fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
 if record.exists():
  d=json.loads(record.read_text());assert d['status']=='SUBMITTED',d
 else:
  q=subprocess.check_output(['squeue','-h','-w','h100-11','-o','%i|%j|%T'],text=True)
  assert not any(x.split('|')[1]=='topd-v7-telemetry-burst' for x in q.splitlines()),q
  d=dict(status='UNKNOWN_BEFORE_SUBMISSION',entry=str(entry),entry_sha256=hashlib.sha256(entry.read_bytes()).hexdigest(),binding_sha256=hashlib.sha256(binding.read_bytes()).hexdigest(),cpu_job=cpu_record['job_id'],cpu_log_sha256=hashlib.sha256(b).hexdigest(),independent_source_review_sha256='REVIEW_SHA',epoch=time.time(),prior_failed_job=3036,prior_owned_pids_actual_absent=sorted(pids),pre_submit_gpu=gpu,protocol='ACTUAL_SHORT_GPU_TELEMETRY_ONLY_NO_MODEL_NO_QA_NO_TRAINING',queue_before=q)
  persist(d)
  res=subprocess.run(['sbatch','--parsable',str(entry)],capture_output=True,text=True)
  d.update(sbatch_returncode=res.returncode,sbatch_stdout=res.stdout,sbatch_stderr=res.stderr)
  persist(d)
  assert res.returncode==0,d
  jid=res.stdout.strip().split(';')[0];assert jid.isdigit(),d
  d.update(status='SUBMITTED',job_id=int(jid));persist(d)
print(json.dumps(d))
INNER'''.replace('REVIEW_SHA',review_sha)
p=subprocess.run(['python3','work/remote_topd.py'],input=remote,text=True,capture_output=True)
print(p.stdout,end='');print(p.stderr,end='')
if not p.returncode:pathlib.Path('outputs/topd-v7-telemetry-burst-submission.json').write_text(p.stdout)
raise SystemExit(p.returncode)
