"""One reviewed real64 submit; uncertain submission is never blindly retried."""
import hashlib,json,pathlib,subprocess,sys
review=pathlib.Path(sys.argv[1]);review_sha=hashlib.sha256(review.read_bytes()).hexdigest()
assert review_sha=='8e9404fb9086b3b8e712808a92b52b617e2e61e59775509f55ee19a277ffc17b'
remote=r'''python3 - <<'INNER'
import fcntl,hashlib,json,os,pathlib,subprocess,time
r=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd');tools=r/'tools/qwen25-32b-window-dual-burst-probe-v1'
entry=tools/'probe.sbatch';binding=r/'logs/qwen25-32b-terminal-fixed413-binding.json'
assert hashlib.sha256(entry.read_bytes()).hexdigest()=='f526da82548b6764c1d25f1feff32421c716ca82838c44123dec66e06d2387d7'
assert hashlib.sha256(binding.read_bytes()).hexdigest()=='7ded91b8c81a18bd56d6123b1fb5c1dc6426368766a34b2e29383388fa1304a2'
for n,h in json.loads((tools/'files.json').read_text()).items():assert hashlib.sha256((tools/n).read_bytes()).hexdigest()==h
cpu_record=json.loads((r/'logs/topd-window-dual-burst-cpu-submission.json').read_text());assert cpu_record['status']=='SUBMITTED'
cpu=r/'logs'/('topd-window-dual-burst-cpu-'+str(cpu_record['job_id'])+'.log');b=cpu.read_bytes();text=b.decode()
assert '94 passed' in text,text
old=r/'logs/qwen25-32b-v8-dual-burst-3056';failed=json.loads((old/'result.json').read_text())
assert failed['status']=='FAIL_TELEMETRY_ONLY_NO_MODEL_NO_TRAINING' and not failed['cleanup_record']['remaining'] and failed['cleanup_record']['actual_cgroup_empty'] and failed['baseline_preserved']
pids={int(x['pid']) for x in failed['owner_identities']}
assert all(not pathlib.Path('/proc',str(p)).exists() for p in pids),pids
cg=pathlib.Path('/sys/fs/cgroup/freezer/slurm_h100-11/uid_0/job_3056')
if cg.exists():assert all(not p.read_text().strip() for p in cg.rglob('cgroup.procs'))
old_queue=subprocess.run(['squeue','-h','-j','3056'],capture_output=True,text=True)
assert not old_queue.stdout.strip()
if old_queue.returncode:assert 'Invalid job id' in old_queue.stderr
node_queue=subprocess.check_output(['squeue','-h','-w','h100-11','-o','%i|%j'],text=True)
assert not any(row.split('|',1)[0]=='3056' for row in node_queue.splitlines())
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True)
assert len(gpu.splitlines())==8
for row in gpu.splitlines():
 uid,used,total,sm=[x.strip() for x in row.split(',')];assert float(used)/float(total)<=.15 and float(sm)<=15
record=r/'logs/topd-window-dual-telemetry-burst-submission.json'
def persist(obj):
 temp=record.with_suffix('.tmp')
 with temp.open('w') as f:json.dump(obj,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(temp,record)
 fd=os.open(str(record.parent),os.O_RDONLY);os.fsync(fd);os.close(fd)
with (r/'logs/topd-window-dual-telemetry-burst-submission.lock').open('a') as lock:
 fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
 if record.exists():
  d=json.loads(record.read_text());assert d['status']=='SUBMITTED',d
 else:
  q=subprocess.check_output(['squeue','-h','-w','h100-11','-o','%i|%j|%T'],text=True)
  assert not any(x.split('|')[1]=='topd-window-dual-telemetry-burst' for x in q.splitlines()),q
  d=dict(status='UNKNOWN_BEFORE_SUBMISSION',entry=str(entry),entry_sha256=hashlib.sha256(entry.read_bytes()).hexdigest(),binding_sha256=hashlib.sha256(binding.read_bytes()).hexdigest(),cpu_job=cpu_record['job_id'],cpu_log_sha256=hashlib.sha256(b).hexdigest(),independent_source_review_sha256='REVIEW_SHA',epoch=time.time(),prior_failed_job=3056,prior_owned_pids_actual_absent=sorted(pids),pre_submit_gpu=gpu,protocol='ACTUAL_SHORT_GPU_TELEMETRY_ONLY_NO_MODEL_NO_QA_NO_TRAINING',queue_before=q)
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
if not p.returncode:pathlib.Path('outputs/topd-window-dual-telemetry-burst-submission.json').write_text(p.stdout)
raise SystemExit(p.returncode)
