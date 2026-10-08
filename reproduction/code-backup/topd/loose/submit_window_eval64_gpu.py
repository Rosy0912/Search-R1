"""One reviewed real64 submit; uncertain submission is never blindly retried."""
import hashlib,json,pathlib,subprocess,sys
review=pathlib.Path(sys.argv[1]);review_sha=hashlib.sha256(review.read_bytes()).hexdigest()
approved=json.loads(review.read_text());assert approved['status']=='PASS' and approved['actual_native_dual_owner_coverage_passed'] and approved['native_job_id']==3062
remote=r'''python3 - <<'INNER'
import fcntl,hashlib,json,os,pathlib,subprocess,time
r=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd');tools=r/'tools/qwen25-32b-window-safety-candidate-v1'
entry=r/'tools/window-eval64.sbatch';binding=r/'logs/qwen25-32b-terminal-fixed413-binding.json'
assert hashlib.sha256(entry.read_bytes()).hexdigest()=='532364049f039092ccdb6c066d30a74fd1d81664cf447e447b24bcb6461b9201'
assert hashlib.sha256(binding.read_bytes()).hexdigest()=='7ded91b8c81a18bd56d6123b1fb5c1dc6426368766a34b2e29383388fa1304a2'
for n,h in json.loads((tools/'files.json').read_text()).items():assert hashlib.sha256((tools/n).read_bytes()).hexdigest()==h
cpu=r/'logs/topd-window-safety-cpu-3059.log';b=cpu.read_bytes();text=b.decode()
assert '73 passed' in text,text
old=r/'logs/qwen25-32b-terminal-fixed413-eval64-3047';failed=json.loads((old/'supervisor-result.json').read_text())
assert failed['status']=='FAIL_NO_TRAINING' and not failed['cleanup']['remaining'] and failed['cleanup']['actual_cgroup_empty'] and not failed['foreign_baseline_missing']
pids={int(s['pid']) for s in failed['cleanup']['signals']}
assert all(not pathlib.Path('/proc',str(p)).exists() for p in pids),pids
cg=pathlib.Path('/sys/fs/cgroup/freezer'+failed['cgroup'])
if cg.exists():assert all(not p.read_text().strip() for p in cg.rglob('cgroup.procs'))
old_queue=subprocess.run(['squeue','-h','-j','3047'],capture_output=True,text=True)
assert not old_queue.stdout.strip()
if old_queue.returncode:assert 'Invalid job id' in old_queue.stderr
node_queue=subprocess.check_output(['squeue','-h','-w','h100-11','-o','%i|%j'],text=True)
assert not any(row.split('|',1)[0]=='3047' for row in node_queue.splitlines())
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True)
assert len(gpu.splitlines())==8
for row in gpu.splitlines():
 uid,used,total,sm=[x.strip() for x in row.split(',')];assert float(used)/float(total)<=.15 and float(sm)<=15
burst_record=json.loads((r/'logs/topd-window-dual-telemetry-burst-submission.json').read_text())
assert burst_record['status']=='SUBMITTED'
burst=r/'logs'/('qwen25-32b-window-dual-burst-'+str(burst_record['job_id']))
proof=json.loads((burst/'result.json').read_text())
assert proof['status']=='PASS_TELEMETRY_ONLY_NO_MODEL_NO_TRAINING_NOT_TEACHER_READY' and proof['baseline_preserved'] and proof['cleanup_record']['actual_cgroup_empty']
assert proof['actual_all8_high_sm_observed'] and proof['actual_high_unequal_dual_positive_windows'] and not proof['parent_cuda_initialized']
assert len(proof['ready'])==2 and all(not pathlib.Path('/proc',str(row['pid'])).exists() for row in proof['ready'])
assert proof['gpu_owner_process_count']==2 and not proof['cleanup_record']['remaining']
assert len(proof['phases'])==3 and all(len(phase['released'])==2 for phase in proof['phases'])
assert sum(len(row['devices']) for phase in proof['phases'] for row in phase['released'])==48
for phase in proof['phases']:
 for row in phase['released']:
  assert not row['errors'] and all(d['allocated']==d['reserved']==0 for d in row['devices'])
assert any(w['phase']==3 for w in proof['actual_high_unequal_dual_positive_windows'])
independent=r/'logs/window-telemetry-native3062-review.json'
assert hashlib.sha256(independent.read_bytes()).hexdigest()=='REVIEW_SHA'
approval=json.loads(independent.read_text())
assert approval['status']=='PASS' and approval['actual_native_dual_owner_coverage_passed'] is True and approval['native_job_id']==burst_record['job_id']==3062
assert approval['source_sha256']=='3316d920f76256c5b45c5bf5e3c16c5c89a26a95cdaf821eb12ab612dc7a7384'
assert approval['adapter_sha256']=='a9845f16e26e226d367541f617ba2ac633fb3c8e955fc0818ff49cc6324b6e2e'
assert approval['result_sha256']==hashlib.sha256((burst/'result.json').read_bytes()).hexdigest()
assert approval['guard_sha256']==hashlib.sha256((burst/'guard.jsonl').read_bytes()).hexdigest()
for n,h in json.loads((r/'tools/qwen25-32b-window-dual-burst-probe-v1/files.json').read_text()).items():assert hashlib.sha256((r/'tools/qwen25-32b-window-dual-burst-probe-v1'/n).read_bytes()).hexdigest()==h
record=r/'logs/qwen25-32b-window-eval64-submission.json'
def persist(obj):
 temp=record.with_suffix('.tmp')
 with temp.open('w') as f:json.dump(obj,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(temp,record)
 fd=os.open(str(record.parent),os.O_RDONLY);os.fsync(fd);os.close(fd)
with (r/'logs/qwen25-32b-window-eval64-submission.lock').open('a') as lock:
 fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
 if record.exists():
  d=json.loads(record.read_text());assert d['status']=='SUBMITTED',d
 else:
  q=subprocess.check_output(['squeue','-h','-w','h100-11','-o','%i|%j|%T'],text=True)
  assert not any(x.split('|')[1]=='qwen25-32b-window-eval64' for x in q.splitlines()),q
  d=dict(status='UNKNOWN_BEFORE_SUBMISSION',entry=str(entry),entry_sha256=hashlib.sha256(entry.read_bytes()).hexdigest(),binding_sha256=hashlib.sha256(binding.read_bytes()).hexdigest(),cpu_job=3059,cpu_log_sha256=hashlib.sha256(b).hexdigest(),independent_source_review_sha256='REVIEW_SHA',epoch=time.time(),prior_failed_job=3047,prior_owned_pids_actual_absent=sorted(pids),pre_submit_gpu=gpu,protocol='ORIGINAL_SAME64_GREEDY_LOCAL_GPU_SEARCH_NO_TRAINING',queue_before=q)
  persist(d)
  res=subprocess.run(['sbatch','--parsable',str(entry),str(binding)],capture_output=True,text=True)
  d.update(sbatch_returncode=res.returncode,sbatch_stdout=res.stdout,sbatch_stderr=res.stderr)
  persist(d)
  assert res.returncode==0,d
  jid=res.stdout.strip().split(';')[0];assert jid.isdigit(),d
  d.update(status='SUBMITTED',job_id=int(jid));persist(d)
print(json.dumps(d))
INNER'''.replace('REVIEW_SHA',review_sha)
p=subprocess.run(['python3','work/remote_topd.py'],input=remote,text=True,capture_output=True)
print(p.stdout,end='');print(p.stderr,end='')
if not p.returncode:pathlib.Path('outputs/qwen25-32b-window-eval64-submission.json').write_text(p.stdout)
raise SystemExit(p.returncode)
