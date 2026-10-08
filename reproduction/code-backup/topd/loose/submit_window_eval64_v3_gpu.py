"""Root-executed unique full64 V3 submission; no import-time action or automatic retry."""
import argparse,hashlib,json,pathlib,re,subprocess
ROOT = pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd')

def checked_review(path, expected_sha):
    path=pathlib.Path(path)
    assert re.fullmatch(r'[A-Za-z0-9_.-]+',path.name), 'Unsafe review filename'
    assert re.fullmatch(r'[0-9a-f]{64}',expected_sha), 'Exact independent review SHA required'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected_sha, 'Independent receipt bytes differ'
    approved=json.loads(path.read_text())
    assert approved['status']=='PASS' and approved['actual_native_dual_owner_coverage_passed'] is True and approved['native_job_id']==3074
    assert approved['source_sha256']=='9e41464d49054a1765fed29f5839f9fd8c568f0040aedbf6cf36b3c096adb3c6'
    assert approved['adapter_sha256']=='ea5349797d2e314f39256699994b5cadc95bc5f9d826f5d965dd2cf80b42cd10'
    assert approved['scope']=='TELEMETRY_ONLY_NO_MODEL_NO_TRAINING_NOT_TEACHER_APPROVAL'
    assert approved['approval_scope']=='OBSERVED_RETURNED_RECORD_FOREIGN_PID_MAXIMA_ONLY_NOT_CONTINUOUS_SM_NOT_TEACHER_QUALITY'
    assert all(approved[k] is True for k in ('actual_zero_own_remeasurement_coverage_passed','actual_all48_allocator_zero_passed','actual_cgroup_empty','foreign_baseline_preserved'))
    assert approved['source_manifest_sha256']=='349ff8477c2e7fa6417f17eb783dd8a8b4c06c487c1eeedd858cbed6f5ea4e8e'
    assert approved['probe_manifest_sha256']=='7ffaa56d9744085b53c7948d9b6b639e430e81993f1a7d67018fa501d430cc48'
    return path,approved

REMOTE_TEMPLATE = r'''
python3 - <<'INNER'
import fcntl,hashlib,json,os,pathlib,subprocess,time
r=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd');tools=r/'tools/qwen25-32b-window-safety-candidate-v3'
entry=r/'tools/window-eval64-v3.sbatch';binding=r/'logs/qwen25-32b-terminal-fixed413-binding.json'
assert hashlib.sha256(entry.read_bytes()).hexdigest()=='f1f06fde3d3cbdb0a80bf4f24b182205133d98d45ea22898cfe9bcfd076b0c05'
assert hashlib.sha256(binding.read_bytes()).hexdigest()=='7ded91b8c81a18bd56d6123b1fb5c1dc6426368766a34b2e29383388fa1304a2'
assert hashlib.sha256((tools/'runtime_safety.py').read_bytes()).hexdigest()=='9e41464d49054a1765fed29f5839f9fd8c568f0040aedbf6cf36b3c096adb3c6'
assert hashlib.sha256((tools/'startup_guard.py').read_bytes()).hexdigest()=='ea5349797d2e314f39256699994b5cadc95bc5f9d826f5d965dd2cf80b42cd10'
for n,h in json.loads((tools/'files.json').read_text()).items():assert hashlib.sha256((tools/n).read_bytes()).hexdigest()==h
cpu=r/'logs/topd-window-dual-burst-cpu-3073.log';b=cpu.read_bytes();text=b.decode()
assert hashlib.sha256(b).hexdigest()=='dd16dd1e44ff994b5216ad8602173c97bef5f3b3aaec1b68b725411ca5fa77d5'
assert hashlib.sha256((tools/'files.json').read_bytes()).hexdigest()=='349ff8477c2e7fa6417f17eb783dd8a8b4c06c487c1eeedd858cbed6f5ea4e8e'
assert '125 passed' in text,text
cpu_record=json.loads((r/'logs/topd-window-dual-v3-burst-cpu-submission.json').read_text())
assert cpu_record['status']=='SUBMITTED' and cpu_record['job_id']==3073
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
assert len(gpu.splitlines())==8 and len({v.split(',')[0].strip() for v in gpu.splitlines()})==8
for row in gpu.splitlines():
 uid,used,total,sm=[x.strip() for x in row.split(',')];assert 0<=float(used)<=float(total) and float(total)>0 and float(used)/float(total)<=.15 and 0<=float(sm)<=15
burst_record=json.loads((r/'logs/topd-window-dual-v3-telemetry-burst-submission.json').read_text())
assert burst_record['status']=='SUBMITTED' and burst_record['job_id']==3074
burst=r/'logs'/('qwen25-32b-window-dual-burst-'+str(burst_record['job_id']))
proof=json.loads((burst/'result.json').read_text())
assert proof['status']=='PASS_TELEMETRY_ONLY_NO_MODEL_NO_TRAINING_NOT_TEACHER_READY' and proof['baseline_preserved'] and proof['cleanup_record']['actual_cgroup_empty']
assert proof['actual_zero_own_remeasurements']
assert proof['actual_all8_high_sm_observed'] and proof['actual_high_unequal_dual_positive_windows'] and not proof['parent_cuda_initialized']
assert len(proof['ready'])==2 and all(not pathlib.Path('/proc',str(row['pid'])).exists() for row in proof['ready'])
assert proof['gpu_owner_process_count']==2 and not proof['cleanup_record']['remaining']
assert [p['requested_seconds'] for p in proof['phases']]==[16.,.25,2.]
assert len(proof['phases'])==3 and all(len(phase['released'])==2 for phase in proof['phases'])
assert sum(len(row['devices']) for phase in proof['phases'] for row in phase['released'])==48
for phase in proof['phases']:
 for row in phase['released']:
  assert row['duration']==phase['requested_seconds']
  assert not row['errors'] and all(d['allocated']==d['reserved']==0 for d in row['devices'])
assert any(w['phase']==1 for w in proof['actual_high_unequal_dual_positive_windows'])
independent=r/'logs'/'REVIEW_NAME'
assert hashlib.sha256(independent.read_bytes()).hexdigest()=='REVIEW_SHA'
approval=json.loads(independent.read_text())
assert approval['status']=='PASS' and approval['actual_native_dual_owner_coverage_passed'] is True and approval['native_job_id']==burst_record['job_id']==3074
assert approval['source_sha256']=='9e41464d49054a1765fed29f5839f9fd8c568f0040aedbf6cf36b3c096adb3c6'
assert approval['adapter_sha256']=='ea5349797d2e314f39256699994b5cadc95bc5f9d826f5d965dd2cf80b42cd10'
assert approval['result_sha256']==hashlib.sha256((burst/'result.json').read_bytes()).hexdigest()
assert approval['guard_sha256']==hashlib.sha256((burst/'guard.jsonl').read_bytes()).hexdigest()
assert approval['scope']=='TELEMETRY_ONLY_NO_MODEL_NO_TRAINING_NOT_TEACHER_APPROVAL'
assert approval['approval_scope']=='OBSERVED_RETURNED_RECORD_FOREIGN_PID_MAXIMA_ONLY_NOT_CONTINUOUS_SM_NOT_TEACHER_QUALITY'
assert all(approval[k] is True for k in ('actual_zero_own_remeasurement_coverage_passed','actual_all48_allocator_zero_passed','actual_cgroup_empty','foreign_baseline_preserved'))
assert approval['source_manifest_sha256']=='349ff8477c2e7fa6417f17eb783dd8a8b4c06c487c1eeedd858cbed6f5ea4e8e'
assert approval['probe_manifest_sha256']=='7ffaa56d9744085b53c7948d9b6b639e430e81993f1a7d67018fa501d430cc48'
seen=set()
for ref in approval['raw_refs']:
 path=pathlib.Path(ref['path'])
 assert path.is_relative_to(burst) and not path.is_symlink() and path.is_file() and str(path) not in seen
 assert hashlib.sha256(path.read_bytes()).hexdigest()==ref['sha256']
 seen.add(str(path))
required={str(burst/'result.json'),str(burst/'guard.jsonl')}|{str(burst/('owner-'+str(i))/name) for i in range(2) for name in ('ready.json','done-1.json','done-2.json','done-3.json')}
assert required<=seen
review_report=approval['independent_report'];report=pathlib.Path(review_report['path'])
assert report.is_relative_to(r/'logs') and not report.is_symlink() and hashlib.sha256(report.read_bytes()).hexdigest()==review_report['sha256']
assert hashlib.sha256((r/'tools/qwen25-32b-window-dual-burst-probe-v3/files.json').read_bytes()).hexdigest()=='7ffaa56d9744085b53c7948d9b6b639e430e81993f1a7d67018fa501d430cc48'
for n,h in json.loads((r/'tools/qwen25-32b-window-dual-burst-probe-v3/files.json').read_text()).items():assert hashlib.sha256((r/'tools/qwen25-32b-window-dual-burst-probe-v3'/n).read_bytes()).hexdigest()==h
record=r/'logs/qwen25-32b-window-eval64-v3-submission.json'
def persist(obj):
 temp=record.with_suffix('.tmp')
 assert not temp.is_symlink() and not record.is_symlink()
 with temp.open('w') as f:json.dump(obj,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(temp,record)
 fd=os.open(str(record.parent),os.O_RDONLY);os.fsync(fd);os.close(fd)
lockpath=r/'logs/qwen25-32b-window-eval64-v3-submission.lock'
assert not lockpath.is_symlink()
with lockpath.open('a') as lock:
 fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
 if record.exists():
  d=json.loads(record.read_text());assert d['status']=='SUBMITTED' and d['entry_sha256']==hashlib.sha256(entry.read_bytes()).hexdigest() and d['independent_source_review_sha256']=='REVIEW_SHA',d
 else:
  q=subprocess.check_output(['squeue','-h','-w','h100-11','-o','%i|%j|%T'],text=True)
  assert not any(x.split('|')[1]=='qwen25-32b-window-eval64' for x in q.splitlines()),q
  d=dict(status='UNKNOWN_BEFORE_SUBMISSION',entry=str(entry),entry_sha256=hashlib.sha256(entry.read_bytes()).hexdigest(),binding_sha256=hashlib.sha256(binding.read_bytes()).hexdigest(),cpu_job=3073,cpu_log_sha256=hashlib.sha256(b).hexdigest(),independent_source_review_sha256='REVIEW_SHA',epoch=time.time(),prior_failed_job=3047,prior_owned_pids_actual_absent=sorted(pids),pre_submit_gpu=gpu,protocol='ORIGINAL_SAME64_GREEDY_LOCAL_GPU_SEARCH_NO_TRAINING',queue_before=q)
  persist(d)
  res=subprocess.run(['sbatch','--parsable',str(entry),str(binding)],capture_output=True,text=True)
  d.update(sbatch_returncode=res.returncode,sbatch_stdout=res.stdout,sbatch_stderr=res.stderr)
  persist(d)
  assert res.returncode==0,d
  jid=res.stdout.strip().split(';')[0];assert jid.isdigit(),d
  d.update(status='SUBMITTED',job_id=int(jid));persist(d)
print(json.dumps(d))
INNER'''

def build_remote(path, expected_sha):
    path,_=checked_review(path,expected_sha)
    script=REMOTE_TEMPLATE.replace('REVIEW_SHA',expected_sha).replace('REVIEW_NAME',path.name)
    # Refuse malformed generated Python before any remote command or sbatch.
    code=script.split("python3 - <<'INNER'\n",1)[1].rsplit('\nINNER',1)[0]
    compile(code,'full64-v3-submission','exec')
    return script

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('review', help='Actual auditor normalized3074 JSON; copied verbatim into ownROOT/logs')
    parser.add_argument('--review-sha256',required=True)
    parser.add_argument('--prepare-only',action='store_true',help='Validate local receipt and compile; no remote action')
    args=parser.parse_args();remote=build_remote(args.review,args.review_sha256)
    if args.prepare_only:
        print(json.dumps(dict(scope='PREPARED_NOT_SUBMITTED',remote_script_sha256=hashlib.sha256(remote.encode()).hexdigest())))
        return
    result=subprocess.run(['python3','work/remote_topd.py'],input=remote,text=True,capture_output=True)
    print(result.stdout,end='');print(result.stderr,end='')
    if not result.returncode:
        pathlib.Path('outputs/qwen25-32b-window-eval64-v3-submission.json').write_text(result.stdout)
    raise SystemExit(result.returncode)

if __name__=='__main__':main()
