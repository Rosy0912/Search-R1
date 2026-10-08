from pathlib import Path
import subprocess,sys,json,re,hashlib,datetime,time
ROOT=Path(__file__).resolve().parents[2]
REPO=ROOT/'work/tree-behavior-mu-ppo'
OUT=Path(__file__).resolve().parent
expected_head='e08e1896ce13655aa3f2b934212d2caea25e949c'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==expected_head
assert subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True)==''
changed=subprocess.check_output(['git','diff','--name-only','927ab58..HEAD'],cwd=REPO,text=True).splitlines()
files={p:hashlib.sha256((REPO/p).read_bytes()).hexdigest() for p in changed}
cases=[('config',8,['test_tree_behavior_mu_config_cpu.BehaviorMuConfigTests','test_tree_behavior_mu_config_cpu.NativeProcessorCaptureTests']),('data',11,['test_tree_behavior_mu_data_cpu.BehaviorMuDataTests']),('actor',21,['test_tree_behavior_mu_actor_cpu.Tests']),('calibration_boundary',5,['test_behavior_mu_calibration_boundary_cpu.CalibrationBoundaryTests']),('calibration_snapshot',4,['test_behavior_mu_calibration_snapshot_cpu.CalibrationSnapshotTests']),('facade_affected',1,['test_tree_raw_tokens_cpu.RawTokenTests.test_actual_facade_keeps_lengths_before_padding_and_legacy_return_arity'])]
records=[]
for label,n,args in cases:
    cmd=[sys.executable,'-B','-m','unittest','-v']+args
    start=time.monotonic();result=subprocess.run(cmd,cwd=REPO,capture_output=True)
    elapsed=time.monotonic()-start
    for suffix,b in [('output.txt',result.stdout),('error.txt',result.stderr)]:
        with (OUT/(label+'.'+suffix)).open('xb') as f:f.write(b)
    log=result.stderr.decode(errors='replace')
    counts=re.findall(r'Ran (\d+) tests? in ',log)
    actual=int(counts[-1]) if counts else None
    methods=re.findall(r'^(test_\S+) \(([^)]+)\) \.\.\. ',log,re.M)
    entry=dict(label=label,command=cmd,expected_unique_tests=n,actual_tests=actual,test_ids=[c+'.'+m for m,c in methods],returncode=result.returncode,wall_seconds=elapsed,stdout_sha256=hashlib.sha256(result.stdout).hexdigest(),stderr_sha256=hashlib.sha256(result.stderr).hexdigest(),passed=result.returncode==0 and actual==n and log.rstrip().endswith('OK'))
    records.append(entry)
    print(json.dumps(entry),flush=True)
    if not entry['passed']:break
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==expected_head
assert subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True)==''
assert files=={p:hashlib.sha256((REPO/p).read_bytes()).hexdigest() for p in changed}
report=dict(status='INDEPENDENT_FROZEN_CPU_RUN',created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),head=expected_head,base='927ab58fdd8240d61aa52745700b2469f5406027',frozen_files_sha256=files,results=records,actual_tests=sum(r['actual_tests'] or 0 for r in records),all_passed=len(records)==6 and all(r['passed'] for r in records),real_hf_fsdp_ray_gpu_acceptance=False)
with (OUT/'CPU_RUN_RESULT.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
sys.exit(0 if report['all_passed'] else 1)
