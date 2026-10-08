"""Perform only exact independently approved owned prepare/readonly observations."""
import base64
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, '/Users/shaoshitong/Documents/remote_codex_control')
import control

base = Path.cwd()
root = '/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct200-safety-20261006'
state = root + '/logs/fresh200-live200-submit-v8'
controls = root + '/tools/fresh200-deployment-controls-v8'
main = dict(path=root + '/logs/FRESH200-LIVE200-INPUTS-v8-PREPHYSICAL-v1.json', bytes=20576810,
    sha256='a6d3e8073ec33eaa5e26faf873aff40213dff7e537bcf1b26e43fb8cda1a77cd')
whole_path = base / 'outputs/gigpo-e1-instruct200/FRESH200-V8-WHOLE-PREPHYSICAL-INDEPENDENT-REVIEW-v1.json'
whole_record = dict(path=root + '/logs/' + whole_path.name, bytes=103552,
    sha256='056ba6554c716a58400f90ac17c80baecdf2b35a64784993d5287a0e44b716ed')
whole_raw = whole_path.read_bytes()
assert len(whole_raw) == whole_record['bytes'] and hashlib.sha256(whole_raw).hexdigest() == whole_record['sha256']
whole = json.loads(whole_raw)
assert whole['status'] == 'PASS_STATIC_PREPHYSICAL_SCOPE'
assert whole['scope'] == 'FRESH200_V8_COMPLETE_CANDIDATE_PREPARE_AND_READONLY_PHYSICAL_ONLY'
assert whole['blocking_findings'] == []
report_path = base / 'outputs/gigpo-e1-instruct200/FRESH200-V8-PREPHYSICAL-INVOCATION-POLICY-SUPPLEMENT-v1.json'
report_raw = report_path.read_bytes()
report_sha = hashlib.sha256(report_raw).hexdigest()
report = json.loads(report_raw)
assert report['status'] == 'PASS_STATIC_PREPHYSICAL_SCOPE'
assert report['scope'] == 'FRESH200_V8_COMPLETE_CANDIDATE_PREPARE_AND_READONLY_PHYSICAL_ONLY'
assert report['blocking_findings'] == []
assert report['original_whole_review'] == whole_record
policy = report['approved_prephysical_policy']
expected = dict(schema='EXACT_FRESH200_V8_PREPHYSICAL_METADATA_INVOCATION_V1', main_input=main,
    original_review_path=root + '/logs/' + report_path.name, output_state=state, single_execution_only=True,
    prepare_allowed=True, read_only_physical_allowed=True, mkdir_state_allowed=True,
    GPU_allowed=False, READY_allowed=False, Slurm_submission_allowed=False, CPU_test_execution_allowed=False,
    foreign_process_actions_allowed=False,
    execution_environment_additions=dict(PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES=''),
    prepare_command=['/usr/bin/python3', controls + '/prepare_candidate_v8.py', main['path'], main['sha256'], str(main['bytes'])],
    physical_command_template=['/usr/bin/python3', controls + '/verify_physical_v8.py', state + '/CANDIDATE.json',
        '<ACTUAL_PREPARED_CANDIDATE_SHA>', '<ACTUAL_PREPARED_CANDIDATE_BYTES>'])
assert policy == expected, 'Original prephysical approval must match every exact argument and scope'
manifest = json.loads((base / 'outputs/gigpo-e1-policy-safety/FRESH200-DEPLOYMENT-CONTROLS-STATIC-MANIFEST-v8.json').read_bytes())
payload = dict(policy=policy, review_sha=report_sha, review_b64=base64.b64encode(report_raw).decode(),
    whole_record=whole_record, whole_b64=base64.b64encode(whole_raw).decode(),
    controls_files=manifest['controls_target_files'])
remote_code = '''import base64,hashlib,json,os,subprocess,sys
from pathlib import Path
p=json.load(sys.stdin);policy=p['policy'];state=Path(policy['output_state']);review_path=Path(policy['original_review_path'])
whole=p['whole_record'];whole_path=Path(whole['path']);whole_raw=base64.b64decode(p['whole_b64'])
assert len(whole_raw)==whole['bytes'] and hashlib.sha256(whole_raw).hexdigest()==whole['sha256']
assert whole_path.parent.resolve()==whole_path.parent and not whole_path.is_symlink()
if whole_path.exists():assert whole_path.read_bytes()==whole_raw
else:
 with whole_path.open('xb') as f:f.write(whole_raw);f.flush();os.fsync(f.fileno())
raw=base64.b64decode(p['review_b64']);assert hashlib.sha256(raw).hexdigest()==p['review_sha']
assert review_path.parent.resolve()==review_path.parent and not review_path.is_symlink()
if review_path.exists():assert review_path.read_bytes()==raw
else:
 with review_path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
def checked(row):
 path=Path(row['path']);assert path.resolve(strict=True)==path and not path.is_symlink();raw=path.read_bytes()
 assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'];return raw
checked(policy['main_input'])
for row in p['controls_files']:checked(row)
assert state.parent.resolve()==state.parent and not state.is_symlink()
if state.exists():assert state.is_dir() and not any(state.iterdir()),'Existing owned state: inspect originals, never replay'
else:state.mkdir()
env=dict(os.environ);env.update(policy['execution_environment_additions']);steps=[]
def execute(cmd):
 result=subprocess.run(cmd,capture_output=True,text=True,env=env)
 step=dict(command=cmd,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr);steps.append(step)
 if result.returncode:
  print(json.dumps(dict(status='PREPHYSICAL_METADATA_FAILED_ORIGINAL_STEPS_PRESERVED',steps=steps,GPU_executed=False,READY=False,submitted=False)));raise SystemExit(result.returncode)
 return json.loads(result.stdout)
prepared=execute(policy['prepare_command']);candidate=prepared['candidate']
assert candidate['path']==str(state/'CANDIDATE.json') and prepared['READY'] is False and prepared['submitted'] is False
checked(candidate)
command=list(policy['physical_command_template']);command[-2]=candidate['sha256'];command[-1]=str(candidate['bytes'])
observed=execute(command);physical=observed['physical']
assert physical['path']==str(state/'PHYSICAL.json') and observed['READY'] is False and observed['submitted'] is False
checked(physical);checked(candidate);checked(policy['main_input'])
assert not any((state/name).exists() for name in ('READY.json','BINDING.json','SUBMIT-INTENT.json','SUBMIT-RECEIPT.json'))
print(json.dumps(dict(status='PREPARED_AND_READONLY_PHYSICAL_OBSERVED_NOT_READY_NOT_GPU',steps=steps,candidate_record=candidate,physical_record=physical,original_prephysical_review_sha256=p['review_sha'],GPU_executed=False,READY=False,submitted=False,formal_iterations=0),indent=2))
'''
inner = '/usr/bin/python3 -c ' + shlex.quote(remote_code)
mid = 'env -u LD_LIBRARY_PATH ssh -T -o ConnectTimeout=15 h100-8 ' + shlex.quote(inner)
route = control.ssh('mgmt-host') + ['env -u LD_LIBRARY_PATH ssh -T -o ConnectTimeout=15 h100-16 ' + shlex.quote(mid)]
result = subprocess.run(route, input=json.dumps(payload).encode(), capture_output=True, timeout=600)
out = base / 'outputs/gigpo-e1-instruct200/FRESH200-V8-ACTUAL-PREPHYSICAL-ONCE-INVOKE-v1.json'
with out.open('xb') as f:
    f.write(result.stdout if result.stdout else json.dumps(dict(route_returncode=result.returncode, route_stderr=result.stderr.decode(errors='replace'))).encode())
    f.write(b'\n')
if result.returncode:
    raise RuntimeError('Preserved actual prephysical failure: ' + str(out) + ' ' + result.stderr.decode(errors='replace'))
record = json.loads(result.stdout)
print(json.dumps(dict(status=record['status'], candidate_record=record['candidate_record'], physical_record=record['physical_record'],
    original_invoke=str(out), GPU_executed=False, READY=False, submitted=False)))
