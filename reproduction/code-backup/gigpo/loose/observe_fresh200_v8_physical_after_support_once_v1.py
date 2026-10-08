"""Observe the already prepared candidate once after a proven metadata copy repair."""
import base64
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

if sys.flags.optimize or not sys.flags.no_site or not sys.flags.ignore_environment:
    raise RuntimeError('Use -E -B -S with active assertions')
sys.path.insert(0, '/Users/shaoshitong/Documents/remote_codex_control')
import control

base = Path.cwd()
root = '/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct200-safety-20261006'
state = root + '/logs/fresh200-live200-submit-v8'
candidate = dict(path=state + '/CANDIDATE.json', bytes=20576810,
    sha256='a6d3e8073ec33eaa5e26faf873aff40213dff7e537bcf1b26e43fb8cda1a77cd')
interpreter = '/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004/env/bin/python'
environment = dict(PATH='/usr/bin:/bin', LANG='C', LC_ALL='C', PYTHONDONTWRITEBYTECODE='1',
    PYTHONNOUSERSITE='1', GIT_OPTIONAL_LOCKS='0', CUDA_VISIBLE_DEVICES='')
command = [interpreter, '-B', '-S', root + '/tools/fresh200-deployment-controls-v8/verify_physical_v8.py',
    candidate['path'], candidate['sha256'], str(candidate['bytes'])]
review_path = base / 'outputs/gigpo-e1-instruct200/FRESH200-V8-KNOWN-MISSING-SUPPORT-RECOVERY-REVIEW-v1.json'
review_raw = review_path.read_bytes()
review = json.loads(review_raw)
assert review['status'] == 'PASS'
assert review['scope'] == 'FRESH200_V8_KNOWN_METADATA_INSTALL_READONLY_PHYSICAL_RECOVERY_ONLY'
assert review['independent_review'] is True and review['blocking_findings'] == []
assert review['candidate_record'] == candidate and review['physical_command'] == command
assert review['execution_environment'] == environment and review['prepare_again_allowed'] is False
assert review['GPU_allowed'] is False and review['READY_allowed'] is False and review['Slurm_submission_allowed'] is False
support = json.loads((base / 'outputs/gigpo-e1-instruct200/FRESH200-V8-MISSING-SUPPORT-EXACT-STAGING-v1.json').read_bytes())['records']
payload = dict(review_b64=base64.b64encode(review_raw).decode(), review_sha=hashlib.sha256(review_raw).hexdigest(),
    review_path=root + '/logs/' + review_path.name, candidate=candidate, support=support,
    environment=environment, command=command)
code = '''import base64,hashlib,json,os,subprocess,sys
from pathlib import Path
if sys.flags.optimize or not sys.flags.no_site or not sys.flags.ignore_environment:raise RuntimeError('Isolated startup required')
p=json.load(sys.stdin);candidate=p['candidate'];state=Path(candidate['path']).parent
assert state.resolve()==state and sorted(x.name for x in state.iterdir())==['CANDIDATE.json']
def check(row):
 path=Path(row['path']);assert path.resolve(strict=True)==path and not path.is_symlink();raw=path.read_bytes()
 assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'];return raw
check(candidate)
for row in p['support']:check(row)
raw=base64.b64decode(p['review_b64']);assert hashlib.sha256(raw).hexdigest()==p['review_sha'];path=Path(p['review_path'])
assert path.parent.resolve()==path.parent and not path.is_symlink()
if path.exists():assert path.read_bytes()==raw
else:
 with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
done=subprocess.run(p['command'],capture_output=True,text=True,env=p['environment'])
out=dict(status='PHYSICAL_AFTER_EXACT_SUPPORT_FAILED' if done.returncode else 'PHYSICAL_AFTER_EXACT_SUPPORT_OBSERVED_NOT_READY',candidate_record=candidate,command=p['command'],returncode=done.returncode,stdout=done.stdout,stderr=done.stderr,review_sha256=p['review_sha'],GPU_executed=False,READY=False,submitted=False)
if not done.returncode:
 observed=json.loads(done.stdout);physical=observed['physical'];assert physical['path']==str(state/'PHYSICAL.json') and observed['READY'] is False and observed['submitted'] is False
 check(physical);check(candidate);out['physical_record']=physical
 assert sorted(x.name for x in state.iterdir())==['CANDIDATE.json','PHYSICAL.json']
print(json.dumps(out));raise SystemExit(done.returncode)
'''
inner = 'env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C ' + shlex.quote(interpreter) + ' -E -B -S -c ' + shlex.quote(code)
mid = 'env -u LD_LIBRARY_PATH ssh -T -o ConnectTimeout=15 h100-8 ' + shlex.quote(inner)
route = control.ssh('mgmt-host') + ['env -u LD_LIBRARY_PATH ssh -T -o ConnectTimeout=15 h100-16 ' + shlex.quote(mid)]
done = subprocess.run(route, input=json.dumps(payload).encode(), capture_output=True, timeout=600)
out_path = base / 'outputs/gigpo-e1-instruct200/FRESH200-V8-ACTUAL-PHYSICAL-AFTER-SUPPORT-INVOKE-v1.json'
with out_path.open('xb') as f:
    f.write(done.stdout if done.stdout else json.dumps(dict(route_returncode=done.returncode, stderr=done.stderr.decode(errors='replace'))).encode())
    f.write(b'\n')
if done.returncode:
    raise RuntimeError('Original known-repair observation failed; inspect, never replay: ' + str(out_path))
record = json.loads(done.stdout)
print(json.dumps({k:record[k] for k in ('status','candidate_record','physical_record','returncode','GPU_executed','READY','submitted')}))
