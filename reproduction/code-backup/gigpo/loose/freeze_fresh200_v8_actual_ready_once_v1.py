"""Stage a genuine independent actual-physical review and freeze READY metadata only."""
import base64
import hashlib
import json
import shlex
import subprocess
import sys
import zlib
from pathlib import Path

if sys.flags.optimize or not sys.flags.no_site or not sys.flags.ignore_environment:
    raise RuntimeError('Use -E -B -S with active assertions')
sys.path.insert(0, '/Users/shaoshitong/Documents/remote_codex_control')
import control

base = Path.cwd()
root = '/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct200-safety-20261006'
state = root + '/logs/fresh200-live200-submit-v8'
interpreter = '/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004/env/bin/python'
environment = dict(PATH='/usr/bin:/bin', LANG='C', LC_ALL='C', PYTHONDONTWRITEBYTECODE='1',
    PYTHONNOUSERSITE='1', GIT_OPTIONAL_LOCKS='0', CUDA_VISIBLE_DEVICES='')
pull = json.loads((base / 'outputs/gigpo-e1-instruct200/actual-prephysical-v8-originals-v2/RAW-PULL-RECEIPT.json').read_bytes())
candidate, physical = [row['remote_record'] for row in pull['files']]
assert candidate['path'] == state + '/CANDIDATE.json' and physical['path'] == state + '/PHYSICAL.json'
review_path = base / 'outputs/gigpo-e1-instruct200/FRESH200-V8-ACTUAL-PHYSICAL-DEPLOYMENT-INDEPENDENT-REVIEW-v1.json'
review_raw = review_path.read_bytes()
review = json.loads(review_raw)
assert review['status'] == 'PASS' and review['scope'] == 'GIGPO_POLICY_SAFETY_OFFICIAL_INSTRUCT_FRESH200_LIVE_V1'
assert review['independent_review'] is True and review['blocking_findings'] == []
assert review['candidate_sha256'] == candidate['sha256'] and review['physical_sha256'] == physical['sha256']
assert review['actual_gpu_acceptance'] is False and review['actual_probability_acceptance'] is False
assert review['READY_freeze_allowed'] is True and review['Slurm_submission_allowed_by_this_review'] is False
assert review['freeze_wrapper_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
review_record = dict(path=root + '/logs/' + review_path.name, bytes=len(review_raw), sha256=hashlib.sha256(review_raw).hexdigest())
spec = dict(candidate=candidate, physical=physical, deployment_review=review_record)
spec_raw = (json.dumps(spec, sort_keys=True, indent=2) + '\n').encode()
spec_record = dict(path=root + '/logs/FRESH200-V8-ACTUAL-READY-FREEZE-SPEC-v1.json', bytes=len(spec_raw), sha256=hashlib.sha256(spec_raw).hexdigest())
command = [interpreter, '-B', '-S', root + '/tools/fresh200-deployment-controls-v8/freeze_ready_v8.py', json.dumps(spec_record)]
payload = dict(review=review_record, review_b64=base64.b64encode(review_raw).decode(), spec=spec_record,
    spec_b64=base64.b64encode(spec_raw).decode(), command=command, environment=environment, state=state)
code = '''import base64,hashlib,json,os,subprocess,sys,zlib
from pathlib import Path
if sys.flags.optimize or not sys.flags.no_site or not sys.flags.ignore_environment:raise RuntimeError('Isolated startup required')
p=json.load(sys.stdin);state=Path(p['state']);assert state.resolve()==state and sorted(x.name for x in state.iterdir())==['CANDIDATE.json','PHYSICAL.json']
for name in ('review','spec'):
 row=p[name];path=Path(row['path']);raw=base64.b64decode(p[name+'_b64'])
 assert path.parent.resolve()==path.parent and not path.is_symlink();assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 if path.exists():assert path.read_bytes()==raw
 else:
  with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
done=subprocess.run(p['command'],capture_output=True,text=True,env=p['environment'])
out=dict(status='ACTUAL_READY_FREEZE_FAILED' if done.returncode else 'ACTUAL_READY_METADATA_FROZEN_NOT_SUBMITTED',command=p['command'],returncode=done.returncode,stdout=done.stdout,stderr=done.stderr,GPU_executed=False,submitted=False)
if not done.returncode:
 result=json.loads(done.stdout);row=result['ready'];assert row['path']==str(state/'READY.json') and result['submitted'] is False and result['probability_acceptance'] is False
 path=Path(row['path']);assert path.resolve(strict=True)==path;raw=path.read_bytes();assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 assert sorted(x.name for x in state.iterdir())==['CANDIDATE.json','PHYSICAL.json','READY.json']
 out.update(ready_record=row,ready_original_zlib_b64=base64.b64encode(zlib.compress(raw,6)).decode())
print(json.dumps(out));raise SystemExit(done.returncode)
'''
inner = 'env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C ' + shlex.quote(interpreter) + ' -E -B -S -c ' + shlex.quote(code)
mid = 'env -u LD_LIBRARY_PATH ssh -T -o ConnectTimeout=15 h100-8 ' + shlex.quote(inner)
route = control.ssh('mgmt-host') + ['env -u LD_LIBRARY_PATH ssh -T -o ConnectTimeout=15 h100-16 ' + shlex.quote(mid)]
done = subprocess.run(route, input=json.dumps(payload).encode(), capture_output=True, timeout=600)
out_path = base / 'outputs/gigpo-e1-instruct200/FRESH200-V8-ACTUAL-READY-ONCE-INVOKE-v1.json'
if not done.returncode:
    result = json.loads(done.stdout)
    ready_raw = zlib.decompress(base64.b64decode(result.pop('ready_original_zlib_b64')))
    row = result['ready_record']
    assert len(ready_raw) == row['bytes'] and hashlib.sha256(ready_raw).hexdigest() == row['sha256']
    target = base / 'outputs/gigpo-e1-instruct200/actual-ready-v8-original-v1'
    target.mkdir(exist_ok=False)
    with (target / 'READY.json').open('xb') as f:
        f.write(ready_raw)
    result['local_original_ready'] = dict(path=str(target / 'READY.json'), bytes=len(ready_raw), sha256=row['sha256'])
    output_raw = (json.dumps(result, indent=2) + '\n').encode()
else:
    output_raw = done.stdout if done.stdout else json.dumps(dict(route_returncode=done.returncode, stderr=done.stderr.decode(errors='replace'))).encode()
with out_path.open('xb') as f:
    f.write(output_raw)
if done.returncode:
    raise RuntimeError('Actual READY failure preserved; inspect, never replay: ' + str(out_path))
print(json.dumps({k:result[k] for k in ('status','ready_record','local_original_ready','returncode','GPU_executed','submitted')}))
