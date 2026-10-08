"""Invoke the reviewed durable single-submit helper once; never infer a job on uncertainty."""
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
interpreter = '/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004/env/bin/python'
environment = dict(PATH='/usr/bin:/bin', LANG='C', LC_ALL='C', PYTHONDONTWRITEBYTECODE='1')
ready = json.loads((base / 'outputs/gigpo-e1-instruct200/FRESH200-V8-ACTUAL-READY-ONCE-INVOKE-v1.json').read_bytes())['ready_record']
assert ready['path'] == state + '/READY.json'
review_path = base / 'outputs/gigpo-e1-instruct200/FRESH200-V8-ACTUAL-READY-SINGLE-SUBMIT-INDEPENDENT-REVIEW-v1.json'
review_raw = review_path.read_bytes()
review = json.loads(review_raw)
assert review['status'] == 'PASS' and review['scope'] == 'FRESH200_LIVE200_SINGLE_SUBMIT_REVIEW'
assert review['independent_review'] is True and review['blocking_findings'] == []
assert review['ready_sha256'] == ready['sha256'] and review['effective_submission_environment'] == environment
assert review['single_submit_wrapper_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
review_record = dict(path=root + '/logs/' + review_path.name, bytes=len(review_raw), sha256=hashlib.sha256(review_raw).hexdigest())
spec_raw = (json.dumps(dict(ready=ready, single_submit_review=review_record), sort_keys=True, indent=2) + '\n').encode()
spec_record = dict(path=root + '/logs/FRESH200-V8-ACTUAL-SINGLE-SUBMIT-SPEC-v1.json', bytes=len(spec_raw), sha256=hashlib.sha256(spec_raw).hexdigest())
command = [interpreter, '-B', '-S', root + '/tools/fresh200-deployment-controls-v8/submit_once_v8.py', json.dumps(spec_record)]
payload = dict(state=state, command=command, environment=environment, review=review_record,
    review_b64=base64.b64encode(review_raw).decode(), spec=spec_record, spec_b64=base64.b64encode(spec_raw).decode())
code = '''import base64,hashlib,json,os,subprocess,sys
from pathlib import Path
if sys.flags.optimize or not sys.flags.no_site or not sys.flags.ignore_environment:raise RuntimeError('Isolated startup required')
p=json.load(sys.stdin);state=Path(p['state']);assert state.resolve()==state and sorted(x.name for x in state.iterdir())==['CANDIDATE.json','PHYSICAL.json','READY.json']
for name in ('review','spec'):
 row=p[name];path=Path(row['path']);raw=base64.b64decode(p[name+'_b64'])
 assert path.parent.resolve()==path.parent and not path.is_symlink();assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 if path.exists():assert path.read_bytes()==raw
 else:
  with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
done=subprocess.run(p['command'],capture_output=True,text=True,env=p['environment'])
out=dict(status='SINGLE_SUBMIT_FAILED_OR_UNCERTAIN_INSPECT_ORIGINALS_NEVER_REPLAY' if done.returncode else 'SINGLE_SUBMIT_HELPER_RETURNED_SUCCESS',command=p['command'],returncode=done.returncode,stdout=done.stdout,stderr=done.stderr)
if not done.returncode:
 result=json.loads(done.stdout);assert result['status']=='SUBMITTED_EXACTLY_ONCE' and result['actual_gpu_acceptance'] is False and result['actual_probability_acceptance'] is False
 out['actual_submission_receipt']=result
print(json.dumps(out));raise SystemExit(done.returncode)
'''
inner = 'env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C ' + shlex.quote(interpreter) + ' -E -B -S -c ' + shlex.quote(code)
mid = 'env -u LD_LIBRARY_PATH ssh -T -o ConnectTimeout=15 h100-8 ' + shlex.quote(inner)
route = control.ssh('mgmt-host') + ['env -u LD_LIBRARY_PATH ssh -T -o ConnectTimeout=15 h100-16 ' + shlex.quote(mid)]
done = subprocess.run(route, input=json.dumps(payload).encode(), capture_output=True, timeout=600)
out_path = base / 'outputs/gigpo-e1-instruct200/FRESH200-V8-ACTUAL-SINGLE-SUBMIT-ONCE-INVOKE-v1.json'
with out_path.open('xb') as f:
    f.write(done.stdout if done.stdout else json.dumps(dict(route_returncode=done.returncode, stderr=done.stderr.decode(errors='replace'))).encode())
    f.write(b'\n')
if done.returncode:
    raise RuntimeError('Submission outcome preserved; inspect intent/receipt/Slurm, NEVER REPLAY: ' + str(out_path))
result = json.loads(done.stdout)['actual_submission_receipt']
print(json.dumps({k:result[k] for k in ('status','job_id','binding','actual_counted_outer_updates','actual_gpu_acceptance','actual_probability_acceptance')}))
