"""One reviewed H100 CPU diagnostic on original evidence; no training or GPU."""
import hashlib
import json
import subprocess
from pathlib import Path

BASE = Path('outputs/gigpo-e1-policy-safety')
review_path = BASE / 'CALIBRATION3289-RAW-ACTION-SUPPORT-DIAGNOSTIC-INDEPENDENT-REVIEW-v1.json'
implementation_path = BASE / 'CALIBRATION3289-RAW-ACTION-SUPPORT-DIAGNOSTIC-IMPLEMENTATION-v1.json'
review_raw, implementation_raw = review_path.read_bytes(), implementation_path.read_bytes()
review, implementation = json.loads(review_raw), json.loads(implementation_raw)
assert review['status'] == 'PASS'
assert review['scope'] == 'STATIC_CPU_ONLY_DIAGNOSTIC_TOOL_REVIEW_NOT_ACTUAL_RAW_SUPPORT_OR_PROBABILITY_ACCEPTANCE'
assert hashlib.sha256(implementation_raw).hexdigest() == review['implementation_sha256']
for row in implementation['code_files'] + [implementation['input']]:
    raw = Path(row['path']).read_bytes()
    assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256']
root = '/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004'
identity = root + '/logs/CALIBRATION3289-RAW-ACTION-SUPPORT-CPU-INPUT-v1.json'
helper = root + '/tools/diagnostic-raw-support-3289-v1/diagnose_raw_support_v1.py'
output = root + '/logs/CALIBRATION3289-RAW-ACTION-SUPPORT-RESIDUAL-v1.json'
input_sha = implementation['input']['sha256']
assert review['input_sha256'] == input_sha
helper_sha = next(row['sha256'] for row in implementation['code_files'] if row['path'].endswith('/diagnose_raw_support_v1.py'))
remote = r'''import hashlib,json,os,subprocess,time,sys
from pathlib import Path
root=Path(ROOT)
helper=Path(HELPER)
identity=Path(IDENTITY)
output=Path(OUTPUT)
stem='CALIBRATION3289-RAW-ACTION-SUPPORT-CPU-'
claim=root/'logs'/(stem+'CLAIM-v1.json')
receipt=root/'logs'/(stem+'EXECUTION-v1.json')
stdout=root/'logs'/(stem+'STDOUT-v1.log')
stderr=root/'logs'/(stem+'STDERR-v1.log')
for path in (output,claim,receipt,stdout,stderr):
 assert not os.path.lexists(path),'Existing diagnostic artifact; inspect instead of repeating '+str(path)
for path,expected in ((helper,HELPER_SHA),(identity,INPUT_SHA)):
 assert path.is_file() and not path.is_symlink() and path.resolve()==path
 assert hashlib.sha256(path.read_bytes()).hexdigest()==expected
def save(path,data):
 with path.open('x') as stream:
  json.dump(data,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
def record(path):
 raw=path.read_bytes()
 return dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
save(claim,dict(scope='ONCE_CPU_READONLY_NOT_TRAINING',wrapper_pid=os.getpid(),started_unix_ns=time.time_ns(),
 review_sha256=REVIEW_SHA,helper_sha256=HELPER_SHA,input_sha256=INPUT_SHA,output=str(output)))
env=os.environ.copy()
for name in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy','LD_LIBRARY_PATH'):
 env.pop(name,None)
env.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
 PYTHONPATH=str(root/'repo-policy-safety-52befe9'),NO_PROXY='localhost,127.0.0.1,::1',no_proxy='localhost,127.0.0.1,::1')
argv=[str(root/'env/bin/python'),str(helper),'--identity',str(identity),'--identity-sha256',INPUT_SHA,'--output',str(output)]
with stdout.open('xb') as out,stderr.open('xb') as err:
 child=subprocess.Popen(argv,env=env,stdout=out,stderr=err)
 exit_code=child.wait()
result=dict(scope='ACTUAL_CPU_SUPPORT_RESIDUAL_EXECUTION_NOT_PROBABILITY_ACCEPTANCE',job_id=3289,node='h100-8',
 child_pid=child.pid,wrapper_pid=os.getpid(),exit_code=exit_code,finished_unix_ns=time.time_ns(),
 claim=record(claim),stdout=record(stdout),stderr=record(stderr),cuda_visible_devices='',models_loaded=False,
 formal_updates_contributed=0)
if output.exists():
 result['output']=record(output);result['diagnostic_status']=json.loads(output.read_text())['status']
save(receipt,result)
print(json.dumps(dict(execution_receipt=record(receipt),actual=result)))
sys.exit(exit_code)
'''
replacements = dict(ROOT=root, HELPER=helper, IDENTITY=identity, OUTPUT=output, INPUT_SHA=input_sha,
                    HELPER_SHA=helper_sha, REVIEW_SHA=hashlib.sha256(review_raw).hexdigest())
# Replace standalone placeholder tokens, not substrings of another identifier.
import re
remote = re.sub(r'\b(ROOT|HELPER|IDENTITY|OUTPUT|INPUT_SHA|HELPER_SHA|REVIEW_SHA)\b',
                lambda match: repr(replacements[match.group()]), remote)
compile(remote, 'one-reviewed-raw-support-diagnostic', 'exec')
script = "/usr/bin/python3 - <<'REMOTE'\n" + remote + '\nREMOTE\n'
process = subprocess.run(['/usr/bin/python3', 'work/remote.py', 'h100-8'], input=script,
                         text=True, capture_output=True)
print(process.stdout, end='')
if process.stderr:
    print(process.stderr, end='')
if process.returncode:
    raise SystemExit(process.returncode)
result = json.loads(process.stdout)
local_receipt = BASE / 'CALIBRATION3289-RAW-ACTION-SUPPORT-ACTUAL-EXECUTION-v1.json'
with local_receipt.open('x') as stream:
    json.dump(result, stream, indent=2);stream.write('\n')
row = result['actual'].get('output')
if row:
    assert row['path'] == output and row['bytes'] <= 1024 * 1024
    fetch = "/usr/bin/python3 - <<'REMOTE'\nimport pathlib,hashlib,sys\np=pathlib.Path(" + repr(output) + ")\nb=p.read_bytes()\nassert len(b)==" + str(row['bytes']) + " and hashlib.sha256(b).hexdigest()==" + repr(row['sha256']) + "\nsys.stdout.buffer.write(b)\nREMOTE\n"
    fetched = subprocess.run(['/usr/bin/python3','work/remote.py','h100-8'],input=fetch.encode(),capture_output=True)
    assert fetched.returncode == 0, fetched.stderr
    assert len(fetched.stdout) == row['bytes'] and hashlib.sha256(fetched.stdout).hexdigest() == row['sha256']
    with (BASE / 'CALIBRATION3289-RAW-ACTION-SUPPORT-ACTUAL-RESULT-v1.json').open('xb') as stream:
        stream.write(fetched.stdout)
    diagnostic = json.loads(fetched.stdout)
    print(json.dumps({key:diagnostic.get(key) for key in ['status','counters','token_residuals','complete_action_joint_residuals','support_errors','error','all_originals_unchanged']}))
