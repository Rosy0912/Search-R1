"""One immutable CPU-only target copy; no native/GPU/healthy admission claim."""
import base64
import hashlib
import json
import shlex
import subprocess
import sys
import uuid
import argparse
from pathlib import Path

sys.path.insert(0, '/Users/shaoshitong/Documents/remote_codex_control')
import control
repo = Path('/Users/shaoshitong/Documents/Search-G1/repo')
sys.path.insert(0, str(repo))
from reproduction.common8.source_files import runtime_files

prep = Path('/Users/shaoshitong/Documents/Search-G1/preparation')
stem = 'behavior-consistency-target-cpu-v3'
claim_path = prep / (stem + '-claim.json')
assert not claim_path.exists(), 'Inspect the existing target copy/result before another attempt'
files = {str(p.relative_to(repo)): base64.b64encode(p.read_bytes()).decode()
         for p in runtime_files(repo)}
hashes = {name: hashlib.sha256(base64.b64decode(data)).hexdigest() for name, data in files.items()}
base = '/dataset/shaoshitong/g1-common8-behavior-cpu-' + uuid.uuid4().hex[:12]
modules = ['reproduction.common8.test_policy_calibration_plan',
           'reproduction.common8.test_freeze_calibration_plan',
           'reproduction.common8.test_safety_calibration_evidence',
           'reproduction.common8.test_calibration_evaluation',
           'reproduction.common8.test_calibration_runtime',
           'reproduction.common8.test_prepare_calibration_run',
           'reproduction.common8.test_adopt_initial_probes',
           'reproduction.common8.test_production_safety_admission',
           'reproduction.common8.test_pipeline_calibration',
           'search_g1.test_policy_safety',
           'reproduction.common8.test_temporary_cache',
           'reproduction.common8.test_sampling_capture',
           'reproduction.common8.test_behavior_trace',
           'reproduction.common8.test_behavior_probability',
           'reproduction.common8.test_behavior_runtime',
           'reproduction.common8.test_generation_behavior']
claim = dict(base=base, source_sha256=hashes, modules=modules,
             scope='actual target CPU/Gloo tests; not GPU, native phase or healthy acceptance')
with claim_path.open('x') as f:
    json.dump(claim, f, indent=2)
payload = dict(base=base, files=files, source_sha256=hashes, modules=modules)
code = r'''
import base64,hashlib,json,os,subprocess,sys,time
from pathlib import Path
p=json.load(sys.stdin);b=Path(p['base']);b.mkdir(exist_ok=False)
for name,data in p['files'].items():
 path=b/name
 assert not name.startswith('/') and '..' not in Path(name).parts
 path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(base64.b64decode(data))
assert {name:hashlib.sha256((b/name).read_bytes()).hexdigest() for name in p['files']}==p['source_sha256']
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(b),OMP_NUM_THREADS='1')
env.pop('PYTHONHOME',None)
python='/dataset/shaoshitong/Search-G1-common8/env/bin/python'
runner="""import contextlib,io,json,sys,time,unittest,torch
started=time.time();stream=io.StringIO()
with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
 suite=unittest.defaultTestLoader.loadTestsFromNames(sys.argv[1:])
 result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
print(json.dumps(dict(tests_run=result.testsRun,skipped=result.skipped,failures=len(result.failures),errors=len(result.errors),successful=result.wasSuccessful(),cuda_available=torch.cuda.is_available(),torch_version=torch.__version__,duration_seconds=time.time()-started,output=stream.getvalue())))
sys.exit(0 if result.wasSuccessful() and not result.skipped and not torch.cuda.is_available() else 1)
"""
r=subprocess.run([python,'-c',runner,*p['modules']],cwd=b,env=env,capture_output=True,text=True,timeout=240)
syntax=subprocess.run(['bash','-n','reproduction/common8/train.sh','reproduction/common8/launch_calibration_native.sh'],cwd=b,capture_output=True,text=True)
proof=dict(base=str(b),checked_unix=time.time(),returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,
 source_sha256={name:hashlib.sha256((b/name).read_bytes()).hexdigest() for name in p['files']},
 modules=p['modules'],shell_syntax_returncode=syntax.returncode,shell_syntax_stderr=syntax.stderr,
 scope='actual target CPU/Gloo regressions; not GPU/native phase/healthy admission')
(b/'target-cpu-result.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
'''
r = subprocess.run(control.route('h100-8', 'python3 -c ' + shlex.quote(code)),
                   input=json.dumps(payload), capture_output=True, text=True, timeout=280)
if r.returncode:
    raise RuntimeError('Target observation failed; inspect same copy: ' + r.stderr[-1800:])
proof = json.loads(r.stdout)
(prep / (stem + '-result.json')).write_text(json.dumps(proof, indent=2))
assert proof['source_sha256'] == hashes
summary = json.loads(proof['stdout']) if proof['stdout'].startswith('{') else dict(output=proof['stdout'])
output = summary.pop('output', '')
if proof['returncode']:
    summary['failure_excerpt'] = output[-4000:]
print(json.dumps(dict(base=base, returncode=proof['returncode'], summary=summary,
                     stderr=proof['stderr'][-2200:], shell_syntax=proof['shell_syntax_returncode']), indent=2))
sys.exit(proof['returncode'])
