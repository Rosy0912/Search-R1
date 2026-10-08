"""Immutable common8 target CPU receipt; importing or dry-checking never uses SSH.

Explicit --execute uses the existing control.route path.  Neither CPU success
nor this proof admits GPU training, physical context parity or healthy weights.
"""
import argparse
import base64
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
import uuid
from pathlib import Path

REPO = Path('/Users/shaoshitong/Documents/Search-G1/repo')
PREP = Path('/Users/shaoshitong/Documents/Search-G1/preparation')
CONTROL_DIR = '/Users/shaoshitong/Documents/remote_codex_control'
TARGET_PYTHON = '/dataset/shaoshitong/Search-G1-common8/env/bin/python'
STEM = 'behavior-consistency-target-cpu-v5'
EXPECTED_FILES = 354
VERSION_FILES = ('verl/version/version', 'verl/single_controller/version/version')
SUFFIXES = ('.py', '.sh', '.yaml', '.yml', '.json')
MODULES = (
    'reproduction.common8.test_policy_calibration_plan',
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
    'reproduction.common8.test_generation_behavior',
)
SCOPE = 'actual target CPU/Gloo regressions; not GPU/native phase/healthy admission'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def runtime_name(name):
    p = Path(name)
    return (not p.is_absolute() and '..' not in p.parts and
            not {'.git', '__pycache__', 'evidence'}.intersection(p.parts) and
            (p.suffix in SUFFIXES or name in VERSION_FILES))


def git(repo, *args):
    r = subprocess.run(['git', '-C', str(repo), *args], capture_output=True, check=True)
    return r.stdout


def build_payload(repo, commit, base, expected_count=EXPECTED_FILES):
    """Bind the complete committed runtime tree to exact current bytes."""
    require(type(commit) is str and re.fullmatch('[0-9a-f]{40}', commit),
            'An explicit full committed source SHA is required')
    require(git(repo, 'rev-parse', 'HEAD').decode().strip() == commit,
            'Source commit must be the current reviewed HEAD')
    names = git(repo, 'ls-tree', '-r', '--name-only', '-z', commit).decode().split('\0')
    names = sorted(name for name in names if name and runtime_name(name))
    require(len(names) == expected_count and set(VERSION_FILES).issubset(names),
            'Complete committed runtime source count/version files differ')
    actual = sorted(str(p.relative_to(repo)) for p in repo.rglob('*')
                    if p.is_file() and runtime_name(str(p.relative_to(repo))))
    require(actual == names, 'Current runtime file set differs from committed source')
    files, hashes = {}, {}
    for name in names:
        path = repo / name
        require(not path.is_symlink(), 'Runtime symlink is not an ordinary committed file: ' + name)
        committed = git(repo, 'show', commit + ':' + name)
        current = path.read_bytes()
        require(current == committed, 'Runtime source changed after commit: ' + name)
        files[name] = base64.b64encode(current).decode()
        hashes[name] = sha(current)
    require(re.fullmatch('/dataset/shaoshitong/g1-common8-behavior-cpu-v5-[0-9a-f]{12}', base),
            'Noncanonical unique target CPU namespace')
    return dict(schema_version=1, base=base, source_commit=commit,
                expected_source_count=expected_count, source_sha256=hashes,
                source_map_sha256=sha(canonical(hashes)), files=files,
                modules=list(MODULES), scope=SCOPE)


CPU_RUNNER = r'''
import contextlib,hashlib,importlib.metadata,io,json,sys,time,unittest,torch
started=time.time();stream=io.StringIO()
before_initialized=torch.cuda.is_initialized()
versions={name:importlib.metadata.version(name) for name in ('torch','transformers','vllm','tensordict')}
with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
 import transformers,vllm
 suite=unittest.defaultTestLoader.loadTestsFromNames(sys.argv[1:])
 result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
available=torch.cuda.is_available();initialized=torch.cuda.is_initialized()
summary=dict(tests_run=result.testsRun,skipped=result.skipped,failures=len(result.failures),
 errors=len(result.errors),successful=result.wasSuccessful(),cuda_available=available,
 cuda_initialized_before=before_initialized,cuda_initialized_after=initialized,
 torch_version=torch.__version__,transformers_module_version=transformers.__version__,
 vllm_module_version=vllm.__version__,distribution_versions=versions,
 duration_seconds=time.time()-started,output=stream.getvalue())
version_ok=(versions['torch'] in ('2.4.0','2.4.0+cu121') and torch.__version__=='2.4.0+cu121' and
 versions['transformers']=='4.46.3' and versions['vllm']=='0.6.3' and versions['tensordict']=='0.5.0')
summary['expected_distribution_versions_matched']=version_ok
print(json.dumps(summary,allow_nan=False))
sys.exit(0 if result.wasSuccessful() and result.testsRun>0 and not result.skipped and
 not available and not before_initialized and not initialized and version_ok else 1)
'''


REMOTE_CODE = r'''
import base64,hashlib,json,os,re,subprocess,sys,time
from pathlib import Path
p=json.load(sys.stdin)
def require(ok,why):
 if not ok:raise ValueError(why)
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(x):return hashlib.sha256(x).hexdigest()
require(p['schema_version']==1 and re.fullmatch('[0-9a-f]{40}',p['source_commit']),'Bad source identity')
require(re.fullmatch('/dataset/shaoshitong/g1-common8-behavior-cpu-v5-[0-9a-f]{12}',p['base']),'Bad namespace')
require(p['expected_source_count']==354 and len(p['files'])==354 and
 set(p['files'])==set(p['source_sha256']),'Bad complete runtime source count')
require(sha(canonical(p['source_sha256']))==p['source_map_sha256'],'Bad canonical source map')
decoded={}
for name,data in p['files'].items():
 path=Path(name)
 require(not path.is_absolute() and '..' not in path.parts and
  not {'.git','__pycache__','evidence'}.intersection(path.parts),'Unsafe source path')
 decoded[name]=base64.b64decode(data,validate=True)
 require(sha(decoded[name])==p['source_sha256'][name],'Source payload hash changed')
b=Path(p['base']);b.mkdir(exist_ok=False)
for name,data in decoded.items():
 path=b/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
before={name:sha((b/name).read_bytes()) for name in p['files']}
require(before==p['source_sha256'],'Target source before tests differs')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(b),OMP_NUM_THREADS='1',
 PYTHONNOUSERSITE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
env.pop('PYTHONHOME',None)
started=time.time()
try:
 r=subprocess.run([p['target_python'],'-c',p['cpu_runner'],*p['modules']],cwd=b,env=env,
  capture_output=True,text=True,timeout=300)
 execution=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,observation_timed_out=False)
except subprocess.TimeoutExpired as e:
 def text(x):return x.decode(errors='replace') if isinstance(x,bytes) else (x or '')
 execution=dict(returncode=124,stdout=text(e.stdout),stderr=text(e.stderr),observation_timed_out=True)
syntax_checks=[]
for filename in ('reproduction/common8/train.sh','reproduction/common8/launch_calibration_native.sh'):
 syntax=subprocess.run(['bash','-n',filename],cwd=b,capture_output=True,text=True,timeout=10)
 syntax_checks.append(dict(path=filename,returncode=syntax.returncode,stdout=syntax.stdout,stderr=syntax.stderr))
syntax_returncode=next((check['returncode'] for check in syntax_checks if check['returncode']),0)
syntax_stderr='\n'.join(check['path']+': '+check['stderr'] for check in syntax_checks if check['stderr'])
after={name:sha((b/name).read_bytes()) for name in p['files']}
proof=dict(schema_version=1,base=str(b),started_unix=started,checked_unix=time.time(),
 source_commit=p['source_commit'],source_map_sha256=p['source_map_sha256'],
 source_sha256=after,source_before_tests_sha256=before,source_unchanged=before==after,
 expected_source_count=354,actual_source_count=len(after),modules=p['modules'],
 shell_syntax_returncode=syntax_returncode,shell_syntax_stderr=syntax_stderr,shell_syntax_checks=syntax_checks,
 cuda_visible_devices='',target_python=p['target_python'],actual_production_updates=0,
 production_admission=False,healthy_checkpoint_accepted=False,scope=p['scope'],**execution)
with (b/'target-cpu-result.json').open('x') as f:json.dump(proof,f,indent=2,allow_nan=False)
print(json.dumps(proof,allow_nan=False))
'''


def validate_proof(proof, payload):
    require(proof['schema_version'] == 1 and proof['base'] == payload['base'] and
            proof['source_commit'] == payload['source_commit'] and
            proof['source_sha256'] == payload['source_sha256'] and
            proof['source_before_tests_sha256'] == payload['source_sha256'] and
            proof['source_map_sha256'] == payload['source_map_sha256'] and
            proof['source_unchanged'] is True and
            proof['expected_source_count'] == proof['actual_source_count'] == EXPECTED_FILES and
            proof['modules'] == list(MODULES), 'Target proof differs from committed payload')
    require(proof['actual_production_updates'] == 0 and
            proof['production_admission'] is False and proof['healthy_checkpoint_accepted'] is False and
            proof['scope'] == SCOPE and proof['target_python'] == TARGET_PYTHON and
            proof['cuda_visible_devices'] == '', 'CPU scope/target identity changed')
    require(proof['returncode'] == 0 and proof['observation_timed_out'] is False and
            proof['shell_syntax_returncode'] == 0, 'Target CPU test/syntax process failed')
    syntax = proof['shell_syntax_checks']
    require([check['path'] for check in syntax] == ['reproduction/common8/train.sh',
                                                 'reproduction/common8/launch_calibration_native.sh'] and
            all(check['returncode'] == 0 for check in syntax),
            'Each shell file must independently pass syntax checking')
    summary = json.loads(proof['stdout'])
    require(summary['tests_run'] > 0 and summary['skipped'] == [] and
            summary['failures'] == summary['errors'] == 0 and summary['successful'] is True and
            summary['cuda_available'] is False and summary['cuda_initialized_before'] is False and
            summary['cuda_initialized_after'] is False and
            summary['expected_distribution_versions_matched'] is True,
            'Full zero-skip CPU tests/installed environment not accepted')
    versions = summary['distribution_versions']
    require(set(versions) == {'torch','transformers','vllm','tensordict'} and
            versions['torch'] in ('2.4.0','2.4.0+cu121') and
            versions['transformers'] == '4.46.3' and versions['vllm'] == '0.6.3' and
            versions['tensordict'] == '0.5.0' and
            summary['torch_version'] == '2.4.0+cu121', 'Installed distribution versions differ')
    return summary


def exclusive_json(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--execute', action='store_true', help='Actually run the one CPU-only target copy')
    args = parser.parse_args(argv)
    base = '/dataset/shaoshitong/g1-common8-behavior-cpu-v5-' + uuid.uuid4().hex[:12]
    payload = build_payload(REPO, args.source_commit, base)
    if not args.execute:
        print(json.dumps(dict(source_commit=args.source_commit,source_count=len(payload['files']),
                              source_map_sha256=payload['source_map_sha256'],modules=list(MODULES),
                              scope='local structure check only; no claim/SSH/target copy'),indent=2))
        return 0
    claim_path, result_path = (PREP / (STEM + suffix) for suffix in ('-claim.json', '-result.json'))
    require(not claim_path.exists() and not result_path.exists(),
            'Prior v5 claim/result exists: inspect the same handle, never repeat or overwrite')
    claim = {key: value for key,value in payload.items() if key != 'files'}
    claim.update(claimed_unix=time.time(),runner_sha256=sha(Path(__file__).read_bytes()),
                 remote_code_sha256=sha(REMOTE_CODE.encode()),cpu_runner_sha256=sha(CPU_RUNNER.encode()))
    exclusive_json(claim_path, claim)
    payload.update(target_python=TARGET_PYTHON,cpu_runner=CPU_RUNNER)
    sys.path.insert(0, CONTROL_DIR)
    import control
    command = 'python3 -c ' + shlex.quote(REMOTE_CODE)
    try:
        r = subprocess.run(control.route('h100-8', command), input=json.dumps(payload),
                           capture_output=True, text=True, timeout=370)
    except subprocess.TimeoutExpired:
        raise RuntimeError('SSH observation timed out; retain claim and inspect the same base: ' + base)
    if r.returncode:
        raise RuntimeError('SSH observation failed; retain claim and inspect the same base: ' + base +
                           '\n' + r.stderr[-1800:])
    proof = json.loads(r.stdout)
    exclusive_json(result_path, proof)
    summary = validate_proof(proof, payload)
    summary.pop('output', None)
    print(json.dumps(dict(base=base,source_commit=args.source_commit,summary=summary,
                          result_sha256=sha(result_path.read_bytes()),result_path=str(result_path),
                          scope=SCOPE),indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
