"""Freeze reviewed fresh200 sources and run CPU regressions only on --execute.

Import and --check-inputs do not import the controller, claim a run, contact a
target, or write files. Target CPU/Gloo success is not training/GPU admission.
"""
import argparse
import base64
import hashlib
import json
import os
import re
import shlex
import stat
import subprocess
import sys
import time
import uuid
from pathlib import Path

REPO = Path('/Users/shaoshitong/Documents/Search-G1/repo')
PREP = Path('/Users/shaoshitong/Documents/Search-G1/preparation')
CONTROL_DIR = '/Users/shaoshitong/Documents/remote_codex_control'
TARGET_PYTHON = '/dataset/shaoshitong/Search-G1-common8/env/bin/python'
STEM = 'fresh200-target-cpu'
SCHEMA_VERSION = 2
ROLE = 'fresh200_target_cpu_regression'
SCOPE = 'actual target CPU/Gloo regressions; not GPU/native phase/healthy admission'
BASE_PATTERN = r'/dataset/shaoshitong/g1-fresh200-cpu-[0-9a-f]{32}'
VERSION_FILES = ('verl/version/version', 'verl/single_controller/version/version')
SUFFIXES = ('.py', '.sh', '.yaml', '.yml', '.json')
EXCLUDED_PARTS = frozenset(('.git', 'evidence', 'cache', 'caches', '__pycache__',
                            '.cache', '.pytest_cache', '.mypy_cache', '.ruff_cache'))
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_SOURCE_BYTES = 16 * 1024 * 1024
CPU_TIMEOUT_SECONDS = 900
RPC_TIMEOUT_SECONDS = 1060
SHELL_FILES = ('reproduction/common8/train.sh',
               'reproduction/common8/launch_calibration_native.sh')
FRESH200_MODULES = (
    'reproduction.common8.test_scheduler',
    'reproduction.common8.test_scheduler_collective',
    'reproduction.common8.test_configuration',
    'reproduction.common8.test_contract',
    'reproduction.common8.test_policy_calibration_plan',
    'reproduction.common8.test_pipeline_calibration',
    'reproduction.common8.test_formal_contract',
    'reproduction.common8.test_formal_validation',
    'reproduction.common8.test_fresh200_bootstrap',
    'reproduction.common8.test_adopt_fresh200_probes',
    'reproduction.common8.test_gpu_training_admission',
)
BEHAVIOR_MODULES = (
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
MODULES = tuple(dict.fromkeys(FRESH200_MODULES + BEHAVIOR_MODULES))
SUMMARY_PREFIX = 'FRESH200_CPU_SUMMARY='


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key: ' + key)
            result[key] = value
        return result
    def constant(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError('Invalid JSON receipt') from error


def exact_keys(document, names, label):
    require(type(document) is dict and set(document) == set(names),
            'Unexpected ' + label + ' schema')


def runtime_name(name):
    if not isinstance(name, str) or not name or any(ord(c) in (0, 10, 13, 92) for c in name):
        return False
    path = Path(name)
    return (not path.is_absolute() and '..' not in path.parts and
            path.as_posix() == name and not EXCLUDED_PARTS.intersection(path.parts) and
            (path.suffix in SUFFIXES or name in VERSION_FILES))


def safe_regular_bytes(path, expected_size=None, max_bytes=2 * 1024 * 1024):
    """Read bounded ordinary bytes; reject links, FIFOs and in-read replacement."""
    path = Path(path)
    require(type(max_bytes) is int and max_bytes >= 0 and
            (expected_size is None or type(expected_size) is int and 0 <= expected_size <= max_bytes),
            'Invalid regular-file read bound')
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode), 'Safe reader requires an ordinary regular file')
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, 'O_CLOEXEC', 0)
    fd = os.open(path, flags)
    try:
        opened = os.fstat(fd)
        require(stat.S_ISREG(opened.st_mode), 'Safe reader opened a nonordinary file')
        def signature(observation):
            return (observation.st_dev, observation.st_ino, observation.st_mode,
                    observation.st_size, observation.st_mtime_ns, observation.st_ctime_ns)
        require(signature(before) == signature(opened), 'Safe reader inode/file changed before open')
        require(0 <= opened.st_size <= max_bytes and
                (expected_size is None or opened.st_size == expected_size),
                'Safe reader file size differs from bound')
        limit = max_bytes if expected_size is None else expected_size
        chunks, count = [], 0
        while True:
            chunk = os.read(fd, min(65536, limit - count + 1))
            if not chunk:
                break
            count += len(chunk)
            require(count <= limit, 'Safe reader file grew beyond bounded size')
            chunks.append(chunk)
        after_fd, after_path = os.fstat(fd), path.lstat()
        require(stat.S_ISREG(after_fd.st_mode) and stat.S_ISREG(after_path.st_mode) and
                signature(opened) == signature(after_fd) == signature(after_path),
                'Safe reader inode/file changed during read')
        require(count == opened.st_size, 'Safe reader short or changed regular-file read')
        return b''.join(chunks)
    finally:
        os.close(fd)


def runtime_map(repo):
    """Inspect the complete runtime file set without following any source link."""
    repo = Path(repo)
    require(repo.is_dir() and not repo.is_symlink(), 'Source root is not an ordinary directory')
    result = {}
    def walk_error(error):
        raise error
    for root, directories, files in os.walk(repo, followlinks=False, onerror=walk_error):
        directories[:] = sorted(name for name in directories if name not in EXCLUDED_PARTS)
        for name in directories:
            require(not (Path(root) / name).is_symlink(), 'Runtime source directory symlink')
        for name in sorted(files):
            path = Path(root) / name
            relative = path.relative_to(repo).as_posix()
            if runtime_name(relative):
                data = safe_regular_bytes(path, max_bytes=MAX_FILE_BYTES)
                result[relative] = sha(data)
    return dict(sorted(result.items()))


def git(repo, *args):
    return subprocess.run(['git', '-C', str(repo), *args], capture_output=True,
                          check=True, timeout=30).stdout


CPU_RUNNER = r'''
"""File-backed runner: multiprocessing spawn may safely re-import this file."""
import contextlib
import importlib.metadata
import io
import json
import os
import platform
import sys
import time
import traceback
import unittest

PREFIX = 'FRESH200_CPU_SUMMARY='

def test_name(case):
    return str(case.id()) if callable(getattr(case, 'id', None)) else str(case)

def main(argv=None):
    modules = list(sys.argv[1:] if argv is None else argv)
    started = time.time()
    stream = io.StringIO()
    summary = dict(tests_run=0, skipped=[], failures=0, errors=0, successful=False,
        expected_failures=0, unexpected_successes=0, expected_failure_details=[],
        unexpected_success_details=[], failure_details=[], error_details=[], module_test_counts={},
        cuda_available=None, cuda_initialized_before=None, cuda_initialized_after=None,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        distribution_versions={}, torch_version=None, transformers_module_version=None,
        vllm_module_version=None, expected_distribution_versions_matched=False,
        python_version=platform.python_version(), python_executable=sys.executable,
        system=platform.system(), modules=modules, runner_error=None,
        duration_seconds=0., output='')
    torch = None
    try:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            import torch
            summary['cuda_initialized_before'] = torch.cuda.is_initialized()
            if summary['cuda_initialized_before'] or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
                raise RuntimeError('CUDA must remain hidden and uninitialized before tests')
            summary['distribution_versions'] = {
                name: importlib.metadata.version(name)
                for name in ('torch', 'transformers', 'vllm', 'tensordict')}
            import transformers
            import vllm
            summary.update(torch_version=str(torch.__version__),
                transformers_module_version=str(transformers.__version__),
                vllm_module_version=str(vllm.__version__))
            suites = []
            for module in modules:
                suite = unittest.defaultTestLoader.loadTestsFromName(module)
                summary['module_test_counts'][module] = suite.countTestCases()
                suites.append(suite)
            result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.TestSuite(suites))
            summary.update(tests_run=result.testsRun,
                skipped=[[test_name(case), str(reason)] for case, reason in result.skipped],
                failures=len(result.failures), errors=len(result.errors),
                expected_failures=len(result.expectedFailures), unexpected_successes=len(result.unexpectedSuccesses),
                expected_failure_details=[dict(test=test_name(case), traceback=str(detail))
                                          for case, detail in result.expectedFailures],
                unexpected_success_details=[test_name(case) for case in result.unexpectedSuccesses],
                failure_details=[dict(test=test_name(case), traceback=str(detail))
                                 for case, detail in result.failures],
                error_details=[dict(test=test_name(case), traceback=str(detail))
                               for case, detail in result.errors], successful=result.wasSuccessful())
            versions = summary['distribution_versions']
            summary['expected_distribution_versions_matched'] = (
                versions['torch'] in ('2.4.0', '2.4.0+cu121') and
                summary['torch_version'] == '2.4.0+cu121' and
                versions['transformers'] == summary['transformers_module_version'] == '4.46.3' and
                versions['vllm'] == '0.6.3' and versions['tensordict'] == '0.5.0')
    except BaseException:
        summary['runner_error'] = traceback.format_exc()
    finally:
        if torch is not None:
            summary['cuda_initialized_after'] = torch.cuda.is_initialized()
            summary['cuda_available'] = torch.cuda.is_available()
            if torch.cuda.is_initialized():
                summary['cuda_initialized_after'] = True
        summary['duration_seconds'] = time.time() - started
        summary['output'] = stream.getvalue()
    counts = summary['module_test_counts']
    accepted = (summary['runner_error'] is None and summary['successful'] and
        summary['tests_run'] > 0 and counts and len(counts) == len(modules) and
        all(type(n) is int and n > 0 for n in counts.values()) and
        sum(counts.values()) == summary['tests_run'] and not summary['skipped'] and
        summary['failures'] == summary['errors'] == 0 and
        summary['expected_failures'] == summary['unexpected_successes'] == 0 and
        summary['cuda_initialized_before'] is False and summary['cuda_initialized_after'] is False and
        summary['cuda_available'] is False and summary['expected_distribution_versions_matched'] and
        summary['system'] == 'Linux')
    print(PREFIX + json.dumps(summary, allow_nan=False), flush=True)
    return 0 if accepted else 1

if __name__ == '__main__':
    sys.exit(main())
'''

PAYLOAD_KEYS = frozenset(('schema_version', 'role', 'base', 'source_commit',
    'expected_source_count', 'source_sha256', 'source_map_sha256', 'files', 'modules',
    'module_count', 'scope', 'target_python', 'cpu_runner', 'cpu_runner_sha256',
    'runner_timeout_seconds'))


def validate_payload(payload):
    exact_keys(payload, PAYLOAD_KEYS, 'source payload')
    require(type(payload['schema_version']) is int and payload['schema_version'] == SCHEMA_VERSION and
            payload['role'] == ROLE and payload['scope'] == SCOPE, 'Wrong payload role/version/scope')
    require(type(payload['source_commit']) is str and
            re.fullmatch('[0-9a-f]{40}', payload['source_commit']), 'Full committed source SHA required')
    require(type(payload['base']) is str and re.fullmatch(BASE_PATTERN, payload['base']),
            'Noncanonical unique target CPU namespace')
    files, hashes = payload['files'], payload['source_sha256']
    count = payload['expected_source_count']
    require(type(count) is int and count > 0 and type(files) is dict and type(hashes) is dict and
            len(files) == len(hashes) == count and set(files) == set(hashes),
            'Complete frozen runtime source count/map differs')
    require(set(VERSION_FILES).issubset(files) and set(SHELL_FILES).issubset(files),
            'Required package versions or shell source absent')
    require(payload['modules'] == list(MODULES) and type(payload['module_count']) is int and
            payload['module_count'] == len(MODULES), 'Exact deduplicated test module union differs')
    require(all(module.replace('.', '/') + '.py' in files for module in MODULES),
            'Requested regression module source absent')
    total = 0
    for name, encoded in files.items():
        require(runtime_name(name), 'Unsafe runtime source path')
        require(type(hashes[name]) is str and re.fullmatch('[0-9a-f]{64}', hashes[name]),
                'Invalid runtime SHA256')
        require(type(encoded) is str, 'Source payload is not base64 text')
        try:
            data = base64.b64decode(encoded, validate=True)
        except (ValueError, base64.binascii.Error) as error:
            raise ValueError('Invalid source base64') from error
        require(base64.b64encode(data).decode() == encoded and len(data) <= MAX_FILE_BYTES and
                sha(data) == hashes[name], 'Source payload bytes/hash differ')
        total += len(data)
    require(total <= MAX_SOURCE_BYTES, 'Source-only CPU payload exceeds its small transfer bound')
    require(payload['source_map_sha256'] == sha(canonical(hashes)), 'Canonical source map SHA differs')
    require(payload['target_python'] == TARGET_PYTHON and payload['cpu_runner'] == CPU_RUNNER and
            payload['cpu_runner_sha256'] == sha(CPU_RUNNER.encode()) and
            type(payload['runner_timeout_seconds']) is int and
            payload['runner_timeout_seconds'] == CPU_TIMEOUT_SECONDS,
            'Target environment or guarded file runner differs')
    return payload


def build_payload(repo, commit, base):
    """Freeze the dynamic complete reviewed HEAD tree and exact current bytes."""
    repo = Path(repo)
    require(type(commit) is str and re.fullmatch('[0-9a-f]{40}', commit),
            'An explicit full committed source SHA is required')
    require(git(repo, 'rev-parse', 'HEAD').decode().strip() == commit,
            'Source commit must be the current reviewed HEAD')
    initial = runtime_map(repo)
    names = []
    for record in git(repo, 'ls-tree', '-r', '-z', commit).split(b'\0'):
        if not record:
            continue
        metadata, raw_name = record.split(b'\t', 1)
        name = raw_name.decode('utf-8')
        if runtime_name(name):
            mode, kind, _ = metadata.decode().split()
            require(mode in ('100644', '100755') and kind == 'blob',
                    'Runtime source symlink or non-file in committed tree: ' + name)
            names.append(name)
    names.sort()
    require(names and len(names) == len(set(names)) and set(initial) == set(names),
            'Current runtime file set differs from committed source')
    files, hashes, total = {}, {}, 0
    for name in names:
        path = repo / name
        size_text = git(repo, 'cat-file', '-s', commit + ':' + name).decode().strip()
        require(re.fullmatch('0|[1-9][0-9]*', size_text) is not None, 'Invalid committed runtime blob size')
        committed_size = int(size_text)
        require(committed_size <= MAX_FILE_BYTES, 'Committed runtime source file is too large')
        total += committed_size
        require(total <= MAX_SOURCE_BYTES, 'Source-only CPU payload exceeds its small transfer bound')
        committed = git(repo, 'show', commit + ':' + name)
        require(len(committed) == committed_size, 'Committed runtime blob size changed')
        current = safe_regular_bytes(path, expected_size=committed_size, max_bytes=MAX_FILE_BYTES)
        require(current == committed, 'Runtime source changed after commit: ' + name)
        files[name] = base64.b64encode(current).decode()
        hashes[name] = sha(current)
    require(initial == runtime_map(repo) == hashes, 'Current runtime source changed while freezing payload')
    require(git(repo, 'rev-parse', 'HEAD').decode().strip() == commit,
            'Reviewed HEAD changed while freezing payload')
    payload = dict(schema_version=SCHEMA_VERSION, role=ROLE, base=base, source_commit=commit,
        expected_source_count=len(names), source_sha256=hashes,
        source_map_sha256=sha(canonical(hashes)), files=files, modules=list(MODULES),
        module_count=len(MODULES), scope=SCOPE, target_python=TARGET_PYTHON,
        cpu_runner=CPU_RUNNER, cpu_runner_sha256=sha(CPU_RUNNER.encode()),
        runner_timeout_seconds=CPU_TIMEOUT_SECONDS)
    return validate_payload(payload)


REMOTE_CODE = r'''
import base64,hashlib,json,os,re,signal,stat,subprocess,sys,time
from pathlib import Path
def need(ok,why):
    if not ok: raise ValueError(why)
require=need
def canonical(x): return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(x): return hashlib.sha256(x).hexdigest()
def pairs(items):
    result={}
    for key,value in items:
        need(key not in result,'Duplicate payload JSON key')
        result[key]=value
    return result
def constant(value): raise ValueError('Nonfinite JSON payload')
p=json.load(sys.stdin,object_pairs_hook=pairs,parse_constant=constant)
keys={'schema_version','role','base','source_commit','expected_source_count','source_sha256',
      'source_map_sha256','files','modules','module_count','scope','target_python','cpu_runner',
      'cpu_runner_sha256','runner_timeout_seconds'}
need(type(p) is dict and set(p)==keys,'Wrong payload schema')
need(type(p['schema_version']) is int and p['schema_version']==2 and
     p['role']=='fresh200_target_cpu_regression','Wrong payload role/version')
need(p['scope']=='actual target CPU/Gloo regressions; not GPU/native phase/healthy admission','Wrong CPU scope')
need(type(p['source_commit']) is str and re.fullmatch('[0-9a-f]{40}',p['source_commit']),'Bad source identity')
need(type(p['base']) is str and re.fullmatch('/dataset/shaoshitong/g1-fresh200-cpu-[0-9a-f]{32}',p['base']),'Bad namespace')
need(p['target_python']=='/dataset/shaoshitong/Search-G1-common8/env/bin/python','Wrong original target environment')
need(type(p['runner_timeout_seconds']) is int and p['runner_timeout_seconds']==900,'Unbounded runner timeout')
need(type(p['cpu_runner']) is str and sha(p['cpu_runner'].encode())==p['cpu_runner_sha256'],'Runner bytes differ')
expected_modules=(
    'reproduction.common8.test_scheduler',
    'reproduction.common8.test_scheduler_collective',
    'reproduction.common8.test_configuration',
    'reproduction.common8.test_contract',
    'reproduction.common8.test_policy_calibration_plan',
    'reproduction.common8.test_pipeline_calibration',
    'reproduction.common8.test_formal_contract',
    'reproduction.common8.test_formal_validation',
    'reproduction.common8.test_fresh200_bootstrap',
    'reproduction.common8.test_adopt_fresh200_probes',
    'reproduction.common8.test_gpu_training_admission',
    'reproduction.common8.test_freeze_calibration_plan',
    'reproduction.common8.test_safety_calibration_evidence',
    'reproduction.common8.test_calibration_evaluation',
    'reproduction.common8.test_calibration_runtime',
    'reproduction.common8.test_prepare_calibration_run',
    'reproduction.common8.test_adopt_initial_probes',
    'reproduction.common8.test_production_safety_admission',
    'search_g1.test_policy_safety',
    'reproduction.common8.test_temporary_cache',
    'reproduction.common8.test_sampling_capture',
    'reproduction.common8.test_behavior_trace',
    'reproduction.common8.test_behavior_probability',
    'reproduction.common8.test_behavior_runtime',
    'reproduction.common8.test_generation_behavior')
need(type(p['modules']) is list and p['modules']==list(expected_modules) and
     type(p['module_count']) is int and p['module_count']==len(expected_modules),'Bad exact module union')
need(type(p['expected_source_count']) is int and p['expected_source_count']>0 and
     type(p['files']) is dict and type(p['source_sha256']) is dict and
     len(p['files'])==len(p['source_sha256'])==p['expected_source_count'] and
     set(p['files'])==set(p['source_sha256']),'Bad frozen complete runtime source count')
need(sha(canonical(p['source_sha256']))==p['source_map_sha256'],'Bad canonical source map')
excluded={'.git','evidence','cache','caches','__pycache__','.cache','.pytest_cache','.mypy_cache','.ruff_cache'}
versions={'verl/version/version','verl/single_controller/version/version'}
shells=('reproduction/common8/train.sh','reproduction/common8/launch_calibration_native.sh')
def runtime_name(name):
    if type(name) is not str or not name or any(ord(c) in (0,10,13,92) for c in name): return False
    path=Path(name)
    return not path.is_absolute() and '..' not in path.parts and path.as_posix()==name and not excluded.intersection(path.parts) and (path.suffix in ('.py','.sh','.yaml','.yml','.json') or name in versions)
need(versions.issubset(p['files']) and set(shells).issubset(p['files']),'Missing version/shell source')
need(all(type(module) is str and re.fullmatch('[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+',module) and
         module.replace('.','/')+'.py' in p['files'] for module in p['modules']),'Missing regression module')
decoded={};total=0
for name,data in p['files'].items():
    need(runtime_name(name),'Unsafe source path')
    need(type(data) is str and type(p['source_sha256'][name]) is str and re.fullmatch('[0-9a-f]{64}',p['source_sha256'][name]),'Bad source encoding/hash')
    raw=base64.b64decode(data,validate=True)
    need(base64.b64encode(raw).decode()==data and len(raw)<=2*1024*1024 and sha(raw)==p['source_sha256'][name],'Source payload bytes differ')
    decoded[name]=raw;total+=len(raw)
need(total<=16*1024*1024,'Source-only payload exceeds small transfer bound')
b=Path(p['base']);parent=b.parent
need(parent.is_dir() and not parent.is_symlink() and parent.resolve()==parent,'Not the canonical target parent')
b.mkdir(mode=0o700,exist_ok=False)
need(not b.is_symlink() and b.stat().st_uid==os.getuid(),'New target namespace is not owned by this runner')
repo=b/'repo';repo.mkdir(mode=0o700)
def exclusive(path,data):
    with path.open('xb') as stream:
        os.chmod(path,0o600);stream.write(data);stream.flush();os.fsync(stream.fileno())
for name,data in decoded.items():
    path=repo/name;path.parent.mkdir(parents=True,exist_ok=True)
    exclusive(path,data)
def safe_regular_bytes(path, expected_size=None, max_bytes=2 * 1024 * 1024):
    """Read bounded ordinary bytes; reject links, FIFOs and in-read replacement."""
    path = Path(path)
    require(type(max_bytes) is int and max_bytes >= 0 and
            (expected_size is None or type(expected_size) is int and 0 <= expected_size <= max_bytes),
            'Invalid regular-file read bound')
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode), 'Safe reader requires an ordinary regular file')
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, 'O_CLOEXEC', 0)
    fd = os.open(path, flags)
    try:
        opened = os.fstat(fd)
        require(stat.S_ISREG(opened.st_mode), 'Safe reader opened a nonordinary file')
        def signature(observation):
            return (observation.st_dev, observation.st_ino, observation.st_mode,
                    observation.st_size, observation.st_mtime_ns, observation.st_ctime_ns)
        require(signature(before) == signature(opened), 'Safe reader inode/file changed before open')
        require(0 <= opened.st_size <= max_bytes and
                (expected_size is None or opened.st_size == expected_size),
                'Safe reader file size differs from bound')
        limit = max_bytes if expected_size is None else expected_size
        chunks, count = [], 0
        while True:
            chunk = os.read(fd, min(65536, limit - count + 1))
            if not chunk:
                break
            count += len(chunk)
            require(count <= limit, 'Safe reader file grew beyond bounded size')
            chunks.append(chunk)
        after_fd, after_path = os.fstat(fd), path.lstat()
        require(stat.S_ISREG(after_fd.st_mode) and stat.S_ISREG(after_path.st_mode) and
                signature(opened) == signature(after_fd) == signature(after_path),
                'Safe reader inode/file changed during read')
        require(count == opened.st_size, 'Safe reader short or changed regular-file read')
        return b''.join(chunks)
    finally:
        os.close(fd)
def inventory():
    # The target repo contains only the transferred payload. Runtime filtering
    # is a local transfer rule, never a target integrity acceptance rule.
    need(not repo.is_symlink() and stat.S_ISDIR(repo.lstat().st_mode),'Target source root is not an ordinary directory')
    expected_directories={'.'}
    for name in p['files']:
        expected_directories.update(parent.as_posix() for parent in Path(name).parents)
    found={};directories_seen=set()
    def walk_error(error):
        raise error
    for root,directories,files in os.walk(repo,followlinks=False,onerror=walk_error):
        root_path=Path(root)
        need(not root_path.is_symlink() and stat.S_ISDIR(root_path.lstat().st_mode),'Target source directory link/nonordinary directory')
        directories_seen.add(root_path.relative_to(repo).as_posix())
        directories.sort()
        for name in directories:
            path=root_path/name
            need(not path.is_symlink() and stat.S_ISDIR(path.lstat().st_mode),'Target source directory link/nonordinary directory')
            relative=path.relative_to(repo).as_posix()
            directories_seen.add(relative)
            need(relative in expected_directories,'Complete target directory graph differs from transferred payload')
        for name in sorted(files):
            path=root_path/name;relative=path.relative_to(repo).as_posix()
            expected_size=len(decoded[relative]) if relative in decoded else None
            found[relative]=sha(safe_regular_bytes(path,expected_size=expected_size))
    need(directories_seen==expected_directories,'Complete target directory graph differs from transferred payload')
    return dict(sorted(found.items()))
initial=inventory();need(initial==p['source_sha256'],'Initial complete target source differs')
runner_dir=b/'runner';runner_dir.mkdir(mode=0o700)
runner=runner_dir/'cpu_runner.py';exclusive(runner,p['cpu_runner'].encode())
for name in ('tmp','cache','ray'): (b/name).mkdir(mode=0o700)
before=inventory();need(before==p['source_sha256'],'Complete target source before tests differs')
runner_size=len(p['cpu_runner'].encode())
runner_before=sha(safe_regular_bytes(runner,expected_size=runner_size))
need(runner_before==p['cpu_runner_sha256'],'Target saved runner differs')
env={key:value for key,value in os.environ.items() if not key.startswith(('SEARCH_G1_','ECO_','SLURM_')) and key not in ('PYTHONHOME','RAY_ADDRESS','PROBE_URL')}
env.update(CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(repo),PYTHONNOUSERSITE='1',
    PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',HF_HUB_OFFLINE='1',
    HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TMPDIR=str(b/'tmp'),
    XDG_CACHE_HOME=str(b/'cache'),HF_HOME=str(b/'cache/hf'),TORCH_HOME=str(b/'cache/torch'),
    TRITON_CACHE_DIR=str(b/'cache/triton'),TORCHINDUCTOR_CACHE_DIR=str(b/'cache/inductor'),
    RAY_TMPDIR=str(b/'ray'))
def group_alive(pid):
    try: os.killpg(pid,0);return True
    except ProcessLookupError: return False
def stop_owned_group(process):
    for sig in (signal.SIGTERM,signal.SIGKILL):
        try: os.killpg(process.pid,sig)
        except ProcessLookupError: break
        try: process.communicate(timeout=10)
        except subprocess.TimeoutExpired: pass
        if not group_alive(process.pid): break
    return not group_alive(process.pid)
def bounded(args,timeout):
    process=subprocess.Popen(args,cwd=repo,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                             text=True,start_new_session=True)
    timed_out=False;cleaned=False
    try: stdout,stderr=process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as error:
        timed_out=True;cleaned=stop_owned_group(process)
        def text(value): return value.decode(errors='replace') if isinstance(value,bytes) else (value or '')
        stdout,stderr=text(error.stdout),text(error.stderr)
    remaining=group_alive(process.pid)
    unexpected_group=remaining and not timed_out
    if remaining: cleaned=stop_owned_group(process)
    return dict(returncode=124 if timed_out else (125 if unexpected_group else process.returncode),
        stdout=stdout,stderr=stderr,observation_timed_out=timed_out,
        owned_process_group_cleaned=cleaned,owned_process_group_remaining=group_alive(process.pid))
started=time.time()
execution=bounded([p['target_python'],str(runner),*p['modules']],p['runner_timeout_seconds'])
syntax=[]
for filename in shells:
    check=bounded(['bash','-n',filename],10);check['path']=filename;syntax.append(check)
inventory_error=None
try: after=inventory()
except Exception as error: after={};inventory_error=type(error).__name__+': '+str(error)
runner_read_error=None
try: runner_after=sha(safe_regular_bytes(runner,expected_size=runner_size))
except Exception as error:
    runner_after=None;runner_read_error=type(error).__name__+': '+str(error)
checked=time.time()
proof=dict(schema_version=2,role=p['role'],base=str(b),repo_path=str(repo),runner_path=str(runner),
    source_commit=p['source_commit'],source_map_sha256=p['source_map_sha256'],
    source_sha256=after,source_initial_sha256=initial,source_before_tests_sha256=before,
    source_after_tests_sha256=after,source_unchanged=initial==before==after==p['source_sha256'],
    source_inventory_error=inventory_error,expected_source_count=p['expected_source_count'],
    actual_source_count=len(after),modules=p['modules'],module_count=p['module_count'],
    cpu_runner_sha256=p['cpu_runner_sha256'],cpu_runner_before_sha256=runner_before,
    cpu_runner_after_sha256=runner_after,cpu_runner_unchanged=runner_before==runner_after==p['cpu_runner_sha256'],
    runner_read_error=runner_read_error,
    target_python=p['target_python'],cuda_visible_devices='',actual_production_updates=0,
    production_admission=False,healthy_checkpoint_accepted=False,scope=p['scope'],
    started_unix=started,checked_unix=checked,duration_seconds=checked-started,
    runner_timeout_seconds=p['runner_timeout_seconds'],shell_syntax_checks=syntax,**execution)
exclusive(b/'target-cpu-result.json',canonical(proof))
print(json.dumps(proof,allow_nan=False),flush=True)
'''

PROOF_KEYS = frozenset(('schema_version', 'role', 'base', 'repo_path', 'runner_path',
    'source_commit', 'source_map_sha256', 'source_sha256', 'source_initial_sha256',
    'source_before_tests_sha256', 'source_after_tests_sha256', 'source_unchanged',
    'source_inventory_error', 'expected_source_count', 'actual_source_count', 'modules',
    'module_count', 'cpu_runner_sha256', 'cpu_runner_before_sha256', 'cpu_runner_after_sha256',
    'cpu_runner_unchanged', 'runner_read_error', 'target_python', 'cuda_visible_devices', 'actual_production_updates',
    'production_admission', 'healthy_checkpoint_accepted', 'scope', 'started_unix',
    'checked_unix', 'duration_seconds', 'runner_timeout_seconds', 'shell_syntax_checks',
    'returncode', 'stdout', 'stderr', 'observation_timed_out', 'owned_process_group_cleaned',
    'owned_process_group_remaining'))
SUMMARY_KEYS = frozenset(('tests_run', 'skipped', 'failures', 'errors', 'successful',
    'expected_failures', 'unexpected_successes', 'expected_failure_details', 'unexpected_success_details',
    'failure_details', 'error_details', 'module_test_counts', 'cuda_available',
    'cuda_initialized_before', 'cuda_initialized_after', 'cuda_visible_devices',
    'distribution_versions', 'torch_version', 'transformers_module_version', 'vllm_module_version',
    'expected_distribution_versions_matched', 'python_version', 'python_executable', 'system',
    'modules', 'runner_error', 'duration_seconds', 'output'))
EXECUTION_KEYS = frozenset(('returncode', 'stdout', 'stderr', 'observation_timed_out',
                           'owned_process_group_cleaned', 'owned_process_group_remaining'))


def execution_ok(document):
    require(type(document['returncode']) is int and document['returncode'] == 0 and
            document['observation_timed_out'] is False and
            type(document['owned_process_group_cleaned']) is bool and
            document['owned_process_group_remaining'] is False and
            type(document['stdout']) is str and type(document['stderr']) is str,
            'Target CPU test/syntax process failed or left owned children')


def extract_summary(stdout):
    require(type(stdout) is str, 'CPU stdout must be text')
    records = [line[len(SUMMARY_PREFIX):] for line in stdout.splitlines()
               if line.startswith(SUMMARY_PREFIX)]
    require(len(records) == 1, 'Exactly one actual CPU summary is required')
    return strict_json(records[0])


def validate_proof(proof, payload):
    validate_payload(payload)
    exact_keys(proof, PROOF_KEYS, 'remote CPU proof')
    require(type(proof['schema_version']) is int and proof['schema_version'] == SCHEMA_VERSION and
            proof['role'] == ROLE and proof['base'] == payload['base'] and
            proof['repo_path'] == payload['base'] + '/repo' and
            proof['runner_path'] == payload['base'] + '/runner/cpu_runner.py',
            'Target proof namespace/version/role differs')
    for key in ('source_commit', 'source_map_sha256', 'expected_source_count', 'modules',
                'module_count', 'scope', 'target_python', 'cpu_runner_sha256', 'runner_timeout_seconds'):
        require(proof[key] == payload[key] and type(proof[key]) is type(payload[key]),
                'Target proof differs from committed payload: ' + key)
    for key in ('source_sha256', 'source_initial_sha256', 'source_before_tests_sha256',
                'source_after_tests_sha256'):
        require(type(proof[key]) is dict and proof[key] == payload['source_sha256'],
                'Complete target runtime source map differs: ' + key)
    require(proof['source_unchanged'] is True and proof['source_inventory_error'] is None and
            type(proof['actual_source_count']) is int and
            proof['actual_source_count'] == payload['expected_source_count'] and
            proof['cpu_runner_before_sha256'] == proof['cpu_runner_after_sha256'] == payload['cpu_runner_sha256'] and
            proof['cpu_runner_unchanged'] is True and proof['runner_read_error'] is None,
            'Target source or file-backed runner changed')
    require(type(proof['actual_production_updates']) is int and proof['actual_production_updates'] == 0 and
            proof['production_admission'] is False and proof['healthy_checkpoint_accepted'] is False and
            proof['cuda_visible_devices'] == '', 'CPU proof scope changed')
    for key in ('started_unix', 'checked_unix', 'duration_seconds'):
        require(type(proof[key]) in (int, float) and 0 <= proof[key] < float('inf'),
                'Invalid actual target timing')
    require(proof['started_unix'] > 0 and proof['checked_unix'] >= proof['started_unix'] and
            proof['duration_seconds'] == proof['checked_unix'] - proof['started_unix'],
            'Actual target timing differs')
    execution_ok(proof)
    syntax = proof['shell_syntax_checks']
    require(type(syntax) is list and len(syntax) == len(SHELL_FILES), 'Each shell file requires its own check')
    for check, filename in zip(syntax, SHELL_FILES):
        exact_keys(check, EXECUTION_KEYS | {'path'}, 'shell syntax check')
        require(check['path'] == filename, 'Shell syntax source differs')
        execution_ok(check)
    summary = extract_summary(proof['stdout'])
    exact_keys(summary, SUMMARY_KEYS, 'actual CPU summary')
    for key in ('tests_run', 'failures', 'errors', 'expected_failures', 'unexpected_successes'):
        require(type(summary[key]) is int and summary[key] >= 0, 'Invalid actual test count')
    counts = summary['module_test_counts']
    require(type(counts) is dict and set(counts) == set(MODULES) and
            all(type(n) is int and n > 0 for n in counts.values()) and
            summary['tests_run'] > 0 and sum(counts.values()) == summary['tests_run'] and
            summary['modules'] == list(MODULES), 'Incomplete actual test module/count evidence')
    require(summary['skipped'] == [] and type(summary['skipped']) is list and
            summary['failure_details'] == summary['error_details'] == [] and
            type(summary['failure_details']) is type(summary['error_details']) is list and
            summary['failures'] == summary['errors'] == 0 and summary['successful'] is True and
            summary['expected_failures'] == summary['unexpected_successes'] == 0 and
            summary['expected_failure_details'] == summary['unexpected_success_details'] == [] and
            type(summary['expected_failure_details']) is type(summary['unexpected_success_details']) is list and
            summary['runner_error'] is None, 'Full zero-skip CPU tests not accepted')
    require(summary['cuda_available'] is False and summary['cuda_initialized_before'] is False and
            summary['cuda_initialized_after'] is False and summary['cuda_visible_devices'] == '' and
            summary['system'] == 'Linux' and summary['python_executable'] == TARGET_PYTHON and
            type(summary['python_version']) is str and bool(summary['python_version']) and
            type(summary['duration_seconds']) in (int, float) and
            0 <= summary['duration_seconds'] < float('inf') and type(summary['output']) is str,
            'Actual target CPU/CUDA/environment evidence differs')
    versions = summary['distribution_versions']
    require(type(versions) is dict and set(versions) == {'torch', 'transformers', 'vllm', 'tensordict'} and
            versions['torch'] in ('2.4.0', '2.4.0+cu121') and summary['torch_version'] == '2.4.0+cu121' and
            versions['transformers'] == summary['transformers_module_version'] == '4.46.3' and
            versions['vllm'] == '0.6.3' and versions['tensordict'] == '0.5.0' and
            type(summary['vllm_module_version']) is str and bool(summary['vllm_module_version']) and
            summary['expected_distribution_versions_matched'] is True,
            'Installed distribution/module environment versions differ')
    return summary


def exclusive_json(path, document):
    path = Path(path)
    with path.open('x') as stream:
        os.chmod(path, 0o600)
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-commit', required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true', help='Actually run one isolated target CPU copy')
    mode.add_argument('--check-inputs', action='store_true', help='Only inspect local committed sources; no RPC or writes')
    args = parser.parse_args(argv)
    base = '/dataset/shaoshitong/g1-fresh200-cpu-' + uuid.uuid4().hex
    payload = build_payload(REPO, args.source_commit, base)
    if not args.execute:
        print(json.dumps(dict(source_commit=payload['source_commit'],
            source_count=payload['expected_source_count'], source_map_sha256=payload['source_map_sha256'],
            modules=payload['modules'], module_count=payload['module_count'],
            cpu_runner_sha256=payload['cpu_runner_sha256'],
            scope='local committed-input check only; no claim, RPC or target copy'), indent=2))
        return 0
    claim_path, result_path = (PREP / (STEM + suffix) for suffix in ('-claim.json', '-result.json'))
    require(not claim_path.exists() and not claim_path.is_symlink() and
            not result_path.exists() and not result_path.is_symlink(),
            'Prior fresh200 claim/result exists: retain and inspect the same handle; never overwrite or repeat')
    claim = {key: value for key, value in payload.items() if key not in ('files', 'cpu_runner')}
    claim.update(role='fresh200_target_cpu_execution_claim', claimed_unix=time.time(),
        launcher_sha256=sha(safe_regular_bytes(Path(__file__), max_bytes=MAX_FILE_BYTES)), remote_code_sha256=sha(REMOTE_CODE.encode()),
        complete_payload_sha256=sha(canonical(payload)), target_host='h100-8')
    exclusive_json(claim_path, claim)
    sys.path.insert(0, CONTROL_DIR)
    import control
    command = 'python3 -c ' + shlex.quote(REMOTE_CODE)
    try:
        observed = subprocess.run(control.route('h100-8', command), input=canonical(payload).decode(),
            capture_output=True, text=True, timeout=RPC_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError('Target observation timed out; retain claim and inspect the same base: ' + base) from error
    require(observed.returncode == 0,
            'Target observation failed; retain claim and inspect the same base: ' + base + '\n' + observed.stderr[-1800:])
    proof = strict_json(observed.stdout)
    exclusive_json(result_path, proof)
    summary = validate_proof(proof, payload)
    display = {key: value for key, value in summary.items() if key != 'output'}
    print(json.dumps(dict(base=base, source_commit=args.source_commit, summary=display,
        result_path=str(result_path), result_sha256=sha(safe_regular_bytes(result_path, max_bytes=MAX_FILE_BYTES)), scope=SCOPE), indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
