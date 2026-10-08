"""External, explicit single-job fresh200/57 H100 formal training orchestration.

--check-inputs reads only local immutable inputs/Git blobs. --execute is the only
submission entry and imports the existing route controller lazily. A durable
intent and target submission claim prevent blind retries after an unknown SSH
outcome. This tool never grants model/health admission or resumes old state.
"""
import argparse
import base64
import datetime
import hashlib
import json
import math
import os
import re
import shlex
import stat
import subprocess
import sys
import time
import types
import uuid
from pathlib import Path

ROOT = Path('/Users/shaoshitong/Documents/Search-G1')
REPO = ROOT / 'repo'
PREP = ROOT / 'preparation'
ROLE = 'formal_inline_fresh200'
NATIVE_WALLTIME = '2-00:00:00'
OLD_BASE = '/mnt/data/shaoshitong/g1d-302569b1'
OLD_PARENT = '/mnt/data/shaoshitong/g1p-302569b1'
OLD_JOB = '3298'
OLD_RUN = '0873410ce2f146499a581314184214f9'
ASSET_ROOT = '/dataset/shaoshitong/Search-G1-common8'
EXPORT_ROOT = '/dataset/shaoshitong/search-r1-rl/tree-grpo'
SOURCE_IP = '10.33.0.13'
CANONICAL_DATA = ('manifest.json', 'train.parquet', 'validation.parquet',
                  'stage.parquet', 'final.parquet', 'exclusions-audit.json')
SHARED_ASSETS = ('env', 'bootstrap-assets', 'bootstrap-tools', 'assets', 'retriever-gpu-lib')
VERSION_FILES = ('verl/version/version', 'verl/single_controller/version/version')
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_SOURCE_BYTES = 16 * 1024 * 1024
MAX_RUNTIME_FILES = 4096
MAX_RUNTIME_PATH_BYTES = 4096
EXCLUDED_PARTS = frozenset(('.git', 'evidence', 'cache', 'caches', '__pycache__',
    '.cache', '.pytest_cache', '.mypy_cache', '.ruff_cache'))
ORIGINAL_CPU_MODULES = (
    'reproduction.common8.test_scheduler', 'reproduction.common8.test_scheduler_collective',
    'reproduction.common8.test_configuration', 'reproduction.common8.test_contract',
    'reproduction.common8.test_policy_calibration_plan', 'reproduction.common8.test_pipeline_calibration',
    'reproduction.common8.test_formal_contract', 'reproduction.common8.test_formal_validation',
    'reproduction.common8.test_fresh200_bootstrap', 'reproduction.common8.test_adopt_fresh200_probes',
    'reproduction.common8.test_gpu_training_admission', 'reproduction.common8.test_freeze_calibration_plan',
    'reproduction.common8.test_safety_calibration_evidence', 'reproduction.common8.test_calibration_evaluation',
    'reproduction.common8.test_calibration_runtime', 'reproduction.common8.test_prepare_calibration_run',
    'reproduction.common8.test_adopt_initial_probes', 'reproduction.common8.test_production_safety_admission',
    'search_g1.test_policy_safety', 'reproduction.common8.test_temporary_cache',
    'reproduction.common8.test_sampling_capture', 'reproduction.common8.test_behavior_trace',
    'reproduction.common8.test_behavior_probability', 'reproduction.common8.test_behavior_runtime',
    'reproduction.common8.test_generation_behavior')
INLINE_CPU_MODULES = ('reproduction.common8.test_formal_inline_safety',
    'reproduction.common8.test_formal_inline_evaluation',
    'reproduction.common8.test_inline_update_evidence',
    'reproduction.common8.test_recoverable_guards')
CPU_MODULES = ORIGINAL_CPU_MODULES + INLINE_CPU_MODULES
CPU_SUMMARY_PREFIX = 'FORMAL_INLINE_CPU_SUMMARY='
CPU_PRODUCER_FILE = 'test_formal_inline_target_cpu.py'
CPU_PRODUCER_TEST_FILE = 'test_formal_inline_target_cpu_runner.py'
NAMES = dict(intent='formal-inline-native-submit-intent.json',
             package='formal-inline-native-package-proof.json',
             handle='formal-inline-retrieval-export-handle.json',
             submitted='formal-inline-native-submit-receipt.json',
             launch='formal-inline-native-launch.json',
             birth='formal-inline-native-birth-observation.json',
             terminal='formal-inline-native-terminal-observation.json')
BIRTH_FIELDS = ('JobId', 'SubmitTime', 'StartTime', 'NodeList', 'BatchHost',
                'Command', 'WorkDir', 'StdOut', 'StdErr', 'Restarts',
                'UserId', 'NumNodes', 'MinMemoryNode', 'TresPerNode', 'JobName')
MATHEMATICS = dict(world_size=8, train_batch_size=512, n_agent=5, rollout_n=1,
    sequence_parallel_size=1, trajectory_count=2560, ppo_mini_batch_size=128,
    ppo_epochs=1, optimizer_steps_per_outer=20, ppo_micro_batch_size=16,
    per_rank_actor_micro_batch=2, log_prob_micro_batch_size=32,
    per_rank_log_prob_micro_batch=4, learning_rate=1e-6,
    lr_warmup_steps_ratio=0.0, learning_rate_schedule='linear_fixed_outer_warmup_then_constant',
    gradient_clip_global_l2=1.0, target_actual_outer_updates=200,
    training_start_checkpoint=None, fresh_actor=True, fixed_original_reference=True,
    split_counts=dict(train=169028, validation=512, stage=64, final=51713),
    advantage_estimator='grpo', original_state_masking=True,
    objective_reduction='original two-row micro masked_mean',
    reference_kl_coefficient=0.0, reference_kl_type='low_var_kl',
    sampling_temperature=1.0, sampling_top_p=1.0, epsilon=0.2,
    compute_dtype='float32', rollout_dtype='float32', allow_tf32=False,
    lr_warmup_steps=57, entropy_coefficient=0.0)
THRESHOLDS = dict(dual_clip=3.0, epsilon=0.2, target_old_policy_kl=0.2,
                  kl_log_bound=30.0, negative_tail_limit=0.01)
SAFETY_MODE = 'inline_first_five'
FORMAL_COUNTING = 'all20_opportunities_processed_and_at_least_one_write'
MATHEMATICS.update(policy_safety_mode=SAFETY_MODE, safeguards=dict(THRESHOLDS),
    optimizer_opportunities_per_outer=20, optimizer_completion_rule=FORMAL_COUNTING,
    optimizer_step_count_semantics='opportunities; actual successful writes reported separately')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode()


def sha(content):
    return hashlib.sha256(content).hexdigest()


def digest(value, length=64):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{' + str(length) + '}', value) is not None


def strict_json(content):
    def pairs(rows):
        result = {}
        for key, value in rows:
            need(key not in result, 'Duplicate JSON key: ' + key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Nonfinite JSON: ' + value)
    def finite_float(value):
        result = float(value)
        need(math.isfinite(result), 'Nonfinite JSON exponent')
        return result
    return json.loads(content, object_pairs_hook=pairs, parse_constant=invalid, parse_float=finite_float)


def sealed(path, expected):
    path = Path(path)
    need(path.is_absolute() and path.resolve() == path and not path.is_symlink() and
         stat.S_ISREG(path.stat().st_mode) and digest(expected), 'Explicit canonical ordinary proof/SHA required')
    content = path.read_bytes()
    need(sha(content) == expected, 'Sealed input bytes changed: ' + str(path))
    return content, strict_json(content)


def save(path, value):
    """Exclusive publication; existing or partial reservations are never reset."""
    path = Path(path)
    with path.open('xb') as stream:
        stream.write(canonical(value) + b'\n')
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def verify_mapping(mapping, count):
    need(isinstance(mapping, dict) and type(count) is int and 0 < count <= MAX_RUNTIME_FILES and len(mapping) == count,
         'Explicit complete source map/count required')
    for name, value in mapping.items():
        path = Path(name)
        need(isinstance(name, str) and name and str(path) == name and not path.is_absolute() and
             '..' not in path.parts and len(name.encode('utf-8')) <= MAX_RUNTIME_PATH_BYTES and
             not EXCLUDED_PARTS.intersection(path.parts) and
             (path.suffix in ('.py', '.sh', '.yaml', '.yml', '.json') or name in VERSION_FILES) and
             digest(value), 'Invalid runtime source mapping entry')
    need(set(VERSION_FILES) <= set(mapping), 'Both extensionless runtime version files are required')


def committed_sources(commit, mapping, count, repo, protocol, cpu_base):
    """Reuse the reviewed CPU producer's bounded NUL Git/current-byte loader.

    The sealed producer is loaded only after its independent review binding has
    been checked. This keeps exclusions, graph limits and source bytes identical
    to the actual target CPU payload, instead of a second Git implementation.
    """
    need(digest(commit, 40), 'Explicit full immutable source commit required')
    verify_mapping(mapping, count)
    payload = protocol.build_payload(repo, commit, cpu_base)
    need(payload['source_commit'] == commit and payload['source_sha256'] == mapping and
         payload['expected_source_count'] == count,
         'Declared map is not the exact complete committed and current source set')
    files = {name: base64.b64decode(encoded, validate=True)
             for name, encoded in payload['files'].items()}
    need({'reproduction/common8/launch_native.sh',
          'reproduction/common8/formal_inline_safety.py'} <= set(files), 'New direct formal inline runtime missing')
    budget = strict_json(files['reproduction/common8/budget.json'])
    need(budget.get('contract_kind') == 'fresh200_warmup57' and budget.get('policy_safety_mode') == SAFETY_MODE and
         all(canonical(budget.get(name)) == canonical(value) for name, value in MATHEMATICS.items()),
         'Committed scientific fresh200/57 budget differs')
    return files


def verify_cpu_modules(modules, mapping):
    need(type(modules) is list and modules == list(CPU_MODULES),
         'Exact ordered formal inline CPU union must contain old25 plus all four new suites')
    need(all(isinstance(module, str) and re.fullmatch(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+', module) and
             module.replace('.', '/') + '.py' in mapping for module in modules),
         'Every actual tested suite must exist in this full future source map')


def load_cpu_protocol(expected_sha256):
    """Load reviewed SHA-sealed stdlib protocol bytes, without controller/RPC."""
    path = Path(__file__).parent / CPU_PRODUCER_FILE
    need(path.is_file() and not path.is_symlink() and digest(expected_sha256),
         'Reviewed same-directory formal inline CPU producer SHA required')
    content = path.read_bytes()
    need(len(content) <= MAX_FILE_BYTES and sha(content) == expected_sha256,
         'Reviewed CPU producer bytes changed')
    module = types.ModuleType('formal_inline_cpu_protocol')
    module.__file__ = str(path)
    exec(compile(content, str(path), 'exec'), module.__dict__)
    need(tuple(module.MODULES) == CPU_MODULES and module.SUMMARY_PREFIX == CPU_SUMMARY_PREFIX and
         module.ROLE == 'formal_inline_target_cpu_regression' and module.SCHEMA_VERSION == 2 and
         tuple(module.SHELL_FILES) == ('reproduction/common8/train.sh',
             'reproduction/common8/launch_calibration_native.sh', 'reproduction/common8/launch_native.sh'),
         'Reviewed producer is not the exact new formal inline protocol')
    return module


def cpu_payload(protocol, proof, commit, mapping, files, required_modules):
    verify_cpu_modules(required_modules, mapping)
    need(set(files) == set(mapping) and all(sha(files[name]) == mapping[name] for name in files),
         'CPU protocol reconstruction must use exact bound committed bytes')
    payload = dict(schema_version=protocol.SCHEMA_VERSION, role=protocol.ROLE,
        base=proof['base'], source_commit=commit, expected_source_count=len(mapping),
        source_sha256=mapping, source_map_sha256=sha(protocol.canonical(mapping)),
        files={name: base64.b64encode(files[name]).decode() for name in files},
        modules=required_modules, module_count=len(required_modules), scope=protocol.SCOPE,
        target_python=protocol.TARGET_PYTHON, cpu_runner=protocol.CPU_RUNNER,
        cpu_runner_sha256=sha(protocol.CPU_RUNNER.encode()),
        runner_timeout_seconds=protocol.CPU_TIMEOUT_SECONDS)
    return protocol.validate_payload(payload)


def verify_cpu(proof, commit, mapping, required_modules, files, execution_claim, protocol, producer_sha256):
    payload = cpu_payload(protocol, proof, commit, mapping, files, required_modules)
    summary = protocol.validate_proof(proof, payload)
    expected = {key: value for key, value in payload.items() if key not in ('files', 'cpu_runner')}
    expected.update(role='formal_inline_target_cpu_execution_claim',
        launcher_sha256=producer_sha256, remote_code_sha256=sha(protocol.REMOTE_CODE.encode()),
        complete_payload_sha256=sha(protocol.canonical(payload)), target_host='h100-8')
    need(type(execution_claim) is dict and set(execution_claim) == set(expected) | {'claimed_unix'} and
         all(type(execution_claim[key]) is type(value) and execution_claim[key] == value
             for key, value in expected.items()),
         'Actual formal inline CPU execution claim does not bind this full typed payload/producer/runner/source')
    claimed = execution_claim['claimed_unix']
    need(type(claimed) in (int, float) and math.isfinite(claimed) and
         0 < claimed <= proof['started_unix'] <= proof['checked_unix'],
         'Actual CPU claim/target execution timeline differs')
    return summary


def verify_method(method, commit, mapping):
    need(type(method['schema_version']) is int and method['schema_version'] == 1 and
         method['role'] == 'formal_inline_training_method' and
         method['source_commit'] == commit and method['source_map_sha256'] == sha(canonical(mapping)) and
         type(method['created_unix']) in (int, float) and math.isfinite(method['created_unix']) and
         0 < method['created_unix'] <= time.time() and
         type(method['target_updates']) is int and method['target_updates'] == 200 and
         type(method['warmup_steps']) is int and method['warmup_steps'] == 57 and
         method['safeguards'] == THRESHOLDS and method['policy_safety_mode'] == SAFETY_MODE and
         method['same_process_after_five'] is True and method['fresh_states'] is True and
         method['optimizer_counting'] == FORMAL_COUNTING and
         method['negative_tail_semantics'] == 'prospective operational candidate; not calibrated health',
         'Wrong prospective direct formal200/57/KL0.2/tail0.01/same-process method')


def job_fields(raw):
    need(isinstance(raw, str) and len(raw.strip().splitlines()) == 1, 'Missing unique raw native scheduler record')
    result = {}
    for token in raw.split():
        need('=' in token, 'Invalid native scheduler token')
        key, value = token.split('=', 1)
        need(key and key not in result, 'Duplicate native scheduler key')
        result[key] = value
    return result


def verify_old_failure(launch, observed, source_cleanup):
    """Failed old science is only a nonduplication/cleanup prerequisite."""
    need(launch['job_id'] == OLD_JOB and launch['base'] == OLD_BASE and launch['parent_base'] == OLD_PARENT and
         observed['job_id'] == OLD_JOB and observed['base'] == OLD_BASE and
         observed['run-manifest.json']['run_id'] == OLD_RUN and
         observed['pipeline-status.json']['run_id'] == OLD_RUN and
         observed['pipeline-status.json']['stage'] == 'FAILED' and
         observed['cleanup-status.json']['run_id'] == OLD_RUN and observed['cleanup-status.json']['cleaned'] is True,
         'Actual old3298 failure and owned cleanup are required; no success inference')
    jobs = observed['process-ledger.json']['jobs']
    need(jobs and all(row['state'] == 'STOPPED' for row in jobs.values()), 'Old owned components are not stopped')
    if 'actual_active_namespace_processes' in observed:
        need(observed['actual_active_namespace_processes'] == [], 'Old observed actual namespace is not empty')
    # Older readonly receipt schemas omitted the /proc census. Every execute
    # must independently obtain the current empty census before any mutation.
    need(source_cleanup['cleaned'] is True and source_cleanup['handle']['export'].startswith(
        '/dataset/shaoshitong/g1-original-export-discovery-v6-') and
        digest(source_cleanup['handle']['source_sha256']), 'Old independent source service cleanup missing')
    if observed['job_returncode'] == 0:
        fields = job_fields(observed['job_stdout'])
        need(fields['JobId'] == OLD_JOB and fields['JobState'] == 'FAILED', 'Old actual native job is not terminal failure')
    else:
        need(observed['job_returncode'] == 1 and 'Invalid job id specified' in observed['job_stderr'] and
             not observed['job_stdout'], 'Only captured failed/cleaned old scope may tolerate scheduler purge')


def verify_review(review, commit, mapping, cpu_sha, method_sha, directory, producer_sha, cpu_claim_sha):
    need(review['role'] == 'formal_inline_preexecution_review' and review['approved'] is True and
         review['source_commit'] == commit and review['source_sha256'] == mapping and
         review['target_cpu_proof_sha256'] == cpu_sha and review['method_sha256'] == method_sha and
         review['cpu_producer_sha256'] == producer_sha and review['cpu_execution_claim_sha256'] == cpu_claim_sha and
         review['blocking_findings'] == [] and review['scope'] == 'staticonly' and
         review['physical_execution_authenticated'] is False and review['production_admission'] is False and
         review['healthy_checkpoint_accepted'] is False and digest(review['actualreview_report_sha256']),
         'Independent exact-source preexecution review required')
    report = Path(review['actualreview_report_path'])
    need(report.is_absolute() and report.resolve() == report and report.is_file() and not report.is_symlink() and
         sha(report.read_bytes()) == review['actualreview_report_sha256'], 'Actual independent review report bytes changed')
    expected = {'submit_formal_inline_native.py', 'test_submit_formal_inline_native.py',
                CPU_PRODUCER_FILE, CPU_PRODUCER_TEST_FILE}
    need(set(review['external_file_sha256']) == expected and review['external_file_sha256'][CPU_PRODUCER_FILE] == producer_sha,
         'Submitter/tests and the exact reviewed CPU producer must all be bound')
    for name, value in review['external_file_sha256'].items():
        path = directory / name
        need(path.is_file() and not path.is_symlink() and digest(value) and sha(path.read_bytes()) == value,
             'Reviewed external submitter/test bytes changed')


def verify_walltime(value):
    need(type(value) is str and value == NATIVE_WALLTIME,
         'This frozen native workflow requires exactly48hours (2-00:00:00)')


def load_inputs(args, repo=REPO):
    """Pure local validation; no controller import, writes or remote calls."""
    mapping_bytes, mapping = sealed(args.source_map, args.source_map_sha256)
    modules_bytes, modules = sealed(args.cpu_modules, args.cpu_modules_sha256)
    verify_cpu_modules(modules, mapping)
    cpu_bytes, cpu = sealed(args.target_cpu_proof, args.target_cpu_proof_sha256)
    claim_bytes, execution_claim = sealed(args.cpu_claim, args.cpu_claim_sha256)
    method_bytes, method = sealed(args.method, args.method_sha256)
    verify_method(method, args.source_commit, mapping)
    review_bytes, review = sealed(args.review_proof, args.review_proof_sha256)
    verify_review(review, args.source_commit, mapping, sha(cpu_bytes), sha(method_bytes), Path(__file__).parent,
                  args.cpu_producer_sha256, sha(claim_bytes))
    protocol = load_cpu_protocol(args.cpu_producer_sha256)
    files = committed_sources(args.source_commit, mapping, args.source_count, repo, protocol, cpu['base'])
    summary = verify_cpu(cpu, args.source_commit, mapping, modules, files, execution_claim,
                         protocol, args.cpu_producer_sha256)
    _, old_launch = sealed(args.previous_launch, args.previous_launch_sha256)
    _, old_observed = sealed(args.previous_observation, args.previous_observation_sha256)
    _, cleanup = sealed(args.previous_source_cleanup, args.previous_source_cleanup_sha256)
    verify_old_failure(old_launch, old_observed, cleanup)
    need(type(args.export_port) is int and 1024 <= args.export_port <= 65535 and args.export_port not in (19874, 19876),
         'Explicit new independent retrieval-export port required')
    verify_walltime(args.time_limit)
    return dict(source_commit=args.source_commit, source_sha256=mapping, source_count=len(mapping),
        source_map_file_sha256=sha(mapping_bytes), source_map_sha256=sha(canonical(mapping)),
        target_cpu_proof_sha256=sha(cpu_bytes), target_cpu_proof_path=str(Path(args.target_cpu_proof)),
        actual_target_cpu_tests=summary['tests_run'], actual_cpu_modules=cpu['modules'],
        cpu_producer_sha256=args.cpu_producer_sha256, cpu_execution_claim_sha256=sha(claim_bytes),
        cpu_modules_file_sha256=sha(modules_bytes),
        method_sha256=sha(method_bytes), review_proof_sha256=sha(review_bytes),
        previous_launch_sha256=args.previous_launch_sha256,
        previous_observation_sha256=args.previous_observation_sha256,
        previous_source_cleanup_sha256=args.previous_source_cleanup_sha256,
        old_source_handle=cleanup['handle'], export_port=args.export_port, time_limit=args.time_limit,
        execution_role=ROLE, actual_production_updates=0,
        production_admission=False, healthy_checkpoint_accepted=False), files, method_bytes


def ray_budget(base):
    path = str(base) + '/ray/session_9999-12-31_23-59-59_999999_9999999999/sockets/plasma_store'
    need(len(path.encode()) <= 107, 'Conservative installed Ray AF_UNIX path exceeds107 bytes')
    return dict(worst_plasma_socket_path=path, worst_plasma_socket_bytes=len(path.encode()), af_unix_limit_bytes=107)


def exporter_bytes(files, export, port):
    source = files['reproduction/common8/assets_export.py'].decode()
    replacements = {
        "BASE=Path('/dataset/shaoshitong/g1-common8-original-retrieval-export')": 'BASE=Path(' + repr(export) + ')',
        "with Server(('10.33.0.13',19874),Handler)": "with Server(('10.33.0.13'," + str(port) + '),Handler)',
        'port=19874': 'port=' + str(port)}
    for old, new in replacements.items():
        need(source.count(old) == 1, 'Committed bounded asset-exporter API changed')
        source = source.replace(old, new)
    need("ROOT=Path('" + EXPORT_ROOT + "')" in source, 'Original H100 retrieval asset root changed')
    return source.encode()


def remote(host, code, payload, timeout=90, *, producer_sha256):
    """Use the SHA-bound CPU producer's owned bounded RPC capture."""
    need(host in ('h100-8', 'h100-9'), 'Only the declared target/source H100 hosts are allowed')
    protocol = load_cpu_protocol(producer_sha256)
    content = canonical(payload)
    need(len(content) <= 2 * MAX_SOURCE_BYTES, 'Source-only RPC input byte budget exceeded')
    sys.path.insert(0, '/Users/shaoshitong/Documents/remote_codex_control')
    import control
    command = 'python3 -c ' + shlex.quote(code)
    try:
        result = protocol.bounded_capture(control.route(host, command), timeout,
            2 * MAX_FILE_BYTES, 64 * 1024, input=content)
    except Exception as error:
        raise RuntimeError('Remote observation unknown; retain the exact intent/claim and observe, never resubmit') from error
    need(result.returncode == 0, 'Remote operation failed; preserve this intent and observe; do not resubmit: ' + result.stderr[-2000:])
    return strict_json(result.stdout)


OLD_READONLY_CODE = r'''
import json,os,subprocess,sys,time
from pathlib import Path
p=json.load(sys.stdin);b=Path('/mnt/data/shaoshitong/g1d-302569b1')
run=json.loads((b/'logs/run-manifest.json').read_text());assert run['run_id']=='0873410ce2f146499a581314184214f9'
stage=json.loads((b/'logs/pipeline-status.json').read_text());cleanup=json.loads((b/'logs/cleanup-status.json').read_text())
ledger=json.loads((b/'logs/process-ledger.json').read_text())
assert stage['run_id']==cleanup['run_id']==ledger['run_id']==run['run_id'] and stage['stage']=='FAILED' and cleanup['cleaned'] is True
assert ledger['jobs'] and all(row['state']=='STOPPED' for row in ledger['jobs'].values())
active=[];marker=('SEARCH_G1_BASE='+str(b)).encode()
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  if proc.stat().st_uid!=os.getuid():continue
  row=(proc/'stat').read_text().rsplit(')',1)[1].split()
  if row[0]=='Z':continue
  if marker in (proc/'environ').read_bytes().split(b'\0'):active.append(int(proc.name))
 except (FileNotFoundError,ProcessLookupError):pass
assert active==[]
for component in ledger['jobs'].values():
 for identity in component['ledger'].values():
  proc=Path('/proc')/str(identity['pid'])
  try:
   row=(proc/'stat').read_text().rsplit(')',1)[1].split()
   assert row[0]=='Z' or (int(row[19]),int(row[3]),proc.stat().st_uid)!=(identity['start_ticks'],identity['sid'],identity['uid']), 'Actual old owned PID birth is still live'
  except (FileNotFoundError,ProcessLookupError):pass
queue=subprocess.run(['squeue','-h','-j','3298','-o','%i|%T|%j'],capture_output=True,text=True,timeout=15)
assert queue.returncode==0 and not queue.stdout.strip()
claim=b/'logs/retrieval-staging-claim.json'
if claim.exists():
 value=json.loads(claim.read_text());assert value['run_id']==run['run_id'] and value['slurm_job_id']=='3298'
 root=Path(value['root']);assert root==Path('/dev/shm/search-g1-common8-'+run['run_id']+'-3298') and not root.exists() and not root.is_symlink()
print(json.dumps(dict(base=str(b),run_id=run['run_id'],job_id='3298',stage='FAILED',owned_cleanup=True,actual_active_namespace_processes=active,checked_unix=time.time(),scope='Read-only nonduplication; not successful old diagnostic admission')))
'''

PREPARE_CODE = r'''
import base64,hashlib,json,os,shutil,stat,sys,time
from pathlib import Path
p=json.load(sys.stdin);anchor=Path('/mnt/data/shaoshitong/g1p-302569b1');shared=Path('/dataset/shaoshitong/Search-G1-common8')
def sync(directory):
 fd=os.open(str(directory),os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
def write(path,content):
 with path.open('xb') as f:f.write(content);f.flush();os.fsync(f.fileno())
 sync(path.parent)
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
assert set(p['files'])==set(p['source_sha256']) and len(p['files'])==p['source_count']
b=Path(p['base']);assert b.resolve()==b and b.parent==Path('/mnt/data/shaoshitong') and b.name.startswith('g1f-') and not b.exists() and not b.is_symlink()
b.mkdir(exist_ok=False);sync(b.parent)
for name in ('repo','data','preparation','logs','tmp','ray','artifacts','cache/triton','cache/inductor','cache/hf'):(b/name).mkdir(parents=True,exist_ok=True)
write(b/'preparation/namespace-owner.json',json.dumps(dict(base=str(b),execution_role=p['execution_role'],created_unix=p['created_unix'],uid=os.getuid(),source_commit=p['source_commit'])).encode())
for name in ('env','bootstrap-assets','bootstrap-tools','assets','retriever-gpu-lib'):
 source=anchor/name;expected=shared/name;assert source.is_dir() and expected.is_dir() and source.resolve()==expected.resolve()
 (b/name).symlink_to(expected.resolve(),target_is_directory=True)
for name,encoded in p['files'].items():
 relative=Path(name);assert str(relative)==name and not relative.is_absolute() and '..' not in relative.parts
 target=b/'repo'/relative;target.parent.mkdir(parents=True,exist_ok=True);content=base64.b64decode(encoded,validate=True)
 assert hashlib.sha256(content).hexdigest()==p['source_sha256'][name];write(target,content)
data_hashes={}
for name in ('manifest.json','train.parquet','validation.parquet','stage.parquet','final.parquet','exclusions-audit.json'):
 source=anchor/'data'/name;assert source.is_file() and not source.is_symlink() and source.resolve()==source
 target=b/'data'/name;shutil.copy2(source,target)
 assert digest(source)==digest(target);data_hashes[name]=digest(target)
 with target.open('rb') as f:os.fsync(f.fileno())
assert data_hashes['manifest.json']=='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
original=anchor/'preparation/official-model-manifest-original.json';assert original.is_file() and not original.is_symlink()
write(b/'preparation/official-model-manifest-original.json',original.read_bytes())
content=base64.b64decode(p['method_bytes'],validate=True);assert hashlib.sha256(content).hexdigest()==p['method_sha256']
write(b/'preparation/formal-inline-method.json',content)
write(b/'preparation/source-package-intent.json',json.dumps({k:v for k,v in p.items() if k not in ('files','method_bytes')},allow_nan=False).encode())
sync(b/'data');sync(b/'repo');sync(b)
print(json.dumps(dict(base=p['base'],execution_role=p['execution_role'],source_commit=p['source_commit'],source_sha256=p['source_sha256'],source_count=p['source_count'],data_sha256=data_hashes,method_sha256=p['method_sha256'],checked_unix=time.time(),scope='Only this single new source namespace and original six canonical byte copies; native preparation validates science/model/data; no old state copied')))
'''

START_SOURCE_CODE = r'''
import base64,hashlib,json,os,socket,subprocess,sys,time
from pathlib import Path
p=json.load(sys.stdin);old=p['old_source_handle'];proc=Path('/proc')/str(old['pid'])
if proc.exists():
 row=(proc/'stat').read_text().rsplit(')',1)[1].split()
 assert row[0]=='Z' or (int(row[19]),int(row[3]),proc.stat().st_uid)!=(old['start_ticks'],old['sid'],old['uid']), 'Old source birth is still active; never stop or reuse it'
b=Path(p['export_base']);assert b.parent==Path('/dataset/shaoshitong') and b.resolve()==b and not b.exists() and not b.is_symlink()
with socket.socket() as sock:sock.bind(('10.33.0.13',p['export_port']))
b.mkdir(mode=0o700,exist_ok=False);source=b/'serve_original_retrieval.py';content=base64.b64decode(p['source'],validate=True)
assert hashlib.sha256(content).hexdigest()==p['export_source_sha256']
def write(path,value):
 with path.open('xb') as f:f.write(value);f.flush();os.fsync(f.fileno())
 fd=os.open(str(path.parent),os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
write(source,content);write(b/'start-claim.json',json.dumps(dict(export=str(b),source_sha256=p['export_source_sha256'],port=p['export_port'],uid=os.getuid(),created_unix=time.time())).encode())
env=dict(os.environ,CUDA_VISIBLE_DEVICES='');env.pop('LD_LIBRARY_PATH',None)
with (b/'server.log').open('x') as log:process=subprocess.Popen(['nice','-n','19','python3',str(source)],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
row=Path('/proc/'+str(process.pid)+'/stat').read_text().rsplit(')',1)[1].split()
write(b/'launcher-process.json',json.dumps(dict(pid=process.pid,start_ticks=int(row[19]),sid=int(row[3]),uid=os.getuid(),source_sha256=p['export_source_sha256'])).encode())
handle=None
for _ in range(100):
 assert process.poll() is None
 try:handle=json.loads((b/'handle.json').read_text());break
 except (FileNotFoundError,json.JSONDecodeError):time.sleep(.1)
assert handle and handle['pid']==process.pid and handle['port']==p['export_port']
handle.update(export=str(b),source_sha256=p['export_source_sha256']);print(json.dumps(handle))
'''

NATIVE_IDENTITY_CODE = r'''
import datetime,hashlib,json,os,re,subprocess,sys,time
from pathlib import Path
p=json.load(sys.stdin);b=Path(p['base']);prep=b/'preparation'
def write(path,value):
 with path.open('x') as f:json.dump(value,f,allow_nan=False);f.flush();os.fsync(f.fileno())
 fd=os.open(str(path.parent),os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
def fields(raw):
 assert len(raw.strip().splitlines())==1
 result={}
 for token in raw.split():
  assert '=' in token
  key,value=token.split('=',1);assert key and key not in result;result[key]=value
 return result
def argv():
 return ['sbatch','--parsable','--partition=gpu','--nodelist=h100-8','--nodes=1','--ntasks=1','--cpus-per-task=64','--gres=gpu:h100:8','--mem=512G','--exclusive','--time='+p['time_limit'],'--job-name='+p['job_name'],'--chdir='+str(b/'repo'),'--output='+str(b/'logs/native-formal-inline-%j.log'),'--export=ALL,SEARCH_G1_BASE='+str(b)+',SEARCH_G1_FORMAL_INLINE_METHOD='+str(prep/'formal-inline-method.json')+',SEARCH_G1_SOURCE_COMMIT='+p['source_commit']+',SEARCH_G1_EXPORT_PORT='+str(p['export_port']),str(b/'repo/reproduction/common8/launch_native.sh')]
def ordinary(path):
 assert path.is_file() and not path.is_symlink() and path.stat().st_uid==os.getuid()
 return json.loads(path.read_text())
def namespace():
 assert b.resolve()==b and b.parent==Path('/mnt/data/shaoshitong') and re.fullmatch('g1f-[0-9a-f]{8}',b.name)
 owner=ordinary(prep/'namespace-owner.json');package=ordinary(prep/'source-package-intent.json')
 assert owner['base']==package['base']==str(b) and owner['uid']==os.getuid()
 assert owner['execution_role']==package['execution_role']==p['execution_role']=='formal_inline_fresh200'
 assert owner['source_commit']==package['source_commit']==p['source_commit']
 assert all(package[key]==p[key] for key in ('source_sha256','source_count','method_sha256','target_cpu_proof_sha256','job_name','time_limit','export_port','export_base','export_source_sha256'))
 assert hashlib.sha256((prep/'formal-inline-method.json').read_bytes()).hexdigest()==p['method_sha256']
 assert len(p['source_sha256'])==p['source_count']
 for name,digest in p['source_sha256'].items():assert hashlib.sha256((b/'repo'/name).read_bytes()).hexdigest()==digest
def claim():
 value=ordinary(prep/'native-submit-claim.json')
 assert value['uid']==os.getuid() and value['argv']==argv()
 assert all(value[key]==p[key] for key in ('base','execution_role','source_commit','source_sha256','source_count','method_sha256','target_cpu_proof_sha256','job_name'))
 return value
def census():
 active=[];marker=('SEARCH_G1_BASE='+str(b)).encode()
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit():continue
  try:
   if proc.stat().st_uid!=os.getuid():continue
   row=(proc/'stat').read_text().rsplit(')',1)[1].split()
   args=(proc/'cmdline').read_bytes().split(b'\0');prefix=str(b).encode()
   owned=(marker in (proc/'environ').read_bytes().split(b'\0') or any(arg==prefix or arg.startswith(prefix+b'/') for arg in args))
   if row[0]!='Z' and owned:active.append(dict(pid=int(proc.name),start_ticks=int(row[19]),sid=int(row[3]),uid=proc.stat().st_uid))
  except (FileNotFoundError,ProcessLookupError):pass
 return active
def candidates():
 records=subprocess.run(['scontrol','show','job','-o'],capture_output=True,text=True,timeout=15)
 queue=subprocess.run(['squeue','-h','-n',p['job_name'],'-o','%i|%j'],capture_output=True,text=True,timeout=15)
 assert records.returncode==queue.returncode==0, 'Actual readonly scheduler census failed'
 matched=[]
 for line in records.stdout.strip().splitlines():
  if line=='No jobs in the system':continue
  if not re.search(r'(?:^| )JobName='+re.escape(p['job_name'])+r'(?: |$)',line):continue
  row=fields(line)
  if row.get('JobName')==p['job_name']:matched.append((row,line))
 queued=[]
 for line in queue.stdout.strip().splitlines():
  job,name=line.split('|');assert job.isdigit() and name==p['job_name'];queued.append(job)
 assert set(queued)<={row['JobId'] for row,raw in matched}, 'Scheduler census race; preserve unknown outcome'
 return matched,dict(scontrol_stdout=records.stdout,squeue_stdout=queue.stdout,checked_unix=time.time())
def bind_candidate(row,raw,entered):
 assert re.fullmatch('[1-9][0-9]*',row['JobId']) and row['JobName']==p['job_name']
 assert row['Command']==str(b/'repo/reproduction/common8/launch_native.sh') and row['WorkDir']==str(b/'repo')
 assert re.fullmatch(r'[^()]+\('+str(os.getuid())+r'\)',row['UserId'])
 assert row['Partition']=='gpu' and row['ReqNodeList']=='h100-8' and row['NumNodes']=='1'
 assert row['MinMemoryNode']=='512G' and row['TresPerNode']=='gres:gpu:h100:8'
 assert row['StdOut']==row['StdErr']==str(b/'logs')+'/native-formal-inline-'+row['JobId']+'.log'
 assert re.fullmatch(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d',row['SubmitTime'])
 stamp=datetime.datetime.fromisoformat(row['SubmitTime']).timestamp()
 assert entered['submitted_unix']-2<=stamp<=entered['deadline_unix']+2, 'Job SubmitTime is outside this reserved sbatch call'
 return dict(base=str(b),execution_role=p['execution_role'],job_name=p['job_name'],source_commit=p['source_commit'],source_sha256=p['source_sha256'],source_count=p['source_count'],method_sha256=p['method_sha256'],target_cpu_proof_sha256=p['target_cpu_proof_sha256'],argv=argv(),returncode=None,job_id=row['JobId'],submitted_unix=entered['submitted_unix'],observed_unix=time.time(),submission_authentication='actual_scheduler_claim_recovery',recovered_native_stdout=raw,recovered_native_fields=row,claim_sha256=hashlib.sha256((prep/'native-submit-claim.json').read_bytes()).hexdigest(),disposition_sha256=hashlib.sha256((prep/'native-submit-disposition.json').read_bytes()).hexdigest(),scope='Actual scheduler-bound reserved submission recovery; not terminal/model/health success')
'''

SUBMIT_CODE = NATIVE_IDENTITY_CODE + r'''
import urllib.request
namespace();assert not (b/'logs/run-manifest.json').exists()
args=argv()
value={key:p[key] for key in ('base','execution_role','source_commit','source_sha256','source_count','method_sha256','target_cpu_proof_sha256','job_name')}
value.update(argv=args,uid=os.getuid(),created_unix=time.time())
write(prep/'native-submit-claim.json',value)
write(prep/'retrieval-source-handle.json',p['handle'])
def prechecks():
 partition=subprocess.run(['scontrol','show','partition','gpu','-o'],capture_output=True,text=True,timeout=15)
 assert partition.returncode==0 and len(partition.stdout.strip().splitlines())==1, 'Actual native partition readonly validation failed'
 part=fields(partition.stdout);assert part['PartitionName']=='gpu' and part['State']=='UP'
 def seconds(value):
  match=re.fullmatch(r'(?:(\d+)-)?(\d+):(\d\d):(\d\d)',value);assert match
  day,hour,minute,second=match.groups();assert int(minute)<60 and int(second)<60
  return int(day or 0)*86400+int(hour)*3600+int(minute)*60+int(second)
 assert part['MaxTime']=='UNLIMITED' or seconds(p['time_limit'])<=seconds(part['MaxTime']), 'Requested one-job full200 walltime exceeds actual partition'
 opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
 for name,size in p['handle']['files'].items():
  with opener.open(urllib.request.Request('http://10.33.0.13:'+str(p['export_port'])+'/'+name,method='HEAD'),timeout=10) as response:assert response.status==200 and int(response.headers['Content-Length'])==size
 return partition.stdout
try:partition_stdout=prechecks()
except Exception as error:
 write(prep/'native-precheck-failed.json',dict(base=str(b),uid=os.getuid(),source_commit=p['source_commit'],stage='FAILED_BEFORE_SBATCH',error_type=type(error).__name__,checked_unix=time.time()))
 raise
submitted=time.time()
entered=dict(state='sbatch_entered',uid=os.getuid(),base=str(b),argv=args,claim_sha256=hashlib.sha256((prep/'native-submit-claim.json').read_bytes()).hexdigest(),submitted_unix=submitted,deadline_unix=submitted+60)
# Mutually exclusive with a verified never-submitted cleanup reservation.
write(prep/'native-submit-disposition.json',entered)
try:result=subprocess.run(args,capture_output=True,text=True,timeout=60)
except subprocess.TimeoutExpired as error:
 write(prep/'native-submit-unknown.json',dict(base=str(b),state='UNKNOWN_AFTER_SBATCH_ENTERED',submitted_unix=submitted,observed_unix=time.time(),error_type=type(error).__name__))
 raise
proof=dict(base=str(b),execution_role='formal_inline_fresh200',job_name=p['job_name'],source_commit=p['source_commit'],source_sha256=p['source_sha256'],source_count=p['source_count'],method_sha256=p['method_sha256'],target_cpu_proof_sha256=p['target_cpu_proof_sha256'],argv=args,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,submitted_unix=submitted,observed_unix=time.time(),partition_readonly_stdout=partition_stdout,actual_production_updates=0,submission_authentication='actual_sbatch_receipt',scope='One formal job1..5 then same-process6..200; submission alone is not physical training admission or success')
if result.returncode==0:
 proof['job_id']=result.stdout.strip().split(';')[0];assert re.fullmatch('[1-9][0-9]*',proof['job_id'])
write(prep/'native-submit-receipt.json',proof)
assert result.returncode==0,result.stderr
print(json.dumps(proof))
'''

OBSERVE_CODE = NATIVE_IDENTITY_CODE + r'''
namespace();out=dict(base=str(b),checked_unix=time.time(),execution_role='formal_inline_fresh200',submission_outcome_known=False)
receipt=prep/'native-submit-receipt.json';recovered=prep/'native-submit-recovered-receipt.json';job=None
if receipt.exists():
 value=ordinary(receipt)
 if value.get('returncode')==0 and value.get('job_id'):
  assert value['argv']==argv() and all(value[key]==p[key] for key in ('base','execution_role','source_commit','source_sha256','method_sha256','job_name'))
  out['submission']=value;job=value['job_id'];out['submission_outcome_known']=True
if not job and recovered.exists():
 value=ordinary(recovered);assert value['submission_authentication']=='actual_scheduler_claim_recovery'
 saved_claim=claim();entered=ordinary(prep/'native-submit-disposition.json')
 assert entered['state']=='sbatch_entered' and entered['uid']==os.getuid() and entered['argv']==argv()
 assert value['claim_sha256']==hashlib.sha256((prep/'native-submit-claim.json').read_bytes()).hexdigest()
 assert value['disposition_sha256']==hashlib.sha256((prep/'native-submit-disposition.json').read_bytes()).hexdigest()
 assert bind_candidate(value['recovered_native_fields'],value['recovered_native_stdout'],entered)['job_id']==value['job_id']
 out['submission']=value;job=value['job_id'];out['submission_outcome_known']=True
if not job:
 matches,scan=candidates();out['scheduler_claim_census']=scan
 disposition=prep/'native-submit-disposition.json'
 if disposition.exists():
  entered=ordinary(disposition);out['submission_disposition']=entered
  if entered['state']=='sbatch_entered':
   saved_claim=claim();assert entered['uid']==os.getuid() and entered['base']==str(b) and entered['argv']==argv()
   assert entered['claim_sha256']==hashlib.sha256((prep/'native-submit-claim.json').read_bytes()).hexdigest()
   assert len(matches)<=1, 'Ambiguous reserved native job name; never resubmit or guess a handle'
   if matches:
    value=bind_candidate(matches[0][0],matches[0][1],entered)
    write(recovered,value);out['submission']=value;job=value['job_id'];out['submission_outcome_known']=True
   else:out['submission_state']='UNKNOWN_OR_PURGED_AFTER_SBATCH_ENTERED'
  else:
   assert entered['state']=='never_submitted_cleanup' and not matches
   out['submission_state']='VERIFIED_NEVER_SUBMITTED_CLEANUP_RESERVED'
 else:
  assert not matches, 'Native candidate without entered claim; preserve unknown outcome'
  out['submission_state']='SBATCH_NOT_ENTERED_CLEANUP_RESERVATION_REQUIRED'
if job:
 out['job_id']=job;r=subprocess.run(['scontrol','show','job','-o',job],capture_output=True,text=True,timeout=15)
 out.update(job_stdout=r.stdout,job_returncode=r.returncode,job_stderr=r.stderr)
 r=subprocess.run(['squeue','-h','-j',job,'-o','%i|%T|%j'],capture_output=True,text=True,timeout=15);out.update(queue_returncode=r.returncode,queue_stdout=r.stdout,queue_stderr=r.stderr)
out['source_hashes_match']=all(hashlib.sha256((b/'repo'/name).read_bytes()).hexdigest()==digest for name,digest in p['source_sha256'].items())
out['method_sha256']=hashlib.sha256((b/'preparation/formal-inline-method.json').read_bytes()).hexdigest()
for name in ('run-manifest.json','pipeline-status.json','cleanup-status.json','process-ledger.json','retrieval-staging-status.json','policy-safety-calibration.json','five-updates-acceptance.json','full-run-acceptance.json'):
 path=b/'logs'/name
 if path.exists():out[name]=json.loads(path.read_text());out[name+'-sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
out['actual_active_namespace_processes']=census();out['native_submit_claim_exists']=(prep/'native-submit-claim.json').exists();out['observation_finished_unix']=time.time()
print(json.dumps(out))
'''

RESERVE_NEVER_SUBMITTED_CLEANUP_CODE = NATIVE_IDENTITY_CODE + r'''
namespace();assert not (b/'logs/run-manifest.json').exists()
assert not (prep/'native-submit-receipt.json').exists() and not (prep/'native-submit-recovered-receipt.json').exists()
matches,scan=candidates();assert matches==[] and census()==[], 'Actual new namespace/scheduler must be empty before cleanup reservation'
path=prep/'native-submit-disposition.json'
proof=dict(state='never_submitted_cleanup',base=str(b),uid=os.getuid(),source_commit=p['source_commit'],source_sha256=p['source_sha256'],method_sha256=p['method_sha256'],job_name=p['job_name'],argv=argv(),checked_unix=time.time(),scheduler_claim_census=scan,actual_active_namespace_processes=[])
if path.exists():
 value=ordinary(path);assert value['state']=='never_submitted_cleanup' and all(value[key]==proof[key] for key in ('base','uid','source_commit','source_sha256','method_sha256','job_name','argv'))
 proof=value
else:write(path,proof)
# Once reserved, the exclusive sbatch-entered claim cannot be published. This
# is positive non-entry evidence; timeout/scheduler absence never substitutes.
assert ordinary(path)['state']=='never_submitted_cleanup'
print(json.dumps(dict(cleanup_eligible=True,disposition=proof,disposition_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),scope='Reserved never-called sbatch; only this exact source exporter may be cleaned')))
'''

CLEANUP_SOURCE_CODE = r'''
import hashlib,json,os,signal,sys,time
from pathlib import Path
h=json.load(sys.stdin);b=Path(h['export']);source=b/'serve_original_retrieval.py'
assert b.resolve()==b and b.parent==Path('/dataset/shaoshitong') and b.name.startswith('g1-formal-inline-export-')
assert hashlib.sha256(source.read_bytes()).hexdigest()==h['source_sha256']
proc=Path('/proc')/str(h['pid'])
if proc.exists():
 fd=os.pidfd_open(h['pid'])
 try:
  row=(proc/'stat').read_text().rsplit(')',1)[1].split()
  assert (int(row[19]),int(row[3]),proc.stat().st_uid)==(h['start_ticks'],h['sid'],h['uid']) and str(source).encode() in (proc/'cmdline').read_bytes()
  if row[0]!='Z':
   signal.pidfd_send_signal(fd,signal.SIGTERM)
   for _ in range(50):
    if not proc.exists() or (proc/'stat').read_text().rsplit(')',1)[1].split()[0]=='Z':break
    time.sleep(.1)
  assert not proc.exists() or (proc/'stat').read_text().rsplit(')',1)[1].split()[0]=='Z'
 finally:os.close(fd)
proof=dict(cleaned=True,handle=h,checked_unix=time.time(),scope='Only this new source SHA/PIDbirth/SID/UID service; no foreign or old process')
with (b/'cleanup.json').open('x') as f:json.dump(proof,f);f.flush();os.fsync(f.fileno())
print(json.dumps(proof))
'''

SOURCE_HANDLE_CODE = r'''
import hashlib,json,os,sys
from pathlib import Path
p=json.load(sys.stdin);b=Path(p['export_base']);source=b/'serve_original_retrieval.py'
assert b.resolve()==b and b.parent==Path('/dataset/shaoshitong') and b.name.startswith('g1-formal-inline-export-')
assert source.is_file() and not source.is_symlink() and source.stat().st_uid==os.getuid()
assert hashlib.sha256(source.read_bytes()).hexdigest()==p['export_source_sha256']
handle_path=b/'handle.json';launcher_path=b/'launcher-process.json'
assert handle_path.is_file() and not handle_path.is_symlink() and launcher_path.is_file() and not launcher_path.is_symlink()
handle=json.loads(handle_path.read_text());launcher=json.loads(launcher_path.read_text())
assert handle['port']==p['export_port'] and all(handle[key]==launcher[key] for key in ('pid','start_ticks','sid','uid'))
assert launcher['source_sha256']==p['export_source_sha256'] and handle['uid']==os.getuid()
handle.update(export=str(b),source_sha256=p['export_source_sha256']);print(json.dumps(handle))
'''


def submit(args, prep=PREP, repo=REPO):
    need(not (prep / NAMES['intent']).exists(), 'Existing attempt/unknown outcome: observe this handle, never execute again')
    inputs, files, method_bytes = load_inputs(args, repo)
    def rpc(host,code,payload,**kwargs):
        return remote(host,code,payload,producer_sha256=inputs['cpu_producer_sha256'],**kwargs)
    old = rpc('h100-8', OLD_READONLY_CODE, {}, timeout=45)
    token = uuid.uuid4().hex[:8]
    base = '/mnt/data/shaoshitong/g1f-' + token
    export = '/dataset/shaoshitong/g1-formal-inline-export-' + token
    source = exporter_bytes(files, export, inputs['export_port'])
    intent = dict(inputs, base=base, export_base=export,
        job_name='search-g1-formal-inline-' + token, export_source_sha256=sha(source),
        ray_socket_budget=ray_budget(base),
        created_unix=time.time(), previous_actual_readonly=old, external_orchestrator_sha256=sha(Path(__file__).read_bytes()),
        scope='Single fresh official formal job; true1..5 acceptance then same-process6..200; recoverable brakes retain partial writes and obtain fresh rollout; never old training state')
    save(prep / NAMES['intent'], intent)
    payload = dict(intent, files={name: base64.b64encode(value).decode() for name, value in files.items()},
                   method_bytes=base64.b64encode(method_bytes).decode())
    prepared = rpc('h100-8', PREPARE_CODE, payload, timeout=180)
    save(prep / NAMES['package'], prepared)
    handle = rpc('h100-9', START_SOURCE_CODE, dict(intent, source=base64.b64encode(source).decode()))
    save(prep / NAMES['handle'], handle)
    rpc('h100-8', OLD_READONLY_CODE, {}, timeout=45)
    submitted = rpc('h100-8', SUBMIT_CODE, dict(intent, handle=handle), timeout=90)
    save(prep / NAMES['submitted'], submitted)
    return {key: value for key, value in submitted.items() if key not in ('source_sha256', 'argv')}


def actual_launch(intent, observed):
    """Only this actual prepared formal manifest can bind the run identity."""
    need(observed['submission_outcome_known'] is True and observed['source_hashes_match'] is True,
         'Actual source/submission observation missing')
    receipt, run = observed['submission'], observed['run-manifest.json']
    need(submission_authenticated(receipt) and receipt['execution_role'] == ROLE and
         receipt['base'] == observed['base'] == intent['base'] and
         isinstance(receipt['job_id'], str) and re.fullmatch('[1-9][0-9]*', receipt['job_id']) is not None and
         receipt['job_id'] == observed['job_id'] and receipt['job_name'] == intent['job_name'] and
         receipt['method_sha256'] == intent['method_sha256'] and digest(run['run_id'], 32) and
         run.get('policy_calibration_role', 'production') == 'production' and
         run['policy_safety_mode'] == run['budget']['policy_safety_mode'] == SAFETY_MODE and
         type(run['target_updates']) is int and run['target_updates'] == 200 and
         type(run['acceptance_update']) is int and run['acceptance_update'] == 5 and
         run['contract_kind'] == 'fresh200_warmup57' and run['profile'] == 'common8' and
         observed['method_sha256'] == intent['method_sha256'], 'Actual fresh formal inline identity differs')
    need(receipt['source_commit'] == run['source_commit'] == intent['source_commit'] and
         receipt['source_sha256'] == run['source_sha256'] == intent['source_sha256'] and
         all(canonical(run['budget'].get(name)) == canonical(value) for name, value in MATHEMATICS.items()),
         'Actual committed science/source differs')
    need(run['initial_model'] == run['reference_model'] == ASSET_ROOT +
         '/bootstrap-assets/instruct300/assets/qwen2.5-3b-instruct-aa8e725' and
         run['model_revision'] == 'aa8e72537993ba99e69dfaafa59ed015b17504d1' and
         run['input_sha256']['manifest.json'] == '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24' and
         isinstance(run['initial_model_sha256'], dict) and run['initial_model_sha256'] and
         all(digest(value) for value in run['initial_model_sha256'].values()) and
         not any(key in run for key in ('measurement_parent_run_id', 'parent_production_run_id',
             'policy_calibration_discovery_base', 'diagnostic_proof', 'diagnostic_terminal_proof')),
         'Formal run must be fresh official/model/data and cannot inherit failed diagnostic state')
    return dict(receipt, run_id=run['run_id'],
                method_sha256=intent['method_sha256'], source_map_sha256=intent['source_map_sha256'],
                target_updates=200, same_process_after_five=True,
                optimizer_counting=FORMAL_COUNTING,
                negative_tail_semantics='Fixed operational candidate0.01, not inferred healthy/calibrated from old failure',
                source_snapshot_semantics='Bound immutable disk hashes; not proof of loaded modules',
                run_identity_binding='Actual prepared single formal manifest; no future UUID or plan SHA guessed')


def submission_authenticated(receipt):
    if type(receipt.get('returncode')) is int and receipt['returncode'] == 0:
        return receipt.get('submission_authentication', 'actual_sbatch_receipt') == 'actual_sbatch_receipt'
    return (receipt.get('returncode') is None and
            receipt.get('submission_authentication') == 'actual_scheduler_claim_recovery' and
            digest(receipt.get('claim_sha256')) and digest(receipt.get('disposition_sha256')) and
            receipt.get('recovered_native_fields') == job_fields(receipt.get('recovered_native_stdout')) and
            receipt['recovered_native_fields'].get('JobId') == receipt.get('job_id'))


def native_record(observed, launch, state, exit_code):
    need(observed['job_returncode'] == 0 and observed['job_id'] == launch['job_id'] and
         observed['source_hashes_match'] is True, 'Actual raw native record required before purge')
    fields = job_fields(observed['job_stdout'])
    need(fields['JobId'] == launch['job_id'] and fields['JobState'] == state and fields['ExitCode'] == exit_code and
         fields['Restarts'] == '0' and fields['NodeList'] == fields['BatchHost'] == 'h100-8' and
         fields['NumNodes'] == '1' and fields['MinMemoryNode'] == '512G' and
         fields['TresPerNode'] == 'gres:gpu:h100:8' and
         fields['Command'] == launch['base'] + '/repo/reproduction/common8/launch_native.sh' and
         fields['WorkDir'] == launch['base'] + '/repo' and fields['JobName'] == launch['job_name'] and
         fields['JobName'].startswith('search-g1-formal-inline-') and
         all(name in fields for name in BIRTH_FIELDS), 'Wrong native birth/terminal allocation identity')
    need(fields['StdOut'] == fields['StdErr'] == launch['base'] + '/logs/native-formal-inline-' + launch['job_id'] + '.log',
         'Actual native log namespace differs')
    return fields


def terminal_proof(birth, observed, launch):
    born = native_record(birth, launch, 'RUNNING', '0:0')
    final = native_record(observed, launch, 'COMPLETED', '0:0')
    need(all(born[name] == final[name] for name in BIRTH_FIELDS), 'Completed record is not the same captured birth')
    def native_time(value):
        need(re.fullmatch(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d', value) is not None,
             'Actual native timestamps required')
        return datetime.datetime.fromisoformat(value)
    need(native_time(final['SubmitTime']) <= native_time(final['StartTime']) <= native_time(final['EndTime']) and
         birth['checked_unix'] <= birth['observation_finished_unix'] <= observed['checked_unix'] <= observed['observation_finished_unix'],
         'Native captured birth/terminal timeline differs')
    need(observed['pipeline-status.json']['stage'] == 'FINISHED_FULL200_FINAL51713_AND_OWNED_CLEANUP' and
         observed['cleanup-status.json']['cleaned'] is True and observed['cleanup-status.json']['run_id'] == launch['run_id'] and
         observed['actual_active_namespace_processes'] == [] and
         all(row['state'] == 'STOPPED' for row in observed['process-ledger.json']['jobs'].values()) and
         observed['queue_returncode'] == 0 and not observed['queue_stdout'].strip(), 'Actual complete owned cleanup required')
    return dict(observed, execution_role=ROLE, captured_birth=birth,
                captured_birth_sha256=sha(canonical(birth) + b'\n'),
                native_success_scope='Captured actual native COMPLETED0:0/Restart0 and birth before purge; not model/health admission')


def observe(prep=PREP, cleanup_source=False):
    intent = strict_json((prep / NAMES['intent']).read_bytes())
    def rpc(host,code,payload,**kwargs):
        return remote(host,code,payload,producer_sha256=intent['cpu_producer_sha256'],**kwargs)
    observed = rpc('h100-8', OBSERVE_CODE, intent)
    stamp = str(time.time_ns())
    save(prep / ('formal-inline-native-observation-' + stamp + '.json'), observed)
    birth_path = prep / NAMES['birth']
    # Preserve actual native birth even while the official fresh0 manifest is
    # still being prepared. No future run identity is created from this record.
    if observed.get('submission_outcome_known') is True and observed.get('job_returncode') == 0:
        receipt = observed['submission']
        need(submission_authenticated(receipt) and receipt['base'] == intent['base'] and receipt['source_commit'] == intent['source_commit'] and
             receipt['source_sha256'] == intent['source_sha256'] and receipt['execution_role'] == ROLE and
             receipt['job_name'] == intent['job_name'],
             'Actual single saved submission identity differs')
        fields = job_fields(observed['job_stdout'])
        if fields['JobState'] == 'RUNNING' and not birth_path.exists():
            native_record(observed, receipt, 'RUNNING', '0:0')
            save(birth_path, observed)
    launch_path = prep / NAMES['launch']
    if 'run-manifest.json' in observed:
        launch = actual_launch(intent, observed)
        if launch_path.exists():
            need(strict_json(launch_path.read_bytes()) == launch, 'Saved actual launch identity changed')
        else:
            save(launch_path, launch)
        if observed.get('job_returncode') == 0:
            fields = job_fields(observed['job_stdout'])
            if fields['JobState'] == 'COMPLETED':
                need(birth_path.exists(), 'Completion cannot infer uncaptured birth; do not authenticate from purge')
                proof = terminal_proof(strict_json(birth_path.read_bytes()), observed, launch)
                if not (prep / NAMES['terminal']).exists():
                    save(prep / NAMES['terminal'], proof)
    if cleanup_source:
        staging = observed.get('retrieval-staging-status.json', {})
        terminal = observed.get('job_returncode') == 0 and job_fields(observed['job_stdout'])['JobState'] in (
            'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL')
        eligible = ((staging.get('stage') == 'COMPLETE' and staging.get('accepted') is True) or
                    (terminal and observed['actual_active_namespace_processes'] == []))
        if not eligible:
            need(observed.get('submission_state') in (
                'SBATCH_NOT_ENTERED_CLEANUP_RESERVATION_REQUIRED',
                'VERIFIED_NEVER_SUBMITTED_CLEANUP_RESERVED') and
                observed['actual_active_namespace_processes'] == [],
                'New source may still be needed; entered/timeout/purged unknown is not absence proof')
            reservation = rpc('h100-8', RESERVE_NEVER_SUBMITTED_CLEANUP_CODE, intent)
            need(reservation['cleanup_eligible'] is True and reservation['disposition']['state'] == 'never_submitted_cleanup',
                 'Positive exclusive never-submitted reservation required')
            save(prep / ('formal-inline-never-submitted-source-cleanup-' + stamp + '.json'), reservation)
        handle_path = prep / NAMES['handle']
        if handle_path.exists():
            handle = strict_json(handle_path.read_bytes())
        else:
            # Recover the already reserved actual service handle after a lost
            # startup response. Never launch another server or retry sbatch.
            handle = rpc('h100-9', SOURCE_HANDLE_CODE, intent)
            save(handle_path, handle)
        need(handle['export'] == intent['export_base'] and handle['source_sha256'] == intent['export_source_sha256'] and
             handle['port'] == intent['export_port'], 'Only the reserved new source service may be cleaned')
        result = rpc('h100-9', CLEANUP_SOURCE_CODE, handle)
        save(prep / ('formal-inline-retrieval-source-cleanup-' + stamp + '.json'), result)
        return result
    return observed


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    mode = result.add_mutually_exclusive_group(required=True)
    for name in ('check-inputs', 'execute', 'observe', 'cleanup-source'):
        mode.add_argument('--' + name, action='store_true')
    result.add_argument('--source-commit')
    result.add_argument('--source-count', type=int)
    result.add_argument('--cpu-producer-sha256')
    result.add_argument('--export-port', type=int, default=19878)
    result.add_argument('--time-limit', default=NATIVE_WALLTIME)
    for name in ('source-map', 'cpu-modules', 'target-cpu-proof', 'cpu-claim', 'method', 'review-proof', 'previous-launch',
                 'previous-observation', 'previous-source-cleanup'):
        result.add_argument('--' + name)
        result.add_argument('--' + name + '-sha256')
    return result


def main():
    args = parser().parse_args()
    if args.observe or args.cleanup_source:
        need(not any((args.source_commit, args.source_map, args.cpu_modules, args.target_cpu_proof, args.method, args.review_proof,
                      args.source_count, args.cpu_claim, args.cpu_producer_sha256)), 'Observation uses only the saved exact intent')
        result = observe(cleanup_source=args.cleanup_source)
    else:
        required = ('source_commit', 'source_count', 'cpu_producer_sha256', 'cpu_claim', 'cpu_claim_sha256', 'source_map', 'source_map_sha256',
            'cpu_modules', 'cpu_modules_sha256', 'target_cpu_proof', 'target_cpu_proof_sha256', 'method', 'method_sha256', 'review_proof',
            'review_proof_sha256', 'previous_launch', 'previous_launch_sha256', 'previous_observation',
            'previous_observation_sha256', 'previous_source_cleanup', 'previous_source_cleanup_sha256')
        need(all(getattr(args, name) is not None for name in required), 'All explicit source/CPU/review/method/history inputs are required')
        if args.execute:
            result = submit(args)
        else:
            inputs, _, _ = load_inputs(args)
            result = dict(role='local_formal_inline_input_check', source_commit=inputs['source_commit'],
                source_count=inputs['source_count'], source_map_sha256=inputs['source_map_sha256'],
                target_cpu_proof_sha256=inputs['target_cpu_proof_sha256'], actual_target_cpu_tests=inputs['actual_target_cpu_tests'],
                method_sha256=inputs['method_sha256'], structural_verified=True,
                actual_production_updates=0, production_admission=False, healthy_checkpoint_accepted=False,
                physical_execution_authenticated=False, scope='Pure local sealed input/Git blob check; no writes/controller/service/job')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
