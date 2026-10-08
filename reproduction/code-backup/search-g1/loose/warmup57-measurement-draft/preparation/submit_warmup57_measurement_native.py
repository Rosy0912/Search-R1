"""External, explicit fresh200/57 two-outer H100 measurement orchestration.

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
import uuid
from pathlib import Path

ROOT = Path('/Users/shaoshitong/Documents/Search-G1')
REPO = ROOT / 'repo'
PREP = ROOT / 'preparation'
ROLE = 'warmup57_measurement'
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
CPU_MODULES = (
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
    'reproduction.common8.test_generation_behavior', 'reproduction.common8.test_warmup_measurement_plan',
    'reproduction.common8.test_warmup_measurement_evidence', 'reproduction.common8.test_warmup_measurement_runtime',
    'reproduction.common8.test_warmup_measurement_dispatch', 'search_g1.test_measurement_safety')
CPU_SUMMARY_PREFIX = 'FRESH200_CPU_SUMMARY='
NAMES = dict(intent='warmup57-measurement-native-submit-intent.json',
             package='warmup57-measurement-native-package-proof.json',
             handle='warmup57-measurement-retrieval-export-handle.json',
             submitted='warmup57-measurement-native-submit-receipt.json',
             launch='warmup57-measurement-native-launch.json',
             birth='warmup57-measurement-native-birth-observation.json',
             terminal='warmup57-measurement-native-terminal-observation.json')
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
PHASE_BUDGET = dict(outer_count=2, initial_logical_update=0, initial_optimizer_writes=0,
    optimizer_writes_per_rank=40, monitors_per_rank=80, global_monitor_count=80,
    before_evaluation_step=0, after_evaluation_step=2, fixed_validation_count=512,
    fixed_validation_source_counts=dict(nq=256, hotpotqa=256), checkpoint_step=2,
    final_evaluation_in_this_phase=False, updates_counted_as_production=0)
THRESHOLDS = dict(dual_clip=3.0, epsilon=0.2, target_old_policy_kl=0.01,
                  kl_log_bound=30.0, negative_tail_limit=None)
HEALTH = dict(method='paired_fixed512_absolute_allowed_drop',
              allowed_drop=dict(weighted_em=0.0, weighted_f1=0.0, valid_action_rate=0.0))
TAIL_METHOD = 'maximum_of_all_80_accepted_global_monitors'


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
    return json.loads(content, object_pairs_hook=pairs, parse_constant=invalid)


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
    need(isinstance(mapping, dict) and type(count) is int and count > 0 and len(mapping) == count,
         'Explicit complete source map/count required')
    for name, value in mapping.items():
        path = Path(name)
        need(isinstance(name, str) and name and str(path) == name and not path.is_absolute() and
             '..' not in path.parts and not {'.git', '__pycache__', 'evidence'}.intersection(path.parts) and
             (path.suffix in ('.py', '.sh', '.yaml', '.yml', '.json') or name in VERSION_FILES) and
             digest(value), 'Invalid runtime source mapping entry')
    need(set(VERSION_FILES) <= set(mapping), 'Both extensionless runtime version files are required')


def committed_sources(commit, mapping, count, repo=REPO):
    """Dynamic exact Git tree, including version files; no old count pin."""
    need(digest(commit, 40), 'Explicit full immutable source commit required')
    verify_mapping(mapping, count)
    tree = subprocess.check_output(['git', 'ls-tree', '-r', commit], cwd=repo, text=True)
    rows = [row.split('\t', 1) for row in tree.splitlines()]
    selected = {name: metadata for metadata, name in rows if
        (Path(name).suffix in ('.py', '.sh', '.yaml', '.yml', '.json') or name in VERSION_FILES) and
        not {'.git', '__pycache__', 'evidence'}.intersection(Path(name).parts)}
    need(set(selected) == set(mapping), 'Declared map is not the exact complete committed source set')
    files = {}
    for name in sorted(selected):
        need(selected[name].split()[0] in ('100644', '100755') and
             selected[name].split()[1] == 'blob', 'Runtime source must be an ordinary Git blob')
        size = int(subprocess.check_output(['git', 'cat-file', '-s', commit + ':' + name], cwd=repo))
        need(0 <= size <= MAX_FILE_BYTES, 'Only bounded small source files may be relayed')
        content = subprocess.check_output(['git', 'cat-file', 'blob', commit + ':' + name], cwd=repo)
        need(len(content) == size, 'Git source size changed')
        need(sha(content) == mapping[name], 'Committed source differs from actual tested bytes: ' + name)
        files[name] = content
    need(sum(map(len, files.values())) <= MAX_SOURCE_BYTES, 'Source-only package is too large for this launcher')
    need({'reproduction/common8/launch_measurement_native.sh',
          'reproduction/common8/warmup_measurement_plan.py',
          'reproduction/common8/warmup_measurement_runtime.py'} <= set(files), 'New measurement runtime missing')
    budget = strict_json(files['reproduction/common8/budget.json'])
    need(budget.get('contract_kind') == 'fresh200_warmup57' and
         all(canonical(budget.get(name)) == canonical(value) for name, value in MATHEMATICS.items()),
         'Committed scientific fresh200/57 budget differs')
    return files


def verify_cpu(proof, commit, mapping, expected_tests):
    lines = [line[len(CPU_SUMMARY_PREFIX):] for line in proof['stdout'].splitlines()
             if line.startswith(CPU_SUMMARY_PREFIX)]
    need(len(lines) == 1, 'Exactly one actual target CPU summary is required')
    summary = strict_json(lines[0])
    need(type(expected_tests) is int and expected_tests > 0 and proof['source_commit'] == commit and
         type(proof['schema_version']) is int and proof['schema_version'] == 2 and
         proof['role'] == 'fresh200_target_cpu_regression' and
         proof['source_sha256'] == mapping and proof['source_map_sha256'] == sha(canonical(mapping)) and
         type(proof['returncode']) is int and proof['returncode'] == 0 and
         proof['observation_timed_out'] is False and proof['owned_process_group_remaining'] is False and
         summary['successful'] is True and summary['cuda_available'] is False and
         summary.get('cuda_initialized_before') is False and summary.get('cuda_initialized_after') is False and
         summary['skipped'] == [] and type(summary['tests_run']) is int and summary['tests_run'] == expected_tests and
         type(summary['failures']) is int and summary['failures'] == 0 and
         type(summary['errors']) is int and summary['errors'] == 0 and
         type(summary['expected_failures']) is int and summary['expected_failures'] == 0 and
         type(summary['unexpected_successes']) is int and summary['unexpected_successes'] == 0 and
         summary['runner_error'] is None and summary['expected_distribution_versions_matched'] is True and
         summary['cuda_visible_devices'] == '' and summary['system'] == 'Linux',
         'Actual exact-source zero-skip target CPU proof required')
    base = Path(proof['base'])
    need(base.is_absolute() and str(base) == proof['base'] and '..' not in base.parts and
         str(base).startswith(('/dataset/shaoshitong/', '/mnt/data/shaoshitong/')),
         'CPU proof must come from the actual target cluster namespace')
    need(isinstance(proof['modules'], list) and set(proof['modules']) == set(CPU_MODULES) and
         len(set(proof['modules'])) == len(proof['modules']) and
         all(isinstance(module, str) and re.fullmatch(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+', module)
             for module in proof['modules']), 'Explicit actual target CPU suite list required')
    need(summary['modules'] == proof['modules'] and set(summary['module_test_counts']) == set(CPU_MODULES) and
         all(type(value) is int and value > 0 for value in summary['module_test_counts'].values()) and
         sum(summary['module_test_counts'].values()) == expected_tests and
         proof['module_count'] == len(CPU_MODULES), 'All actual old25/new5 CPU suites must have real tests')
    need(all(module.replace('.', '/') + '.py' in mapping for module in CPU_MODULES),
         'Every tested suite must exist in this committed runtime map')
    for key in ('source_initial_sha256', 'source_before_tests_sha256', 'source_after_tests_sha256'):
        need(proof.get(key) == mapping, 'Target source changed before/after CPU tests')
    need(proof['source_unchanged'] is True and proof['source_inventory_error'] is None and
         proof['expected_source_count'] == proof['actual_source_count'] == len(mapping) and
         proof['cpu_runner_unchanged'] is True and digest(proof['cpu_runner_sha256']) and
         proof['cpu_runner_before_sha256'] == proof['cpu_runner_after_sha256'] == proof['cpu_runner_sha256'] and
         proof['runner_read_error'] is None, 'Actual source/runner inventory changed')
    checks = proof.get('shell_syntax_checks')
    need(isinstance(checks, list) and checks and
         len({row['path'] for row in checks}) == len(checks) and
         {'reproduction/common8/train.sh', 'reproduction/common8/launch_calibration_native.sh',
          'reproduction/common8/launch_measurement_native.sh'} <= {row['path'] for row in checks} and
         all(type(row.get('returncode')) is int and row['returncode'] == 0 and
             row['path'] in mapping and row['observation_timed_out'] is False and
             row['owned_process_group_remaining'] is False for row in checks),
         'Actual target shell syntax receipts required')


def verify_method(method):
    need(type(method['schema_version']) is int and method['schema_version'] == 2 and
         method['role'] == 'warmup57_measurement_method' and
         type(method['created_unix']) in (int, float) and math.isfinite(method['created_unix']) and
         0 < method['created_unix'] <= time.time() and method['phase_budget'] == PHASE_BUDGET and
         method['health_rule'] == HEALTH and method['negative_tail_method'] == TAIL_METHOD and
         method['candidate_thresholds'] == THRESHOLDS and
         method['mathematics_sha256'] == sha(canonical(MATHEMATICS)),
         'Wrong prospective warmup57/40-write/max80/zero-drop measurement method')


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


def verify_review(review, commit, mapping, cpu_sha, method_sha, directory):
    need(review['role'] == 'warmup57_measurement_preexecution_review' and review['approved'] is True and
         review['source_commit'] == commit and review['source_sha256'] == mapping and
         review['target_cpu_proof_sha256'] == cpu_sha and review['method_sha256'] == method_sha and
         review['blocking_findings'] == [] and review['scope'] == 'staticonly' and
         review['physical_execution_authenticated'] is False and review['production_admission'] is False and
         review['healthy_checkpoint_accepted'] is False and digest(review['actualreview_report_sha256']),
         'Independent exact-source preexecution review required')
    report = Path(review['actualreview_report_path'])
    need(report.is_absolute() and report.resolve() == report and report.is_file() and not report.is_symlink() and
         sha(report.read_bytes()) == review['actualreview_report_sha256'], 'Actual independent review report bytes changed')
    expected = {'submit_warmup57_measurement_native.py', 'test_submit_warmup57_measurement_native.py'}
    need(set(review['external_file_sha256']) == expected, 'Both reviewed external files must be bound')
    for name, value in review['external_file_sha256'].items():
        path = directory / name
        need(path.is_file() and not path.is_symlink() and digest(value) and sha(path.read_bytes()) == value,
             'Reviewed external submitter/test bytes changed')


def load_inputs(args, repo=REPO):
    """Pure local validation; no controller import, writes or remote calls."""
    mapping_bytes, mapping = sealed(args.source_map, args.source_map_sha256)
    files = committed_sources(args.source_commit, mapping, args.source_count, repo)
    cpu_bytes, cpu = sealed(args.target_cpu_proof, args.target_cpu_proof_sha256)
    verify_cpu(cpu, args.source_commit, mapping, args.expected_cpu_tests)
    method_bytes, method = sealed(args.method, args.method_sha256)
    verify_method(method)
    review_bytes, review = sealed(args.review_proof, args.review_proof_sha256)
    verify_review(review, args.source_commit, mapping, sha(cpu_bytes), sha(method_bytes), Path(__file__).parent)
    _, old_launch = sealed(args.previous_launch, args.previous_launch_sha256)
    _, old_observed = sealed(args.previous_observation, args.previous_observation_sha256)
    _, cleanup = sealed(args.previous_source_cleanup, args.previous_source_cleanup_sha256)
    verify_old_failure(old_launch, old_observed, cleanup)
    need(type(args.export_port) is int and 1024 <= args.export_port <= 65535 and args.export_port != 19874,
         'Explicit new independent retrieval-export port required')
    return dict(source_commit=args.source_commit, source_sha256=mapping, source_count=len(mapping),
        source_map_file_sha256=sha(mapping_bytes), source_map_sha256=sha(canonical(mapping)),
        target_cpu_proof_sha256=sha(cpu_bytes), target_cpu_proof_path=str(Path(args.target_cpu_proof)),
        actual_target_cpu_tests=args.expected_cpu_tests, actual_cpu_modules=cpu['modules'],
        method_sha256=sha(method_bytes), review_proof_sha256=sha(review_bytes),
        previous_launch_sha256=args.previous_launch_sha256,
        previous_observation_sha256=args.previous_observation_sha256,
        previous_source_cleanup_sha256=args.previous_source_cleanup_sha256,
        old_source_handle=cleanup['handle'], export_port=args.export_port,
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


def remote(host, code, payload, timeout=90):
    """Existing h100→mgmt/SOCKS route only; never config/VPN mutation."""
    need(host in ('h100-8', 'h100-9'), 'Only the declared target/source H100 hosts are allowed')
    sys.path.insert(0, '/Users/shaoshitong/Documents/remote_codex_control')
    import control
    command = 'python3 -c ' + shlex.quote(code)
    result = subprocess.run(control.route(host, command), input=json.dumps(payload, allow_nan=False),
                            capture_output=True, text=True, timeout=timeout)
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
assert set(p['files'])==set(p['source_sha256']) and len(p['files'])==p['source_count']
for key in ('base','parent_base'):
 b=Path(p[key]);assert b.resolve()==b and b.parent==Path('/mnt/data/shaoshitong') and not b.exists() and not b.is_symlink()
for key in ('base','parent_base'):
 b=Path(p[key]);b.mkdir(exist_ok=False);sync(b.parent)
 for name in ('repo','data','preparation','logs','tmp','ray','artifacts','cache/triton','cache/inductor','cache/hf'):(b/name).mkdir(parents=True,exist_ok=True)
 write(b/'preparation/namespace-owner.json',json.dumps(dict(base=str(b),measurement_base=p['base'],execution_role=p['execution_role'],created_unix=p['created_unix'],uid=os.getuid(),source_commit=p['source_commit'])).encode())
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
  def digest(path):
   h=hashlib.sha256()
   with path.open('rb') as f:
    for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
   return h.hexdigest()
  assert digest(source)==digest(target);data_hashes[name]=digest(target)
 assert data_hashes['manifest.json']=='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
 original=anchor/'preparation/official-model-manifest-original.json';assert original.is_file() and not original.is_symlink()
 write(b/'preparation/official-model-manifest-original.json',original.read_bytes())
 content=base64.b64decode(p['method_bytes'],validate=True);assert hashlib.sha256(content).hexdigest()==p['method_sha256']
 write(b/'preparation/warmup57-measurement-method.json',content)
 write(b/'preparation/source-package-intent.json',json.dumps({k:v for k,v in p.items() if k not in ('files','method_bytes')},allow_nan=False).encode())
 sync(b/'data');sync(b/'repo');sync(b)
print(json.dumps(dict(base=p['base'],parent_base=p['parent_base'],source_commit=p['source_commit'],source_sha256=p['source_sha256'],source_count=p['source_count'],data_sha256=data_hashes,method_sha256=p['method_sha256'],checked_unix=time.time(),scope='Only new source namespaces and original six canonical byte copies; native preparation validates science/model/data; no old state copied')))
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

SUBMIT_CODE = r'''
import hashlib,json,os,subprocess,sys,time,urllib.request
from pathlib import Path
p=json.load(sys.stdin);b=Path(p['base']);parent=Path(p['parent_base'])
assert not (b/'logs/run-manifest.json').exists() and not (parent/'logs/run-manifest.json').exists()
for namespace in (b,parent):
 assert hashlib.sha256((namespace/'preparation/warmup57-measurement-method.json').read_bytes()).hexdigest()==p['method_sha256']
 for name,digest in p['source_sha256'].items():assert hashlib.sha256((namespace/'repo'/name).read_bytes()).hexdigest()==digest
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
for name,size in p['handle']['files'].items():
 with opener.open(urllib.request.Request('http://10.33.0.13:'+str(p['export_port'])+'/'+name,method='HEAD'),timeout=10) as response:assert response.status==200 and int(response.headers['Content-Length'])==size
launcher=b/'repo/reproduction/common8/launch_measurement_native.sh'
args=['sbatch','--parsable','--partition=gpu','--nodelist=h100-8','--nodes=1','--ntasks=1','--cpus-per-task=64','--gres=gpu:h100:8','--mem=512G','--exclusive','--time=08:00:00','--job-name='+p['job_name'],'--chdir='+str(b/'repo'),'--output='+str(b/'logs/native-warmup57-%j.log'),'--export=ALL,SEARCH_G1_BASE='+str(b)+',SEARCH_G1_PRODUCTION_BASE='+str(parent)+',SEARCH_G1_MEASUREMENT_METHOD='+str(b/'preparation/warmup57-measurement-method.json')+',SEARCH_G1_SOURCE_COMMIT='+p['source_commit']+',SEARCH_G1_EXPORT_PORT='+str(p['export_port']),str(launcher)]
def write(path,value):
 with path.open('x') as f:json.dump(value,f,allow_nan=False);f.flush();os.fsync(f.fileno())
 fd=os.open(str(path.parent),os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
write(b/'preparation/native-submit-claim.json',dict(argv=args,created_unix=time.time(),base=str(b),execution_role='warmup57_measurement',method_sha256=p['method_sha256']))
write(b/'preparation/retrieval-source-handle.json',p['handle'])
submitted=time.time();result=subprocess.run(args,capture_output=True,text=True,timeout=60)
proof=dict(base=str(b),parent_base=str(parent),execution_role='warmup57_measurement',source_commit=p['source_commit'],source_sha256=p['source_sha256'],source_count=p['source_count'],method_sha256=p['method_sha256'],target_cpu_proof_sha256=p['target_cpu_proof_sha256'],argv=args,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,submitted_unix=submitted,observed_unix=time.time(),actual_production_updates=0,scope='Native submission only; actual run/parent UUIDs are not guessed')
if result.returncode==0:
 proof['job_id']=result.stdout.strip().split(';')[0];assert proof['job_id'].isdigit()
write(b/'preparation/native-submit-receipt.json',proof)
assert result.returncode==0,result.stderr
print(json.dumps(proof))
'''

OBSERVE_CODE = r'''
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
p=json.load(sys.stdin);b=Path(p['base']);parent=Path(p['parent_base']);out=dict(base=str(b),parent_base=str(parent),checked_unix=time.time(),execution_role='warmup57_measurement')
receipt=b/'preparation/native-submit-receipt.json';out['submission_outcome_known']=receipt.exists()
job=None
if receipt.exists():
 out['submission']=json.loads(receipt.read_text());job=out['submission'].get('job_id')
if job:
 out['job_id']=job;r=subprocess.run(['scontrol','show','job','-o',job],capture_output=True,text=True,timeout=15)
 out.update(job_stdout=r.stdout,job_returncode=r.returncode,job_stderr=r.stderr)
 r=subprocess.run(['squeue','-h','-j',job,'-o','%i|%T|%j'],capture_output=True,text=True,timeout=15);out.update(queue_returncode=r.returncode,queue_stdout=r.stdout,queue_stderr=r.stderr)
out['source_hashes_match']=all(hashlib.sha256((namespace/'repo'/name).read_bytes()).hexdigest()==digest for namespace in (b,parent) for name,digest in p['source_sha256'].items())
for name in ('run-manifest.json','pipeline-status.json','cleanup-status.json','process-ledger.json','retrieval-staging-status.json','warmup57-measurement-plan.json','warmup57-measurement-segment.json'):
 path=b/'logs'/name
 if path.exists():out[name]=json.loads(path.read_text());out[name+'-sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
path=parent/'logs/run-manifest.json'
if path.exists():out['parent-run-manifest.json']=json.loads(path.read_text());out['parent_manifest_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
active=[];marker=('SEARCH_G1_BASE='+str(b)).encode()
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  if proc.stat().st_uid!=os.getuid():continue
  row=(proc/'stat').read_text().rsplit(')',1)[1].split()
  if row[0]=='Z':continue
  if marker in (proc/'environ').read_bytes().split(b'\0') or str(b).encode() in (proc/'cmdline').read_bytes():active.append(dict(pid=int(proc.name),start_ticks=int(row[19]),sid=int(row[3]),uid=proc.stat().st_uid))
 except (FileNotFoundError,ProcessLookupError):pass
out['actual_active_namespace_processes']=active;out['native_submit_claim_exists']=(b/'preparation/native-submit-claim.json').exists();out['observation_finished_unix']=time.time()
print(json.dumps(out))
'''

CLEANUP_SOURCE_CODE = r'''
import hashlib,json,os,signal,sys,time
from pathlib import Path
h=json.load(sys.stdin);b=Path(h['export']);source=b/'serve_original_retrieval.py'
assert b.resolve()==b and b.parent==Path('/dataset/shaoshitong') and b.name.startswith('g1-warmup57-export-')
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


def submit(args, prep=PREP, repo=REPO):
    need(not (prep / NAMES['intent']).exists(), 'Existing attempt/unknown outcome: observe this handle, never execute again')
    inputs, files, method_bytes = load_inputs(args, repo)
    old = remote('h100-8', OLD_READONLY_CODE, {}, timeout=45)
    token = uuid.uuid4().hex[:8]
    base, parent = '/mnt/data/shaoshitong/g1m-' + token, '/mnt/data/shaoshitong/g1q-' + token
    export = '/dataset/shaoshitong/g1-warmup57-export-' + token
    source = exporter_bytes(files, export, inputs['export_port'])
    intent = dict(inputs, base=base, parent_base=parent, export_base=export,
        job_name='search-g1-warmup57-' + token, export_source_sha256=sha(source),
        ray_socket_budgets={key: ray_budget(value) for key, value in (('child', base), ('parent', parent))},
        created_unix=time.time(), previous_actual_readonly=old, external_orchestrator_sha256=sha(Path(__file__).read_bytes()),
        scope='Fresh official two-outer warmup measurement, separate real fresh0 parent; never old training state')
    save(prep / NAMES['intent'], intent)
    payload = dict(intent, files={name: base64.b64encode(value).decode() for name, value in files.items()},
                   method_bytes=base64.b64encode(method_bytes).decode())
    prepared = remote('h100-8', PREPARE_CODE, payload, timeout=180)
    save(prep / NAMES['package'], prepared)
    handle = remote('h100-9', START_SOURCE_CODE, dict(intent, source=base64.b64encode(source).decode()))
    save(prep / NAMES['handle'], handle)
    remote('h100-8', OLD_READONLY_CODE, {}, timeout=45)
    submitted = remote('h100-8', SUBMIT_CODE, dict(intent, handle=handle), timeout=90)
    save(prep / NAMES['submitted'], submitted)
    return {key: value for key, value in submitted.items() if key not in ('source_sha256', 'argv')}


def actual_launch(intent, observed):
    """Only actual manifests can bind parent/run identities after preparation."""
    need(observed['submission_outcome_known'] is True and observed['source_hashes_match'] is True,
         'Actual source/submission observation missing')
    receipt, run, parent = observed['submission'], observed['run-manifest.json'], observed['parent-run-manifest.json']
    need(receipt['returncode'] == 0 and receipt['execution_role'] == ROLE and
         receipt['base'] == observed['base'] == intent['base'] and
         receipt['parent_base'] == observed['parent_base'] == intent['parent_base'] and
         receipt['method_sha256'] == intent['method_sha256'] and digest(run['run_id'], 32) and
         digest(parent['run_id'], 32) and run['run_id'] != parent['run_id'] and
         run['policy_calibration_role'] == ROLE and run['measurement_parent_run_id'] == parent['run_id'] and
         run['measurement_method_sha256'] == intent['method_sha256'] and
         parent.get('policy_calibration_role', 'production') == 'production' and
         parent['completed_updates'] == 0 and run['target_updates'] == parent['target_updates'] == 200 and
         run['contract_kind'] == parent['contract_kind'] == 'fresh200_warmup57', 'Actual child/fresh0 parent identity differs')
    for document in (run, parent):
        need(document['source_commit'] == intent['source_commit'] and document['source_sha256'] == intent['source_sha256'] and
             all(canonical(document['budget'].get(name)) == canonical(value) for name, value in MATHEMATICS.items()),
             'Actual committed science/source differs')
    for key in ('initial_model_sha256', 'input_sha256', 'calibration_question_ids', 'model_revision'):
        need(run[key] == parent[key], 'Actual measurement and parent scientific graph differs: ' + key)
    return dict(receipt, run_id=run['run_id'], parent_run_id=parent['run_id'],
                parent_manifest_sha256=observed['parent_manifest_sha256'],
                method_sha256=intent['method_sha256'], source_map_sha256=intent['source_map_sha256'],
                parent_identity_binding='Actual prepared manifests; no future UUID or plan SHA guessed')


def native_record(observed, launch, state, exit_code):
    need(observed['job_returncode'] == 0 and observed['job_id'] == launch['job_id'] and
         observed['source_hashes_match'] is True, 'Actual raw native record required before purge')
    fields = job_fields(observed['job_stdout'])
    need(fields['JobId'] == launch['job_id'] and fields['JobState'] == state and fields['ExitCode'] == exit_code and
         fields['Restarts'] == '0' and fields['NodeList'] == fields['BatchHost'] == 'h100-8' and
         fields['NumNodes'] == '1' and fields['MinMemoryNode'] == '512G' and
         fields['TresPerNode'] == 'gres/gpu:h100:8' and
         fields['Command'] == launch['base'] + '/repo/reproduction/common8/launch_measurement_native.sh' and
         fields['WorkDir'] == launch['base'] + '/repo' and fields['JobName'].startswith('search-g1-warmup57-') and
         all(name in fields for name in BIRTH_FIELDS), 'Wrong native birth/terminal allocation identity')
    need(fields['StdOut'] == fields['StdErr'] == launch['base'] + '/logs/native-warmup57-' + launch['job_id'] + '.log',
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
    need(observed['pipeline-status.json']['stage'] == 'FINISHED_WARMUP57_MEASUREMENTS_AND_OWNED_CLEANUP' and
         observed['cleanup-status.json']['cleaned'] is True and observed['cleanup-status.json']['run_id'] == launch['run_id'] and
         observed['actual_active_namespace_processes'] == [] and
         all(row['state'] == 'STOPPED' for row in observed['process-ledger.json']['jobs'].values()) and
         observed['queue_returncode'] == 0 and not observed['queue_stdout'].strip(), 'Actual complete owned cleanup required')
    return dict(observed, execution_role=ROLE, captured_birth=birth,
                captured_birth_sha256=sha(canonical(birth) + b'\n'),
                native_success_scope='Captured actual native COMPLETED0:0/Restart0 and birth before purge; not model/health admission')


def observe(prep=PREP, cleanup_source=False):
    intent = strict_json((prep / NAMES['intent']).read_bytes())
    observed = remote('h100-8', OBSERVE_CODE, intent)
    stamp = str(time.time_ns())
    save(prep / ('warmup57-measurement-native-observation-' + stamp + '.json'), observed)
    launch_path = prep / NAMES['launch']
    if 'run-manifest.json' in observed and 'parent-run-manifest.json' in observed:
        launch = actual_launch(intent, observed)
        if launch_path.exists():
            need(strict_json(launch_path.read_bytes()) == launch, 'Saved actual launch identity changed')
        else:
            save(launch_path, launch)
        if observed.get('job_returncode') == 0:
            fields = job_fields(observed['job_stdout'])
            birth_path = prep / NAMES['birth']
            if fields['JobState'] == 'RUNNING' and not birth_path.exists():
                native_record(observed, launch, 'RUNNING', '0:0')
                save(birth_path, observed)
            if fields['JobState'] == 'COMPLETED':
                need(birth_path.exists(), 'Completion cannot infer uncaptured birth; do not authenticate from purge')
                proof = terminal_proof(strict_json(birth_path.read_bytes()), observed, launch)
                if not (prep / NAMES['terminal']).exists():
                    save(prep / NAMES['terminal'], proof)
    if cleanup_source:
        staging = observed.get('retrieval-staging-status.json', {})
        terminal = observed.get('job_returncode') == 0 and job_fields(observed['job_stdout'])['JobState'] in (
            'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL')
        need((staging.get('stage') == 'COMPLETE' and staging.get('accepted') is True) or
             (terminal and observed['actual_active_namespace_processes'] == []), 'New source may still be needed')
        handle = strict_json((prep / NAMES['handle']).read_bytes())
        need(handle['export'] == intent['export_base'] and handle['source_sha256'] == intent['export_source_sha256'] and
             handle['port'] == intent['export_port'], 'Only the reserved new source service may be cleaned')
        result = remote('h100-9', CLEANUP_SOURCE_CODE, handle)
        save(prep / ('warmup57-measurement-retrieval-source-cleanup-' + stamp + '.json'), result)
        return result
    return observed


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    mode = result.add_mutually_exclusive_group(required=True)
    for name in ('check-inputs', 'execute', 'observe', 'cleanup-source'):
        mode.add_argument('--' + name, action='store_true')
    result.add_argument('--source-commit')
    result.add_argument('--source-count', type=int)
    result.add_argument('--expected-cpu-tests', type=int)
    result.add_argument('--export-port', type=int, default=19876)
    for name in ('source-map', 'target-cpu-proof', 'method', 'review-proof', 'previous-launch',
                 'previous-observation', 'previous-source-cleanup'):
        result.add_argument('--' + name)
        result.add_argument('--' + name + '-sha256')
    return result


def main():
    args = parser().parse_args()
    if args.observe or args.cleanup_source:
        need(not any((args.source_commit, args.source_map, args.target_cpu_proof, args.method, args.review_proof,
                      args.source_count, args.expected_cpu_tests)), 'Observation uses only the saved exact intent')
        result = observe(cleanup_source=args.cleanup_source)
    else:
        required = ('source_commit', 'source_count', 'expected_cpu_tests', 'source_map', 'source_map_sha256',
            'target_cpu_proof', 'target_cpu_proof_sha256', 'method', 'method_sha256', 'review_proof',
            'review_proof_sha256', 'previous_launch', 'previous_launch_sha256', 'previous_observation',
            'previous_observation_sha256', 'previous_source_cleanup', 'previous_source_cleanup_sha256')
        need(all(getattr(args, name) is not None for name in required), 'All explicit source/CPU/review/method/history inputs are required')
        if args.execute:
            result = submit(args)
        else:
            inputs, _, _ = load_inputs(args)
            result = dict(role='local_warmup57_measurement_input_check', source_commit=inputs['source_commit'],
                source_count=inputs['source_count'], source_map_sha256=inputs['source_map_sha256'],
                target_cpu_proof_sha256=inputs['target_cpu_proof_sha256'], actual_target_cpu_tests=inputs['actual_target_cpu_tests'],
                method_sha256=inputs['method_sha256'], structural_verified=True,
                actual_production_updates=0, production_admission=False, healthy_checkpoint_accepted=False,
                physical_execution_authenticated=False, scope='Pure local sealed input/Git blob check; no writes/controller/service/job')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
