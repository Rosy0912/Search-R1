"""Draft independent official0 discovery after the recorded 3294 failure.

External orchestration only; no execution is authorized by this draft. Root
must provide the actual new full commit, original target CPU proof SHA/count,
and independently sealed old Hydra/source-cleanup observations. Neither new
runtime identity nor a future proof/count/snapshot path is guessed. Historical
scheduler terminal bytes are never inferred from a
RUNNING snapshot, workflow status, process exit, or a purged controller entry.
"""
import argparse
import base64
import datetime
import hashlib
import inspect
import json
import math
import os
import re
import stat
import subprocess
import sys
import time
import uuid
from pathlib import Path

PREP = Path('/Users/shaoshitong/Documents/Search-G1/preparation')
REPO = PREP.parent / 'repo'
sys.path.insert(0, str(PREP))
import submit_policy_confirmation_native as common

NAMES = dict(intent='policy-discovery-v6-native-submit-intent.json',
             package='policy-discovery-v6-native-package-proof.json',
             handle='policy-discovery-v6-retrieval-export-handle.json',
             launch='policy-discovery-v6-native-launch.json')
OLD_COMMIT = 'e670dfacab4ff0146f7ea0ca60e3f49675a4d2f6'
OLD_RUN = 'ee0d52bd4edb465993214a2a48f7521a'
OLD_PARENT_RUN = '00f18587fb4444faa62210eb72c84e98'
OLD_BASE = '/mnt/data/shaoshitong/g1d-a6933c85'
OLD_PARENT_BASE = '/mnt/data/shaoshitong/g1p-a6933c85'
OLD_JOB = '3294'
OLD_LAUNCH_NAME = 'policy-discovery-v5-native-launch.json'
OLD_LAUNCH_SHA256 = '1d50b96ed51f916e21891c1db88af03cdc3dea035bdec8471aa66959877c041f'
TERMINAL_NAME = 'policy-discovery-v5-observation-1791256551399966.json'
TERMINAL_SHA256 = '681b38622bb4c32284dbd203fbede46f61efa46db6bde04de450797d64b2df70'
BIRTH_NAME = 'policy-discovery-v5-observation-1791254632444188.json'
BIRTH_SHA256 = 'c57f8e39d081ff114d9ce9c2c3e8f532c0cb15ef54caf2039b44d65dbf0f4889'
CPU_OLD_NAME = 'behavior-consistency-target-cpu-v4-result.json'
CPU_MODULES = ('reproduction.common8.test_policy_calibration_plan',
 'reproduction.common8.test_freeze_calibration_plan',
 'reproduction.common8.test_safety_calibration_evidence',
 'reproduction.common8.test_calibration_evaluation', 'reproduction.common8.test_calibration_runtime',
 'reproduction.common8.test_prepare_calibration_run', 'reproduction.common8.test_adopt_initial_probes',
 'reproduction.common8.test_production_safety_admission', 'reproduction.common8.test_pipeline_calibration',
 'search_g1.test_policy_safety', 'reproduction.common8.test_temporary_cache',
 'reproduction.common8.test_sampling_capture', 'reproduction.common8.test_behavior_trace',
 'reproduction.common8.test_behavior_probability', 'reproduction.common8.test_behavior_runtime',
 'reproduction.common8.test_generation_behavior')
INVALID_JOB = 'slurm_load_jobs error: Invalid job id specified'
BIRTH_FIELDS = ('JobId', 'SubmitTime', 'StartTime', 'NodeList', 'BatchHost',
                'Command', 'WorkDir', 'StdOut', 'StdErr', 'Restarts',
                'UserId', 'NumNodes', 'MinMemoryNode', 'TresPerNode')
SOURCE_HANDLE_NAME = 'policy-discovery-v5-retrieval-export-handle.json'
SOURCE_HANDLE_SHA256 = '50608a82969fba4f935920f8c361f70b86fd4cdd0d61ab9bd7ceb45c00d8b1a3'


def _digest(value, length=64):
    return isinstance(value, str) and len(value) == length and re.fullmatch('[0-9a-f]+', value) is not None


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def source_set_schema(document):
    """Only the exact historical three snapshots; no directory-wide exclusion."""
    common.require(set(document) == {'checked_unix', 'namespaces'} and _number(document['checked_unix']) and
                   document['checked_unix'] > 0 and isinstance(document['namespaces'], list) and
                   len(document['namespaces']) == 2, 'Invalid original scoped source-set observation')
    for index, namespace in enumerate(document['namespaces']):
        expected_base = (OLD_BASE, OLD_PARENT_BASE)[index]
        count = (3, 0)[index]
        common.require(set(namespace) == {'base','expected_count','actual_count','extra','missing','changed'} and
                       namespace['base'] == expected_base and type(namespace['expected_count']) is int and
                       namespace['expected_count'] == 353 and type(namespace['actual_count']) is int and
                       namespace['actual_count'] == 353 + count and namespace['missing'] == namespace['changed'] == [] and
                       isinstance(namespace['extra'], list) and len(namespace['extra']) == count,
                       'Original declared353 source set/missing/changed/namespaces differs')
        paths = [row['path'] for row in namespace['extra']]
        if count:
            common.require(len(set(paths)) == 3 and paths == sorted(paths) and
                all(isinstance(name, str) and re.fullmatch(
                    r'outputs/\d{4}-\d\d-\d\d/\d\d-\d\d-\d\d/\.hydra/(config|hydra|overrides)\.yaml', name)
                    for name in paths) and len({str(Path(name).parent) for name in paths}) == 1 and
                {Path(name).name for name in paths} == {'config.yaml','hydra.yaml','overrides.yaml'},
                'Need the exact independently sealed three same-run Hydra snapshot paths')
        for row in namespace['extra']:
            common.require(set(row) == {'path','bytes','sha256','mtime','symlink'} and
                           type(row['bytes']) is int and 0 < row['bytes'] <= 131072 and _digest(row['sha256']) and
                           _number(row['mtime']) and row['mtime'] <= document['checked_unix'] and row['symlink'] is False,
                           'Original snapshot byte/hash/type metadata differs')
    return document


def read_source_set_observation(raw_bytes, expected_sha256):
    common.require(isinstance(raw_bytes, bytes) and _digest(expected_sha256) and
                   common.sha(raw_bytes) == expected_sha256,
                   'Explicit root-sealed original source-set observation SHA is required')
    return source_set_schema(common.strict_json(raw_bytes))


def _verify_original_mapping(namespace, mapping):
    """Authenticate every declared353 source file before any import/exception."""
    repo = namespace / 'repo'
    common.require(namespace.resolve() == namespace and len(mapping) == 353 and
                   {'verl/version/version', 'verl/single_controller/version/version'} <= set(mapping),
                   'Original complete353 namespace/source map differs')
    for name, digest in mapping.items():
        path = repo / name
        before = path.lstat()
        _verify_metadata_file(path, dict(size_bytes=before.st_size, mtime_ns=before.st_mtime_ns,
            mode=oct(stat.S_IMODE(before.st_mode)), sha256=digest))


def _verify_new_mapping(namespace, mapping):
    """Authenticate this explicit complete354 new source before importing it."""
    common.require(namespace.resolve() == namespace and len(mapping) == 354 and
                   {'verl/version/version', 'verl/single_controller/version/version',
                    'reproduction/common8/sampling_capture.py'} <= set(mapping),
                   'New complete354 namespace/source map differs')
    for name, digest in mapping.items():
        path = namespace / 'repo' / name
        before = path.lstat()
        _verify_metadata_file(path, dict(size_bytes=before.st_size, mtime_ns=before.st_mtime_ns,
            mode=oct(stat.S_IMODE(before.st_mode)), sha256=digest))


def _verify_original_source_set(namespace, mapping, captured):
    """Check exact extras and actual ordinary bytes; never alter old snapshots."""
    repo = namespace / 'repo'
    common.require(captured['base'] == str(namespace), 'Source-set observation namespace differs')
    names = {str(path.relative_to(repo)) for path in repo.rglob('*') if path.is_file() and
        (path.suffix in ('.py', '.sh', '.yaml', '.yml', '.json') or
         str(path.relative_to(repo)) in ('verl/version/version', 'verl/single_controller/version/version')) and
        not {'.git', '__pycache__', 'evidence'}.intersection(path.relative_to(repo).parts)}
    extras = {row['path'] for row in captured['extra']}
    common.require(not extras.intersection(mapping) and names == set(mapping) | extras and
                   len(mapping) == captured['expected_count'] == 353 and len(names) == captured['actual_count'],
                   'Source set differs outside the exact historical Hydra byte snapshots')
    for row in captured['extra']:
        path = repo / row['path']
        before = path.lstat()
        common.require(path.resolve() == path and not path.is_symlink() and stat.S_ISREG(before.st_mode) and
                       before.st_uid == os.getuid() and before.st_size == row['bytes'] and before.st_mtime == row['mtime'],
                       'Original generated snapshot size/type/mtime changed')
        descriptor = os.open(str(path), os.O_RDONLY | os.O_NOFOLLOW)
        try:
            opened = os.fstat(descriptor)
            common.require((opened.st_dev,opened.st_ino,opened.st_size,opened.st_mtime_ns) ==
                           (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns) and stat.S_ISREG(opened.st_mode),
                           'Generated snapshot was replaced during open')
            with os.fdopen(descriptor, 'rb', closefd=False) as stream: content = stream.read()
            after = os.fstat(descriptor)
        finally: os.close(descriptor)
        final = path.lstat()
        common.require(len(content) == row['bytes'] and common.sha(content) == row['sha256'] and
                       (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns) ==
                       (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns) and
                       (final.st_dev,final.st_ino,final.st_size,final.st_mtime_ns,final.st_ctime_ns) ==
                       (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns),
                       'Original generated snapshot bytes/SHA/identity changed')


def _verify_metadata_file(path, expected):
    """Stream only an already sealed ordinary file; never load its raw journal."""
    path = Path(path); before = path.lstat()
    common.require(path.resolve() == path and stat.S_ISREG(before.st_mode) and
        before.st_uid == os.getuid() and type(expected['size_bytes']) is int and expected['size_bytes'] >= 0 and
        before.st_size == expected['size_bytes'] and before.st_mtime_ns == expected['mtime_ns'] and
        oct(stat.S_IMODE(before.st_mode)) == expected['mode'] and _digest(expected['sha256']),
        'Original diagnostic file type/owner/size/mode/time changed')
    descriptor = os.open(str(path), os.O_RDONLY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(descriptor)
        common.require((opened.st_dev,opened.st_ino,opened.st_size,opened.st_mtime_ns,opened.st_ctime_ns) ==
                       (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns),
                       'Original diagnostic file changed during open')
        digest = hashlib.sha256()
        with os.fdopen(descriptor, 'rb', closefd=False) as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''): digest.update(block)
        after = os.fstat(descriptor)
    finally: os.close(descriptor)
    final = path.lstat()
    common.require(digest.hexdigest() == expected['sha256'] and
        (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns) ==
        (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns) ==
        (final.st_dev,final.st_ino,final.st_size,final.st_mtime_ns,final.st_ctime_ns),
        'Original diagnostic file bytes/SHA changed')


def _job_fields(raw):
    """Strict one-line scoped scontrol fields; duplicate keys are not evidence."""
    common.require(isinstance(raw, str) and len(raw.strip().splitlines()) == 1,
                   'Missing unique actual scheduler record')
    fields = {}
    for token in raw.split():
        common.require('=' in token, 'Invalid scoped scheduler field')
        key, value = token.split('=', 1)
        common.require(key and key not in fields, 'Duplicate scheduler field')
        fields[key] = value
    return fields


def _scheduler_time(raw):
    # Slurm's recorded wall-clock timezone is not invented here. Job birth
    # timestamps must match the independently captured initial record exactly.
    common.require(isinstance(raw, str) and re.fullmatch(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d', raw),
                   'Missing actual scheduler timestamp')
    return datetime.datetime.fromisoformat(raw)


def _snapshot_binding(observed, launch, run_id, parent_run_id):
    common.require(observed['base'] == launch['base'] and observed['parent_base'] == launch['parent_base'] and
                   observed['job_id'] == launch['job_id'] and observed['source_hashes_match'] is True and
                   observed['run-manifest.json']['run_id'] == run_id and
                   observed['run-manifest.json']['parent_production_run_id'] == parent_run_id and
                   observed['run-manifest.json']['source_commit'] == launch['source_commit'] and
                   observed['run-manifest.json']['policy_calibration_role'] == 'policy_safety_calibration_discovery',
                   'Captured run/source/native namespace differs from launch')
    common.require(_number(observed['checked_unix']) and _number(observed['observation_finished_unix']) and
                   launch['submitted_unix'] <= observed['checked_unix'] <= observed['observation_finished_unix'],
                   'Invalid real capture/launch timeline')


def read_terminal_observation(raw_bytes, *, expected_sha256, launch_bytes, expected_launch_sha256,
                              expected_run_id, expected_parent_run_id, expected_state, expected_exit_code,
                              birth_observation_bytes, expected_birth_sha256):
    """Read original actual Slurm terminal bytes; no live/GPU/health authentication.

    Reusable by confirmation's small and native readers with an explicit
    COMPLETED/0:0 expectation. FAILED/1:0 is exclusively previous-failure scope.
    Caller-supplied hashes must already be sealed in its original contract.
    """
    common.require((expected_state, expected_exit_code) in (('FAILED', '1:0'), ('COMPLETED', '0:0')),
                   'Only an explicit actual failure or successful completion is supported')
    for content, digest in ((raw_bytes, expected_sha256), (launch_bytes, expected_launch_sha256),
                            (birth_observation_bytes, expected_birth_sha256)):
        common.require(isinstance(content, bytes) and _digest(digest) and common.sha(content) == digest,
                       'Original scheduler/launch/birth bytes SHA changed')
    common.require(_digest(expected_run_id, 32) and _digest(expected_parent_run_id, 32) and
                   expected_run_id != expected_parent_run_id, 'Invalid independent run identities')
    launch, observed, birth = map(common.strict_json, (launch_bytes, raw_bytes, birth_observation_bytes))
    common.require(type(launch['returncode']) is int and launch['returncode'] == 0 and
                   isinstance(launch['job_id'], str) and launch['job_id'].isdigit() and
                   launch['stdout'].strip().split(';')[0] == launch['job_id'] and
                   _number(launch['submitted_unix']) and _digest(launch['source_commit'], 40) and
                   isinstance(launch['source_sha256'], dict) and bool(launch['source_sha256']) and
                   all(_digest(value) for value in launch['source_sha256'].values()),
                   'Missing genuine original successful sbatch/source handle')
    for key in ('base', 'parent_base'):
        path = Path(launch[key])
        common.require(path.is_absolute() and '..' not in path.parts and str(path) == launch[key],
                       'Noncanonical recorded native namespace')
    common.require(launch['base'] != launch['parent_base'], 'Run namespaces collide')
    _snapshot_binding(observed, launch, expected_run_id, expected_parent_run_id)
    _snapshot_binding(birth, launch, expected_run_id, expected_parent_run_id)
    common.require(type(observed['job_returncode']) is int and observed['job_returncode'] == 0 and
                   observed['job_stderr'] == '' and type(birth['job_returncode']) is int and
                   birth['job_returncode'] == 0 and birth['job_stderr'] == '',
                   'Snapshot is not an actual successful scheduler query')
    terminal, initial = _job_fields(observed['job_stdout']), _job_fields(birth['job_stdout'])
    common.require(terminal['JobState'] == expected_state and terminal['ExitCode'] == expected_exit_code and
                   initial['JobState'] == 'RUNNING' and initial['ExitCode'] == '0:0' and
                   all(terminal[key] == initial[key] for key in BIRTH_FIELDS),
                   'No matching actual terminal state and original scheduler job birth')
    command = str(Path(launch['base']) / 'repo/reproduction/common8/launch_calibration_native.sh')
    workdir = str(Path(launch['base']) / 'repo')
    argv = launch['argv']
    common.require(isinstance(argv, list) and all(isinstance(value, str) for value in argv) and
                   argv[0] == 'sbatch' and argv[-1] == command and '--chdir=' + workdir in argv and
                   terminal['JobId'] == launch['job_id'] and terminal['Command'] == command and
                   terminal['WorkDir'] == workdir and terminal['NodeList'] == terminal['BatchHost'] == 'h100-8' and
                   terminal['NumNodes'] == '1' and terminal['MinMemoryNode'] == '512G' and
                   terminal['TresPerNode'] == 'gres:gpu:h100:8' and terminal['Restarts'] == '0',
                   'Original scheduler command/native resource identity changed')
    output = [value.split('=', 1)[1] for value in argv if value.startswith('--output=')]
    common.require(len(output) == 1 and terminal['StdOut'] == terminal['StdErr'] ==
                   output[0].replace('%j', launch['job_id']), 'Original scheduler output path differs')
    submit, start, end = (_scheduler_time(terminal[key]) for key in ('SubmitTime', 'StartTime', 'EndTime'))
    common.require(submit <= start <= end and
                   birth['observation_finished_unix'] <= observed['checked_unix'],
                   'Terminal timestamp is not after the independently observed job birth')
    return dict(role='actual_historical_slurm_terminal_bytes', observation_sha256=expected_sha256,
                launch_sha256=expected_launch_sha256, birth_observation_sha256=expected_birth_sha256,
                job_id=launch['job_id'], run_id=expected_run_id, parent_production_run_id=expected_parent_run_id,
                base=launch['base'], parent_base=launch['parent_base'], source_commit=launch['source_commit'],
                source_map_sha256=common.sha(common.canonical(launch['source_sha256'])),
                checked_unix=observed['checked_unix'], observation_finished_unix=observed['observation_finished_unix'],
                slurm_terminal_state=expected_state, exit_code=expected_exit_code, scheduler_fields=terminal,
                production_admission=False, healthy_checkpoint_accepted=False,
                scope='Original scheduler terminal bytes only; all runtime/health checks remain separate')


def verify_terminal_current(terminal, current_scontrol, current_squeue=None):
    """Purge can preserve an already recorded terminal fact, never create one."""
    saved = terminal['scheduler_fields']
    common.require((terminal['slurm_terminal_state'], terminal['exit_code']) in
                   (('FAILED', '1:0'), ('COMPLETED', '0:0')) and
                   saved['JobState'] == terminal['slurm_terminal_state'] and saved['ExitCode'] == terminal['exit_code'],
                   'Missing explicitly authenticated historical terminal state')
    common.require(type(current_scontrol.returncode) is int, 'Invalid actual scheduler return code')
    if current_scontrol.returncode == 0:
        actual = _job_fields(current_scontrol.stdout)
        common.require(current_scontrol.stderr == '' and
                       all(actual[key] == saved[key] for key in BIRTH_FIELDS +
                           ('JobState', 'ExitCode', 'EndTime', 'NumNodes', 'MinMemoryNode', 'TresPerNode')),
                       'Current scheduler record disagrees with original captured terminal identity')
        route = 'actual_current_terminal_matches_original_terminal_bytes'
    else:
        common.require(not current_scontrol.stdout.strip() and current_scontrol.stderr.strip() == INVALID_JOB and
                       current_squeue is not None and type(current_squeue.returncode) is int and
                       not current_squeue.stdout.strip() and
                       ((current_squeue.returncode == 0 and not current_squeue.stderr.strip()) or
                        (current_squeue.returncode != 0 and current_squeue.stderr.strip() == INVALID_JOB)),
                       'Scoped controller purge and queue absence were not actually observed')
        route = 'original_actual_terminal_bytes_current_scoped_purged'
    return dict(route=route, slurm_terminal_state=terminal['slurm_terminal_state'],
                exit_code=terminal['exit_code'], original_terminal_sha256=terminal['observation_sha256'])


def check_commit(commit):
    common.require(_digest(commit, 40) and commit not in
                   (OLD_COMMIT, '130d17a0d6f0a3ab4b4aee730928f461e25105f0',
                    'a09a5530fd16fa0c04c0aa157ee56462aab825f4',
                    '61f6e1da532e08553410666b430d1933a0e4a49a'),
                   'Explicit new full committed source is required; historical runtimes cannot admit retry')


def previous_schema(observed, old):
    """The child has a real training claim, but no recorded Adam or outer."""
    common.require(old['base'] == OLD_BASE and old['parent_base'] == OLD_PARENT_BASE and old['job_id'] == OLD_JOB and
                   old['source_commit'] == OLD_COMMIT and len(old['source_sha256']) == 353,
                   'Wrong original failed native handle/source')
    common.require(observed['source_hashes_match'] is True and observed['actual_updates'] == [] and
                   observed['actual_behavior_gates'] == [] and observed['active_owned_pids'] == [] and
                   observed['actual_namespace_processes'] == [] and
                   type(observed['run-manifest.json']['completed_updates']) is int and
                   observed['run-manifest.json']['completed_updates'] == 0 and
                   observed['child_manifest_slurm_job_id'] == OLD_JOB and _digest(observed['child_manifest_sha256']),
                   'Actual zero-completed/zero-recorded-write failure identity differs')
    for key in ('pipeline-status.json', 'cleanup-status.json', 'process-ledger.json'):
        common.require(observed[key]['run_id'] == OLD_RUN, 'Wrong captured previous owned run')
    common.require(observed['pipeline-status.json']['stage'] == 'FAILED' and
                   observed['cleanup-status.json']['cleaned'] is True and
                   all(child['state'] == 'STOPPED' for child in observed['process-ledger.json']['jobs'].values()) and
                   type(observed['process-ledger.json']['jobs']['trainer']['exit_code']) is int and
                   observed['process-ledger.json']['jobs']['trainer']['exit_code'] == 1,
                   'Original failed workflow/trainer exit/owned cleanup differs')
    child = observed['child_actual_unstarted']
    common.require(child == dict(training_claim_exists=True, optimizer_evidence_paths=[], train_log_exists=True,
                                 checkpoint_entries=[], actual_updates_exists=False, segment_exists=False,
                                 segment_complete_exists=False) and observed['child_actual_unstarted_verified'] is False,
                   'Previous failed training claim is not zero recorded writes/zero completed outer')
    parent = observed['parent_run_manifest']
    common.require(parent['run_id'] == OLD_PARENT_RUN and parent['source_commit'] == OLD_COMMIT and
                   type(parent['completed_updates']) is int and parent['completed_updates'] == 0 and
                   _digest(parent['manifest_sha256']) and observed['parent_actual_unstarted_verified'] is True and
                   observed['parent_actual_unstarted'] == dict(training_claim_exists=False, optimizer_evidence_paths=[],
                       train_log_exists=False, checkpoint_entries=[], actual_updates_exists=False,
                       segment_exists=False, segment_complete_exists=False), 'Original production parent is not fresh0')
    common.require('BehaviorTraceError: included_special_or_pad_token' in observed['trainer.log_tail'],
                   'Wrong original first-rollout boundary failure; do not relabel it as KL/Adam failure')
    return observed


def original_inputs(prep=PREP, *, previous_source_cleanup_path=None, previous_source_cleanup_sha256=None):
    launch_bytes = (prep / OLD_LAUNCH_NAME).read_bytes()
    terminal_bytes, birth_bytes = (prep / TERMINAL_NAME).read_bytes(), (prep / BIRTH_NAME).read_bytes()
    terminal = read_terminal_observation(terminal_bytes, expected_sha256=TERMINAL_SHA256,
        launch_bytes=launch_bytes, expected_launch_sha256=OLD_LAUNCH_SHA256, expected_run_id=OLD_RUN,
        expected_parent_run_id=OLD_PARENT_RUN, expected_state='FAILED', expected_exit_code='1:0',
        birth_observation_bytes=birth_bytes, expected_birth_sha256=BIRTH_SHA256)
    old, observed = common.strict_json(launch_bytes), common.strict_json(terminal_bytes)
    previous_schema(observed, old)
    cpu_bytes = (prep / CPU_OLD_NAME).read_bytes(); cpu = common.strict_json(cpu_bytes)
    common.require(common.sha(cpu_bytes) == old['target_cpu_proof_sha256'] and cpu['source_sha256'] == old['source_sha256'] and
                   common.strict_json(cpu['stdout'])['tests_run'] == 239 and tuple(cpu['modules']) == CPU_MODULES,
                   'Original actual239 proof/source bytes changed')
    handle_bytes = (prep / SOURCE_HANDLE_NAME).read_bytes()
    common.require(common.sha(handle_bytes) == SOURCE_HANDLE_SHA256, 'Original v4 tiny source handle bytes changed')
    handle = common.strict_json(handle_bytes)
    common.require(previous_source_cleanup_path is not None and _digest(previous_source_cleanup_sha256),
                   'Explicit root-sealed actual old tiny-source cleanup bytes are required')
    cleanup_bytes = Path(previous_source_cleanup_path).read_bytes()
    common.require(common.sha(cleanup_bytes) == previous_source_cleanup_sha256,
                   'Original actual old tiny-source cleanup SHA changed')
    cleanup = common.strict_json(cleanup_bytes)
    common.require(cleanup['cleaned'] is True and cleanup['handle'] == handle and _number(cleanup['checked_unix']),
                   'Original source service has not been scoped-cleaned')
    return old, cpu, dict(previous_terminal_bytes=base64.b64encode(terminal_bytes).decode(),
                          previous_birth_bytes=base64.b64encode(birth_bytes).decode(),
                          previous_launch_bytes=base64.b64encode(launch_bytes).decode(),
                          previous_terminal=terminal, previous_source_cleanup_sha256=common.sha(cleanup_bytes))


def validate_cpu(proof, old_proof, expected_cpu_tests):
    common.require(type(expected_cpu_tests) is int and expected_cpu_tests > 0,
                   'Explicit root-provided new actual CPU test count is required')
    summary = common.strict_json(proof['stdout'])
    common.require(type(proof['returncode']) is int and proof['returncode'] == 0 and
                   type(proof['shell_syntax_returncode']) is int and proof['shell_syntax_returncode'] == 0 and
                   summary['successful'] is True and summary['cuda_available'] is False and summary['skipped'] == [] and
                   type(summary['tests_run']) is int and summary['tests_run'] == expected_cpu_tests and
                   type(summary['failures']) is int and summary['failures'] == 0 and
                   type(summary['errors']) is int and summary['errors'] == 0 and
                   tuple(proof['modules']) == CPU_MODULES and len(proof['source_sha256']) == 354 and
                   all(_digest(value) for value in proof['source_sha256'].values()) and
                   proof['source_sha256'] != old_proof['source_sha256'] and _number(proof['checked_unix']) and
                   proof['checked_unix'] > old_proof['checked_unix'],
                   'Explicit new sixteen-suite/count/full354/zero-skip target CPU proof is required')
    path = Path(proof['base'])
    common.require(path.is_absolute() and str(path) == proof['base'] and '..' not in path.parts and
                   (proof['base'].startswith('/dataset/shaoshitong/') or proof['base'].startswith('/mnt/data/shaoshitong/')),
                   'Not a canonical target-cluster CPU proof')
    return proof['source_sha256']


def committed_sources(commit, mapping, repo=REPO):
    """Exact complete354 Git tree/blobs, from the explicit immutable commit."""
    check_commit(commit)
    names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', commit], cwd=repo, text=True).splitlines()
    expected = {name for name in names if
        (Path(name).suffix in ('.py', '.sh', '.yaml', '.yml', '.json') or
         name in ('verl/version/version', 'verl/single_controller/version/version')) and
        not {'.git', '__pycache__', 'evidence'}.intersection(Path(name).parts)}
    common.require(len(expected) == 354 and expected == set(mapping) and
                   {'verl/version/version', 'verl/single_controller/version/version'} <= expected,
                   'CPU proof is not the complete committed354 map including both runtime version files')
    files = {}
    for name in sorted(expected):
        content = subprocess.check_output(['git', 'cat-file', 'blob', commit + ':' + name], cwd=repo)
        common.require(common.sha(content) == mapping[name], 'Source changed after actual CPU proof: ' + name)
        files[name] = content
    return files


def load_inputs(commit=None, cpu_path=None, expected_cpu_tests=None, *, target_cpu_proof_sha256=None,
                previous_source_set_path=None, previous_source_set_sha256=None,
                previous_source_cleanup_path=None, previous_source_cleanup_sha256=None, prep=PREP, repo=REPO):
    check_commit(commit)
    common.require(cpu_path is not None and type(expected_cpu_tests) is int and expected_cpu_tests > 0 and
        _digest(target_cpu_proof_sha256) and previous_source_set_path is not None and
        _digest(previous_source_set_sha256) and previous_source_cleanup_path is not None and
        _digest(previous_source_cleanup_sha256), 'Explicit new CPU SHA/count and root-sealed old source/cleanup inputs are required')
    old, old_cpu, history = original_inputs(prep, previous_source_cleanup_path=previous_source_cleanup_path,
        previous_source_cleanup_sha256=previous_source_cleanup_sha256)
    source_set_bytes = Path(previous_source_set_path).read_bytes()
    source_set = read_source_set_observation(source_set_bytes, previous_source_set_sha256)
    common.require(source_set['checked_unix'] >= history['previous_terminal']['checked_unix'],
                   'Original stopped-run source observation must follow its captured failure')
    history['previous_source_set_bytes'] = base64.b64encode(source_set_bytes).decode()
    history['previous_source_set_sha256'] = previous_source_set_sha256
    cpu_bytes = Path(cpu_path).read_bytes(); proof = common.strict_json(cpu_bytes)
    common.require(common.sha(cpu_bytes) == target_cpu_proof_sha256 and
                   target_cpu_proof_sha256 != old['target_cpu_proof_sha256'], 'Actual new CPU bytes SHA differs or is old239')
    files = committed_sources(commit, validate_cpu(proof, old_cpu, expected_cpu_tests), repo)
    new_recipe(files)
    return dict(old_launch=old, **history, target_cpu_proof_sha256=common.sha(cpu_bytes), source_commit=commit,
                source_sha256=proof['source_sha256'], actual_target_cpu_tests=expected_cpu_tests), files


def verify_old_manifest(raw, observed):
    common.require(common.sha(raw) == observed['child_manifest_sha256'], 'Original full failed manifest bytes changed')
    run = common.strict_json(raw); subset = observed['run-manifest.json']
    required = {'run_id', 'completed_updates', 'source_commit', 'train_log', 'checkpoint_dir',
                'policy_calibration_role', 'parent_production_run_id'}
    common.require(set(subset) == required and all(run.get(key) == value for key, value in subset.items()) and
                   run['run_id'] == OLD_RUN and run['source_commit'] == OLD_COMMIT and run['profile'] == 'common8' and
                   run['parent_production_run_id'] == OLD_PARENT_RUN and run['policy_calibration_slurm_job_id'] == OLD_JOB and
                   type(run['completed_updates']) is int and run['completed_updates'] == 0,
                   'Actual original failed full manifest/run/job differs')
    return run


def previous_readonly(payload):
    """Actual old file/source/process/cleanup reread; never resume/reset the child."""
    terminal_bytes = base64.b64decode(payload['previous_terminal_bytes'], validate=True)
    terminal = read_terminal_observation(terminal_bytes, expected_sha256=TERMINAL_SHA256,
        launch_bytes=base64.b64decode(payload['previous_launch_bytes'], validate=True),
        expected_launch_sha256=OLD_LAUNCH_SHA256, expected_run_id=OLD_RUN, expected_parent_run_id=OLD_PARENT_RUN,
        expected_state='FAILED', expected_exit_code='1:0',
        birth_observation_bytes=base64.b64decode(payload['previous_birth_bytes'], validate=True),
        expected_birth_sha256=BIRTH_SHA256)
    old = payload['old_launch']
    common.require(common.canonical(old) == common.canonical(common.strict_json(
        base64.b64decode(payload['previous_launch_bytes'], validate=True))), 'Original launch payload differs')
    observed = previous_schema(common.strict_json(terminal_bytes), old)
    b = Path(old['base']); common.require(b.resolve() == b, 'Original namespace symlink rejected')
    # First authenticate all declared353 bytes, before importing any old code.
    for namespace in (b, Path(old['parent_base'])):
        _verify_original_mapping(namespace, old['source_sha256'])
    source_set = read_source_set_observation(base64.b64decode(payload['previous_source_set_bytes'], validate=True),
                                           payload['previous_source_set_sha256'])
    common.require(source_set['checked_unix'] >= terminal['checked_unix'],
                   'Old source snapshot predates the actual terminal capture')
    # Then separately authenticate only the original three known generated files.
    for namespace, captured in zip((b, Path(old['parent_base'])), source_set['namespaces']):
        _verify_original_source_set(namespace, old['source_sha256'], captured)
    sys.path.insert(0, str(b / 'repo'))
    from reproduction.a800.owned_processes import same, identity
    from reproduction.common8.freeze_calibration_plan import _unstarted
    run = verify_old_manifest((b / 'logs/run-manifest.json').read_bytes(), observed)
    for name in ('pipeline-status.json', 'cleanup-status.json', 'process-ledger.json'):
        common.require(common.strict_json((b / 'logs' / name).read_bytes()) == observed[name],
                       'Actual previous failure/cleanup/ledger bytes changed')
    common.require(not list((b / 'logs').glob('optimizer-rank-*')) and
                   not list((b / 'logs').glob('policy-calibration-optimizer-rank-*')) and
                   not list((b / 'logs').glob('behavior-probability-outer-*.json')),
                   'Previous failed run acquired optimizer/write/gate evidence')
    for name in ('actual-updates.jsonl', 'policy-calibration-segment.json', 'policy-calibration-segment-complete.json'):
        common.require(not (b / 'logs' / name).exists(), 'Previous failure acquired a completed outer/segment')
    checkpoints = Path(run['checkpoint_dir'])
    common.require(checkpoints == b / 'checkpoints/search-g1-common8' and
                   (not checkpoints.exists() or not any(checkpoints.iterdir())), 'Original failed checkpoint namespace changed')
    training = Path(run['train_log'])
    common.require(training == b / 'logs/search-g1-common8-training.log' and training.is_file() and
                   (b / 'logs/training-launch-claim.json').exists(), 'Original real attempted training claim/log is absent')
    claim = common.strict_json((b / 'logs/training-launch-claim.json').read_bytes())
    common.require(claim['run_id'] == OLD_RUN and claim['slurm_job_id'] == OLD_JOB and
                   type(claim['shell_pid']) is int and claim['shell_pid'] > 0 and
                   type(claim['shell_start_ticks']) is int and claim['shell_start_ticks'] > 0 and
                   _number(claim['claimed_unix']) and claim['claimed_unix'] <= observed['pipeline-status.json']['unix'],
                   'Original attempted trainer claim/run/birth differs')
    common.require((b / 'logs/trainer.log').read_text(errors='replace').endswith(observed['trainer.log_tail']),
                   'Actual original first-rollout failure log changed')
    for child in observed['process-ledger.json']['jobs'].values():
        for process in child['ledger'].values():
            common.require(not same(process) or identity(process['pid'])['state'] == 'Z', 'Old owned process remains active')
    marker = ('SEARCH_G1_BASE=' + str(b)).encode()
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit(): continue
        try:
            if proc.stat().st_uid != os.getuid() or marker not in (proc / 'environ').read_bytes().split(b'\0'): continue
            common.require((proc / 'stat').read_text().rsplit(')', 1)[1].split()[0] == 'Z', 'Original namespace still active')
        except (FileNotFoundError, ProcessLookupError): pass
    root = Path('/dev/shm') / ('search-g1-common8-' + OLD_RUN + '-' + OLD_JOB)
    for name, target in (('retrieval-staging-claim.json', root), ('corpus-cache-claim.json', root / 'datasets-cache')):
        claim = common.strict_json((b / 'logs' / name).read_bytes())
        common.require(claim['run_id'] == OLD_RUN and claim['slurm_job_id'] == OLD_JOB and claim['uid'] == os.getuid() and
                       claim['phase'] == 'CLEANED' and Path(claim['root']) == target, 'Original owned cleanup claim differs')
    common.require(not root.exists() and not root.is_symlink(), 'Original owned shm root remains')
    parent = Path(old['parent_base']); raw = (parent / 'logs/run-manifest.json').read_bytes()
    current_parent = common.strict_json(raw)
    common.require(parent.resolve() == parent and common.sha(raw) == observed['parent_run_manifest']['manifest_sha256'] and
                   current_parent['run_id'] == OLD_PARENT_RUN and current_parent['source_commit'] == OLD_COMMIT,
                   'Original unstarted parent bytes changed')
    _unstarted(parent, current_parent)
    command = subprocess.run(['scontrol', 'show', 'job', '-o', OLD_JOB], capture_output=True, text=True, timeout=15)
    queue = None
    if command.returncode != 0:
        queue = subprocess.run(['squeue', '-h', '-j', OLD_JOB, '-o', '%i|%T|%j|%Z|%o'],
                               capture_output=True, text=True, timeout=15)
    route = verify_terminal_current(terminal, command, queue)
    return dict(role='previous_recorded_zero_write_discovery_readonly_verification', run_id=OLD_RUN, job_id=OLD_JOB,
                source_commit=OLD_COMMIT, actual_completed_updates=0, recorded_optimizer_write_marker_count=0,
                optimizer_state_tensor_verification_performed=False,
                original_failed_run_preserved=True, terminal_observation=route, owned_namespace_empty=True,
                owned_shm_root_absent=True, production_admission=False, healthy_checkpoint_accepted=False,
                original_generated_hydra_snapshots_proof_sha256=payload['previous_source_set_sha256'],
                declared_source_file_count=353, original_generated_snapshot_count=3,
                scope='Original actual FAILED/1:0 + zero recorded writes/outer + unchanged source/files/owned cleanup')


def previous_code():
    helpers = '\n'.join(inspect.getsource(fn) for fn in (common.canonical, common.sha, common.require, common.strict_json))
    names = (_digest, _number, source_set_schema, read_source_set_observation, _verify_original_mapping,
             _verify_original_source_set, _verify_metadata_file, _job_fields, _scheduler_time, _snapshot_binding, read_terminal_observation,
             verify_terminal_current, previous_schema, verify_old_manifest, previous_readonly)
    functions = '\n'.join(inspect.getsource(fn) for fn in names)
    constants = '\n'.join(name + '=' + repr(globals()[name]) for name in
        ('OLD_COMMIT','OLD_RUN','OLD_PARENT_RUN','OLD_BASE','OLD_PARENT_BASE','OLD_JOB','TERMINAL_SHA256',
         'OLD_LAUNCH_SHA256','BIRTH_SHA256','INVALID_JOB','BIRTH_FIELDS'))
    return ('import base64,datetime,hashlib,json,math,os,re,stat,subprocess,sys\nfrom pathlib import Path\nfrom types import SimpleNamespace\n' +
            helpers + '\ncommon=SimpleNamespace(canonical=canonical,sha=sha,require=require,strict_json=strict_json)\n' +
            constants + '\n' + functions + '\np=json.load(sys.stdin);print(json.dumps(previous_readonly(p)))\n')


def new_recipe(files):
    """Check effective literal overrides, rejecting duplicate scientific keys."""
    recipe = common.strict_json(files['reproduction/common8/recipe.json'])
    common.require(isinstance(recipe, list) and all(isinstance(value, str) and '=' in value for value in recipe),
                   'Invalid committed common8 recipe')
    effective = {}
    for override in recipe:
        key, value = override.split('=', 1)
        common.require(key not in effective, 'Duplicate effective recipe override')
        effective[key] = value
    expected = {'actor_rollout_ref.rollout.top_p': '1.0', 'actor_rollout_ref.rollout.temperature': '1',
                'actor_rollout_ref.rollout.n_agent': '5', 'actor_rollout_ref.rollout.n': '1',
                'data.train_batch_size': '512', 'actor_rollout_ref.actor.ppo_mini_batch_size': '128',
                'actor_rollout_ref.actor.ppo_micro_batch_size': '16',
                'actor_rollout_ref.rollout.log_prob_micro_batch_size': '32',
                'actor_rollout_ref.ref.log_prob_micro_batch_size': '32',
                'actor_rollout_ref.actor.ppo_epochs': '1', 'actor_rollout_ref.actor.optim.lr': '1e-6'}
    common.require(all(effective.get(key) == value for key, value in expected.items()),
                   'New neutral sampling/scientific budget recipe differs')
    for name in ('sampling_capture', 'behavior_trace', 'behavior_probability', 'behavior_runtime', 'generation_behavior'):
        common.require('reproduction/common8/' + name + '.py' in files, 'Missing actual behavior source')
    return effective


PREPARE_CODE = r'''
import base64,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
p=json.load(sys.stdin);old=Path(p['old_launch']['parent_base'])
for directory in (p['parent_base'],p['base']):
 b=Path(directory);assert not b.exists();b.mkdir(parents=True,exist_ok=False)
 for name in ('repo','data','preparation','logs','tmp','ray','artifacts','cache/triton','cache/inductor','cache/hf'):(b/name).mkdir(parents=True,exist_ok=True)
 for name in ('env','bootstrap-assets','bootstrap-tools','assets','retriever-gpu-lib'):
  source=old/name;assert source.is_dir();(b/name).symlink_to(source.resolve(),target_is_directory=True)
 for name,data in p['files'].items():
  relative=Path(name);assert not relative.is_absolute() and '..' not in relative.parts
  target=b/'repo'/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(base64.b64decode(data))
  assert hashlib.sha256(target.read_bytes()).hexdigest()==p['source_sha256'][name]
 for name in ('manifest.json','train.parquet','validation.parquet','stage.parquet','final.parquet','exclusions-audit.json'):
  source=old/'data'/name;target=b/'data'/name;assert not source.is_symlink();shutil.copy2(source,target)
  assert hashlib.sha256(source.read_bytes()).hexdigest()==hashlib.sha256(target.read_bytes()).hexdigest()
 (b/'preparation/official-model-manifest-original.json').write_bytes((old/'preparation/official-model-manifest-original.json').read_bytes())
 with (b/'preparation/source-package-intent.json').open('x') as f:json.dump({k:v for k,v in p.items() if k!='files'},f,indent=2);f.flush();os.fsync(f.fileno())
b=Path(p['base']);env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(b/'repo'),HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1')
code='import verl,verl.single_controller;from verl.workers.fsdp_workers import ActorRolloutRefWorker;from reproduction.common8 import prepare_calibration_run,calibration_runtime,temporary_cache,adopt_initial_probes,production_safety_admission,behavior_trace,behavior_probability,behavior_runtime,generation_behavior;from reproduction.common8.verify_data import verify;import pathlib,sys,json,inspect,hashlib;from types import SimpleNamespace;from ray._private.node import Node;s=inspect.getsource(Node._prepare_socket_file);good="/"+"a"*106;dummy=SimpleNamespace(_sockets_dir="/unused",_make_inc_temp=lambda **kwargs:good);assert sys.platform=="linux" and Node._prepare_socket_file(dummy,None,"plasma_store")==good\ntry:\n dummy._make_inc_temp=lambda **kwargs:"/"+"a"*107;Node._prepare_socket_file(dummy,None,"plasma_store")\nexcept OSError:pass\nelse:raise AssertionError("Actual installed Ray accepted108-byteAF_UNIX")\nprint(json.dumps(dict(data=verify(pathlib.Path(sys.argv[1])),installed_ray_socket_check_sha256=hashlib.sha256(s.encode()).hexdigest(),installed_limit_tested_bytes=[107,108],scope="Actual installed AF_UNIX method CPU boundary check; no Ray/GPU startup")))'
r=subprocess.run([str(b/'env/bin/python'),'-c',code,str(b/'data')],cwd=b/'repo',env=env,capture_output=True,text=True,timeout=75)
proof=dict(base=str(b),parent_base=p['parent_base'],checked_unix=time.time(),returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,source_commit=p['source_commit'],source_sha256=p['source_sha256'],ray_socket_budgets=p['ray_socket_budgets'],actual_production_updates=0,production_admission=False,healthy_checkpoint_accepted=False,scope='CPU imports+originalsixbytes+actualinstalledRayboundary; not GPU admission or new training')
with (b/'preparation/package-cpu-proof.json').open('x') as f:json.dump(proof,f,indent=2);f.flush();os.fsync(f.fileno())
assert r.returncode==0,r.stderr
syntax=subprocess.run(['bash','-n',str(b/'repo/reproduction/common8/launch_calibration_native.sh'),str(b/'repo/reproduction/common8/train.sh')],capture_output=True,text=True);assert syntax.returncode==0,syntax.stderr
print(json.dumps(proof))
'''


SUBMIT_CODE = r'''
import hashlib,json,os,subprocess,sys,time,urllib.request
from pathlib import Path
p=json.load(sys.stdin);b=Path(p['base']);parent=Path(p['parent_base'])
for namespace in (b,parent):
 assert not (namespace/'logs/run-manifest.json').exists()
 for name,digest in p['source_sha256'].items():assert hashlib.sha256((namespace/'repo'/name).read_bytes()).hexdigest()==digest
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
for name,size in p['handle']['files'].items():
 with opener.open(urllib.request.Request('http://10.33.0.13:19874/'+name,method='HEAD'),timeout=10) as response:assert response.status==200 and int(response.headers['Content-Length'])==size
args=['sbatch','--parsable','--partition=gpu','--nodelist=h100-8','--nodes=1','--ntasks=1','--cpus-per-task=64','--gres=gpu:h100:8','--mem=512G','--exclusive','--time=08:00:00','--job-name=search-g1-policy-discovery-v6','--chdir='+str(b/'repo'),'--output='+str(b/'logs/native-discovery-v6-%j.log'),'--export=ALL,SEARCH_G1_BASE='+str(b)+',SEARCH_G1_PRODUCTION_BASE='+str(parent)+',SEARCH_G1_CALIBRATION_ROLE=policy_safety_calibration_discovery,SEARCH_G1_SOURCE_COMMIT='+p['source_commit'],str(b/'repo/reproduction/common8/launch_calibration_native.sh')]
with (b/'preparation/native-submit-claim.json').open('x') as f:json.dump(dict(argv=args,created_unix=time.time(),base=str(b),parent_base=str(parent),target_cpu_proof_sha256=p['target_cpu_proof_sha256'],actual_target_cpu_tests=p['actual_target_cpu_tests'],scope='Independent retry after stoppedfailed3294; production0'),f);f.flush();os.fsync(f.fileno())
with (b/'preparation/retrieval-source-handle.json').open('x') as f:json.dump(p['handle'],f,indent=2);f.flush();os.fsync(f.fileno())
fd=os.open(str(b/'preparation'),os.O_DIRECTORY)
try:os.fsync(fd)
finally:os.close(fd)
r=subprocess.run(args,capture_output=True,text=True)
proof=dict(base=str(b),parent_base=str(parent),source_commit=p['source_commit'],source_sha256=p['source_sha256'],target_cpu_proof_sha256=p['target_cpu_proof_sha256'],actual_target_cpu_tests=p['actual_target_cpu_tests'],argv=args,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,submitted_unix=time.time(),actual_production_updates=0)
if r.returncode==0:proof['job_id']=r.stdout.strip().split(';')[0];assert proof['job_id'].isdigit()
with (b/'preparation/native-launch.json').open('x') as f:json.dump(proof,f,indent=2);f.flush();os.fsync(f.fileno())
assert r.returncode==0,r.stderr;print(json.dumps(proof))
'''


def submit(commit=None, cpu_path=None, expected_cpu_tests=None, *, target_cpu_proof_sha256=None,
           previous_source_set_path=None, previous_source_set_sha256=None,
           previous_source_cleanup_path=None, previous_source_cleanup_sha256=None, prep=PREP, repo=REPO):
    common.require(not (prep / NAMES['intent']).exists(), 'v6 already reserved; observe same handle, never repeat')
    inputs, files = load_inputs(commit, cpu_path, expected_cpu_tests,
        target_cpu_proof_sha256=target_cpu_proof_sha256,
        previous_source_set_path=previous_source_set_path, previous_source_set_sha256=previous_source_set_sha256,
        previous_source_cleanup_path=previous_source_cleanup_path,
        previous_source_cleanup_sha256=previous_source_cleanup_sha256, prep=prep, repo=repo)
    token = uuid.uuid4().hex[:8]
    parent = '/mnt/data/shaoshitong/g1p-' + token
    base = '/mnt/data/shaoshitong/g1d-' + token
    budgets = dict(parent=common.ray_socket_budget(parent), discovery=common.ray_socket_budget(base))
    # New source and prior cleanup are independently read before any mutation.
    previous = common.remote('h100-8', previous_code(), inputs, timeout=75,
                             interpreter=inputs['old_launch']['base'] + '/env/bin/python')
    export = '/dataset/shaoshitong/g1-original-export-discovery-v6-' + token
    source = files['reproduction/common8/assets_export.py'].decode()
    needle = "BASE=Path('/dataset/shaoshitong/g1-common8-original-retrieval-export')"
    common.require(source.count(needle) == 1, 'Explicit new source asset exporter mismatch')
    source = source.replace(needle, 'BASE=Path(' + repr(export) + ')').encode()
    intent = dict(inputs, base=base, parent_base=parent, export_base=export,
                  export_source_sha256=common.sha(source), created_unix=time.time(),
                  external_orchestrator_sha256=common.sha(Path(__file__).read_bytes()),
                  external_common_helper_sha256=common.sha(Path(common.__file__).read_bytes()),
                  ray_socket_budgets=budgets, previous_readonly_proof=previous,
                  actual_production_updates=0, scope='New official0 parent and discovery; original failed3294 remains immutable; recorded0 is not an Adam-state tensor audit')
    common.save(prep / NAMES['intent'], intent)
    payload = dict(intent, files={name: base64.b64encode(data).decode() for name, data in files.items()})
    package = common.remote('h100-8', PREPARE_CODE, payload)
    common.save(prep / NAMES['package'], package)
    # Package includes real installedRay boundary + original data byte proof;
    # source process starts only after this preparation has actually succeeded.
    handle = common.remote('h100-9', common.START_SOURCE_CODE, dict(export=export,
                            source=base64.b64encode(source).decode(), source_sha256=common.sha(source)))
    common.save(prep / NAMES['handle'], handle)
    common.remote('h100-8', previous_code(), inputs, timeout=75,
                  interpreter=inputs['old_launch']['base'] + '/env/bin/python')
    launch = common.remote('h100-8', SUBMIT_CODE, dict(intent, handle=handle))
    common.save(prep / NAMES['launch'], launch)
    return {key: value for key, value in launch.items() if key not in ('source_sha256', 'argv')}


OBSERVE_CODE = common.OBSERVE_CODE.replace('import json,os,subprocess,sys,time',
    'import hashlib,json,os,subprocess,sys,time').replace('print(json.dumps(out))', r'''
out['parent_base']=p['parent_base'];out['job_id']=job
mapping=p['source_sha256']
out['source_hashes_match']=all((b/'repo'/name).resolve()==b/'repo'/name and
 hashlib.sha256((b/'repo'/name).read_bytes()).hexdigest()==digest for name,digest in mapping.items())
manifest=b/'logs/run-manifest.json'
if manifest.exists():
 raw=manifest.read_bytes();run=json.loads(raw);out['child_manifest_sha256']=hashlib.sha256(raw).hexdigest()
 out['child_manifest_slurm_job_id']=run['policy_calibration_slurm_job_id']
 out['run-manifest.json']={key:run[key] for key in ('run_id','completed_updates','source_commit','train_log',
  'checkpoint_dir','policy_calibration_role','parent_production_run_id')}
out['observation_finished_unix']=time.time()
print(json.dumps(out))
''')


def observe(prep=PREP, cleanup_source=False):
    intent = common.strict_json((prep / NAMES['intent']).read_bytes())
    observation = common.remote('h100-8', OBSERVE_CODE, intent)
    suffix = str(time.time_ns())
    common.save(prep / ('policy-discovery-v6-observation-' + suffix + '.json'), observation)
    if not cleanup_source:
        return observation
    common.require(common.cleanup_eligible(observation), 'Original assets still needed or native/namespace state unknown')
    path = prep / NAMES['handle']
    handle = common.strict_json(path.read_bytes()) if path.exists() else common.remote('h100-9', common.SOURCE_HANDLE_CODE, intent)
    common.require(handle['export'] == intent['export_base'] and
                   handle['source_sha256'] == intent['export_source_sha256'], 'Wrong v6 source handle')
    proof = common.remote('h100-9', common.CLEANUP_SOURCE_CODE, handle)
    common.save(prep / ('policy-discovery-v6-retrieval-source-cleanup-' + suffix + '.json'), proof)
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--submit', action='store_true'); group.add_argument('--observe', action='store_true')
    group.add_argument('--cleanup-source', action='store_true'); group.add_argument('--check-inputs', action='store_true')
    parser.add_argument('--source-commit'); parser.add_argument('--target-cpu-proof')
    parser.add_argument('--target-cpu-proof-sha256')
    parser.add_argument('--expected-cpu-tests', type=int)
    parser.add_argument('--previous-source-set-proof'); parser.add_argument('--previous-source-set-sha256')
    parser.add_argument('--previous-source-cleanup-proof'); parser.add_argument('--previous-source-cleanup-sha256')
    args = parser.parse_args()
    inputs = dict(target_cpu_proof_sha256=args.target_cpu_proof_sha256,
        previous_source_set_path=args.previous_source_set_proof,
        previous_source_set_sha256=args.previous_source_set_sha256,
        previous_source_cleanup_path=args.previous_source_cleanup_proof,
        previous_source_cleanup_sha256=args.previous_source_cleanup_sha256)
    if args.observe or args.cleanup_source:
        common.require(not any((args.source_commit, args.target_cpu_proof, args.expected_cpu_tests, *inputs.values())),
                       'Observe only the original saved v6 handle')
        result = observe(cleanup_source=args.cleanup_source)
    else:
        common.require(all((args.source_commit, args.target_cpu_proof, args.expected_cpu_tests, *inputs.values())),
                       'Explicit new full commit/CPU SHA/count and sealed old source/cleanup proof paths/SHAs are required')
        if args.submit:
            result = submit(args.source_commit, args.target_cpu_proof, args.expected_cpu_tests, **inputs)
        else:
            checked, files = load_inputs(args.source_commit, args.target_cpu_proof, args.expected_cpu_tests, **inputs)
            result = dict(role='v6_discovery_local_structure_check', source_commit=checked['source_commit'],
                          runtime_file_count=len(files), actual_target_cpu_tests=checked['actual_target_cpu_tests'],
                          actual_production_updates=0, production_admission=False, healthy_checkpoint_accepted=False,
                          physical_execution_authenticated=False, scope='No remote call or service/submission')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
