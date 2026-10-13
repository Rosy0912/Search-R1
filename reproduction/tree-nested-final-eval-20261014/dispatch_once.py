"""Request one owned final evaluator after verified completed training; never retry blindly."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import time
import uuid

from contract import (FROZEN_SOURCE, GPU_UUIDS, ROOT, RUN, TRAIN_UNIT, read_json,
                      require, sha, verify_code_manifest, verify_handoff)
from eval_merge import atomic_immutable


def owned_unit(unit):
    require(isinstance(unit, str) and len(unit) <= 255
            and re.fullmatch(r'sst-tree-nested-final-eval-[A-Za-z0-9_-]+\.service', unit)
            and unit != TRAIN_UNIT, 'A distinct owned final-evaluation service is required')
    return unit


def ordinary_directory(path):
    """Allow the already bound retained mount; reject redirects inside this run."""
    path = Path(path)
    require(path.is_absolute() and '..' not in path.parts and path.is_relative_to(RUN),
            'Owned retained-run directory required')
    current = RUN
    require(current.is_dir(), 'Bound original retained run is unavailable')
    for part in path.relative_to(RUN).parts:
        current /= part
        if not current.exists() and not current.is_symlink():
            current.mkdir()
        info = current.lstat()
        require(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode),
                'Persistence directory inside the run is redirected: ' + str(current))
    require(path.resolve().is_relative_to(RUN.resolve()), 'Persistence leaves retained storage')


def unit_state(unit, environment, run):
    result = run(['systemctl', 'show', unit,
                  '--property=LoadState,ActiveState,MainPID,ControlGroup'],
                 env=environment, text=True, capture_output=True, timeout=15)
    require(result.returncode == 0 and not result.stderr,
            'Service state lookup is ambiguous; inspect the exact owned service')
    fields = {}
    for line in result.stdout.splitlines():
        require('=' in line, 'Malformed service state')
        key, value = line.split('=', 1)
        require(key not in fields, 'Duplicate service state field')
        fields[key] = value
    require(set(fields) == {'LoadState', 'ActiveState', 'MainPID', 'ControlGroup'},
            'Incomplete service state')
    return fields


def empty_owned_cgroup(unit, state):
    relative = state['ControlGroup'] or '/system.slice/' + unit
    require(relative == '/system.slice/' + unit, 'Unexpected owned service cgroup')
    observations = []
    for base in ('/sys/fs/cgroup', '/sys/fs/cgroup/systemd', '/sys/fs/cgroup/freezer'):
        scope = Path(base) / relative.lstrip('/')
        if not scope.exists():
            continue
        require(scope.is_dir() and not scope.is_symlink(), 'Owned cgroup is redirected')
        require((scope / 'cgroup.procs').is_file(), 'Owned cgroup process list is unavailable')
        for path in sorted(scope.rglob('cgroup.procs')):
            pids = path.read_text().split()
            require(all(pid.isdecimal() for pid in pids), 'Malformed owned cgroup process list')
            require(not pids, 'Previous owned evaluator still has live cgroup processes')
            observations.append(str(path))
    return observations


def prior_admission(directory, handoff, environment, run):
    require(not (directory / 'scores' / 'final-metrics.json').exists()
            and not (directory / 'scores' / 'final-metrics.json').is_symlink(),
            'This final-evaluation goal already has metrics; do not restart it')
    records = []
    for path in sorted(directory.glob('*dispatch*once*.json')):
        info = path.lstat()
        require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1,
                'Previous once journal is not an ordinary owned file')
        record = read_json(path)
        require(record.get('evaluation_only') is True and record.get('run') == str(RUN)
                and record.get('train_unit') == TRAIN_UNIT,
                'Previous once journal has a different evaluation identity')
        require(record.get('status') != 'COMPLETE', 'Final evaluation is already complete')
        require(record.get('status') == 'REQUESTED',
                'Previous launch is unknown; reconcile its exact service without retrying')
        records.append((path, sha(path), record))
    previous = []
    for path, journal_sha, record in records:
        unit = owned_unit(record.get('unit'))
        observed = unit_state(unit, environment, run)
        require(observed['ActiveState'] in ('inactive', 'failed') and observed['MainPID'] == '0',
                'Previous owned evaluator is still active; no duplicate launch')
        terminal_path = ROOT / 'runs' / unit.removesuffix('.service') / 'run-result.json'
        pin = handoff.get('prior_evaluator_terminal')
        terminal_sha = sha(terminal_path)
        expected_pin = dict(path=str(terminal_path), sha256=terminal_sha)
        if pin != expected_pin:
            # Earlier failed launches remain durable: a later requested once journal
            # must already have explicitly pinned and admitted that exact predecessor.
            admitted = [proof for successor_path, _, successor in records if successor_path != path
                        for proof in successor.get('prior_evaluators', [])
                        if proof.get('journal') == str(path) and proof.get('journal_sha256') == journal_sha
                        and proof.get('unit') == unit and proof.get('terminal') == expected_pin]
            require(bool(admitted), 'Explicit handoff must pin the latest previous evaluator terminal result')
        pin = expected_pin
        terminal = read_json(terminal_path)
        require(terminal.get('status') != 'COMPLETE', 'Previous evaluator completed this goal')
        require(terminal.get('status') == 'FAILED' and terminal.get('unit') == unit
                and terminal.get('evaluation_only') is True
                and terminal.get('own_cleanup_residual') == {},
                'Previous evaluator terminal cleanup is not proved')
        previous.append(dict(journal=str(path), journal_sha256=journal_sha, unit=unit,
                             actual_state=observed, terminal=pin,
                             empty_cgroups=empty_owned_cgroup(unit, observed)))
    return previous


def stable_update(path, initial, value):
    """Replace only our unchanged once record and durably publish its request outcome."""
    original = path.lstat()
    require(stat.S_ISREG(original.st_mode) and original.st_nlink == 1
            and read_json(path) == initial, 'Own launch journal changed before finalization')
    expected_sha = sha(path)
    temporary = path.with_name(path.name + '.update.' + uuid.uuid4().hex)
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        current = path.lstat()
        require(stat.S_ISREG(current.st_mode) and current.st_nlink == 1
                and (current.st_dev, current.st_ino) == (original.st_dev, original.st_ino)
                and sha(path) == expected_sha, 'Own launch journal changed during finalization')
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def output_text(value):
    return value.decode(errors='replace') if isinstance(value, bytes) else value


def dispatch(args, run=subprocess.run):
    require(socket.gethostname().split('.')[0] == 'h100-10', 'Actual host must be h100-10')
    tools = Path(args.tools)
    require(tools.is_absolute() and tools.is_dir() and not tools.is_symlink(),
            'Already deployed independent wrapper directory required')
    require(Path(__file__).resolve() == (tools / 'dispatch_once.py').resolve(),
            'Use the manifest-bound deployed dispatcher')
    unit = owned_unit(args.unit)
    handoff = verify_handoff(args.handoff, args.handoff_sha)
    manifest = verify_code_manifest(tools, args.code_manifest, args.code_manifest_sha)
    directory = RUN / 'output' / 'final-eval-fast'
    journal = Path(args.journal)
    require(journal.is_absolute() and journal.parent == directory
            and re.fullmatch(r'[A-Za-z0-9_-]*dispatch[A-Za-z0-9_-]*once[A-Za-z0-9_-]*\.json', journal.name),
            'Once journal must be directly in the retained run final-eval-fast directory')
    ordinary_directory(directory)
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(('SLURM_', 'SBATCH_', 'SQUEUE_', 'SCANCEL_', 'SACCT_', 'SCONTROL_'))}
    lock = os.open(directory / 'dispatch.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        require(stat.S_ISREG(os.fstat(lock).st_mode) and os.fstat(lock).st_nlink == 1,
                'Dispatch lock must be ordinary')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not journal.exists() and not journal.is_symlink(),
                'Existing once journal forbids another request')
        previous = prior_admission(directory, handoff, environment, run)
        observed = unit_state(unit, environment, run)
        require(observed['LoadState'] == 'not-found' and observed['MainPID'] == '0'
                and observed['ActiveState'] == 'inactive', 'Known service identity already exists')
        require(not (ROOT / 'runs' / unit.removesuffix('.service')).exists(),
                'Owned evaluator run directory already exists; choose a new explicit service identity')
        argv = ['systemd-run', '--unit=' + unit, '--property=MemoryMax=384G',
                '--property=CPUQuota=3200%', '--property=TasksMax=4096',
                '--property=KillMode=control-group', '--property=TimeoutStopSec=60',
                '--property=Nice=5', '--property=Restart=no',
                '--setenv=PYTHONPATH=' + ':'.join(map(str, (tools, FROZEN_SOURCE, ROOT / 'tools'))),
                '--setenv=PYTHONOPTIMIZE=0', '--setenv=PYTHONUNBUFFERED=1',
                '--setenv=CUDA_VISIBLE_DEVICES=' + ','.join(GPU_UUIDS),
                str(ROOT / 'env' / 'bin' / 'python'), str(tools / 'supervisor.py'),
                '--handoff', args.handoff, '--handoff-sha', args.handoff_sha,
                '--code-manifest', args.code_manifest, '--code-manifest-sha', args.code_manifest_sha,
                '--unit', unit]
        initial = dict(status='LAUNCH_UNKNOWN', launch_id=uuid.uuid4().hex, start_epoch=time.time(),
                       actual_host='h100-10', unit=unit, run=str(RUN), train_unit=TRAIN_UNIT,
                       evaluation_only=True, physical_gpus=8, slurm_reserved_gpus=0,
                       tools=str(tools), code_manifest_path=args.code_manifest,
                       code_manifest_sha256=args.code_manifest_sha,
                       eval_wrapper_source_commit=manifest['eval_wrapper_source_commit'],
                       code_files=manifest['files'], handoff_path=args.handoff,
                       handoff_sha256=args.handoff_sha, prior_evaluators=previous,
                       gpu_uuids=list(GPU_UUIDS), submit_argv=argv, journal=str(journal),
                       service_state_before_request=observed, foreign_jobs_modified=False)
        atomic_immutable(journal, initial)
        result = None
        try:
            result = run(argv, env=environment, text=True, capture_output=True, timeout=30)
            require(result.returncode == 0,
                    'Service request is unconfirmed; inspect its exact identity without retrying')
            final = dict(initial, status='REQUESTED', end_epoch=time.time(),
                         systemd_start_returncode=result.returncode,
                         systemd_start_stdout=result.stdout, systemd_start_stderr=result.stderr)
        except BaseException as error:
            final = dict(initial, status='UNKNOWN', end_epoch=time.time(),
                         error_type=type(error).__name__, error=str(error),
                         systemd_start_returncode=None if result is None else result.returncode,
                         systemd_start_stdout=output_text(getattr(error, 'stdout', None)) if result is None else result.stdout,
                         systemd_start_stderr=output_text(getattr(error, 'stderr', None)) if result is None else result.stderr)
            stable_update(journal, initial, final)
            raise
        stable_update(journal, initial, final)
        return final
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        os.close(lock)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('tools', 'code-manifest', 'code-manifest-sha', 'handoff', 'handoff-sha', 'unit', 'journal'):
        parser.add_argument('--' + name, required=True)
    print(json.dumps(dispatch(parser.parse_args()), sort_keys=True))


if __name__ == '__main__':
    main()
