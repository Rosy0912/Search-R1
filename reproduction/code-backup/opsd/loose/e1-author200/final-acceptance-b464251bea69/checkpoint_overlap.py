"""Durable CPU-only checkpoint inspection launcher, outside the training repo.

This file does not restore training state, sign, release, or schedule GPU work.
The pinned pipeline validates observed files and owns its separate CPU results.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys

SOURCE = '188f4e5e7b63cf479290d0888a4a075499886750'
ROOT = Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct200-author-20261006')
RUN = ROOT / 'runs/formal-3261'
REPO = ROOT / 'repo-188f4e5e7b63'
PYTHON = Path('/mnt/data/shaoshitong/search-r1-rl/opsd/env/bin/python')
HERE = Path(__file__).resolve().parent
STEPS = (25, 50, 75, 100, 125, 150, 175, 200)
SCOPE = 'AUTHOR200_CPU_STORED_CONTENT_INSPECTION_NOT_TRAINING_CONTROL'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def unique(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    def number(value):
        import math
        result = float(value)
        need(math.isfinite(result), 'Nonfinite JSON number')
        return result
    return json.loads(raw, object_pairs_hook=pairs, parse_float=number,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode() + b'\n'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda: handle.read(8 * 1024 ** 2), b''):
            digest.update(part)
    return digest.hexdigest()


def physical(path, directory=False):
    path = Path(path)
    need(path.is_absolute() and path.resolve(strict=True) == path
        and path.stat().st_uid == 0, 'Owned physical path required: ' + str(path))
    need(path.is_dir() if directory else path.is_file(), 'Unexpected path kind')
    return path


def pin(value):
    need(type(value) is str and re.fullmatch('[0-9a-f]{64}', value), 'Actual SHA required')
    return value


def pinned_bytes(path, expected):
    path = physical(path)
    before = path.stat()
    fields = ('st_dev', 'st_ino', 'st_uid', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    with path.open('rb') as handle:
        need(tuple(getattr(os.fstat(handle.fileno()), key) for key in fields)
            == tuple(getattr(before, key) for key in fields), 'Immutable input replaced before open')
        raw = handle.read()
        opened_after = os.fstat(handle.fileno())
    after = physical(path).stat()
    need(tuple(getattr(before, key) for key in fields)
        == tuple(getattr(opened_after, key) for key in fields)
        == tuple(getattr(after, key) for key in fields), 'Immutable input changed while reading')
    need(hashlib.sha256(raw).hexdigest() == pin(expected), 'Immutable input SHA differs')
    return raw


def write(path, value):
    with Path(path).open('xb') as handle:
        handle.write(canonical(value))
        handle.flush()
        os.fsync(handle.fileno())
    parent = os.open(str(Path(path).parent), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def host():
    need(os.getuid() == 0 and socket.gethostname().split('.')[0] == 'h100-10',
        'Only own h100-10 CPU execution is allowed')


def process(pid, require_live=True, training_job=False):
    need(type(pid) is int and pid > 0, 'Actual PID required')
    path = Path('/proc') / str(pid)
    try:
        status = (path / 'status').read_text()
        cgroup = (path / 'cgroup').read_text()
        start = (path / 'stat').read_text().rsplit(')', 1)[1].split()[19]
    except FileNotFoundError:
        if require_live:
            raise
        return {'pid': pid, 'present': False}
    uids = list(map(int, re.search(r'^Uid:\s+(.*)', status, re.M).group(1).split()))
    state = re.search(r'^State:\s+(\S+)', status, re.M).group(1)
    need(uids == [0] * 4, 'Process owner changed')
    if require_live:
        need(state in ('R', 'S', 'D', 'I'), 'Process is not alive')
    if training_job:
        need(re.search(r'(?:^|/)job_3261(?:/|$)', cgroup, re.M), 'Foreign job process')
    return {'pid': pid, 'present': True, 'start_ticks': start, 'state': state,
        'uids': uids, 'cgroup_original': cgroup, 'status_original': status}


def tools(expected):
    host()
    physical(HERE, True)
    path = physical(HERE / 'manifest.json')
    value = unique(pinned_bytes(path, expected))
    need(type(value.get('version')) is int and value.get('version') == 1 and value.get('source_commit') == SOURCE
        and value.get('kind') == 'AUTHOR200_EXTERNAL_CHECKPOINT_TOOLS'
        and value.get('scope') == SCOPE, 'External tool identity differs')
    required = {'checkpoint_overlap.py', 'checkpoint_cpu_pipeline.py',
        'observe_author200_checkpoint.py', 'inspect_author200_checkpoint_cpu.py',
        'hf_author200_cpu.py'}
    need(required <= set(value['tools']), 'Required reviewed tool missing')
    for name, expected_sha in value['tools'].items():
        need(type(name) is str and Path(name).name == name and name.endswith('.py'),
            'Unsafe tool path')
        pinned_bytes(HERE / name, expected_sha)
    for filename, expected_sha in value['dependencies'].items():
        pinned_bytes(filename, expected_sha)
    need(canonical(value['contract']) == canonical({'training_job_id': '3261', 'horizon': 200,
        'checkpoint_steps': list(STEPS), 'global_mini_batch': 32,
        'adam_steps_per_rank_per_outer': 16, 'warmup_steps': 0,
        'learning_rate': 5e-6, 'cuda_work': False}), 'External inspection contract differs')
    return value


def allocation():
    host()
    physical(RUN, True)
    physical(REPO, True)
    need(subprocess.check_output(['git', '--no-optional-locks', '-C', str(REPO),
        'rev-parse', 'HEAD'], text=True).strip() == SOURCE, 'Frozen source differs')
    need(not subprocess.check_output(['git', '--no-optional-locks', '-C', str(REPO),
        'status', '--porcelain'], text=True).strip(), 'Frozen source is dirty')
    raw = subprocess.check_output(['scontrol', 'show', 'job', '3261', '-o'],
        text=True, timeout=20)
    fields = dict(item.split('=', 1) for item in raw.split() if '=' in item)
    need((fields.get('JobId'), fields.get('JobState'), fields.get('Restarts'),
        fields.get('NodeList'), fields.get('TresPerNode'), fields.get('OverSubscribe'))
        == ('3261', 'RUNNING', '0', 'h100-10', 'gres:gpu:h100:8', 'NO'),
        'The original live allocation is required for overlap mode')
    driver = process(1533312, training_job=True)
    need(driver['start_ticks'] == '1829085316', 'Original training process changed')
    return {'scontrol_original': raw, 'driver': driver}


def cpu_command(request, request_pin):
    return ['srun', '--jobid=3261', '--overlap', '--nodes=1', '--ntasks=1',
        '--cpus-per-task=1', '--gres=none', '--nodelist=h100-10', 'env',
        'CUDA_VISIBLE_DEVICES=', 'OMP_NUM_THREADS=1', 'MKL_NUM_THREADS=1',
        'OPENBLAS_NUM_THREADS=1', 'PYTHONDONTWRITEBYTECODE=1',
        'PYTHONPATH=' + str(REPO), str(PYTHON), str(HERE / 'checkpoint_cpu_pipeline.py'),
        '--request', str(request), '--request-sha256', request_pin]


def validate_request(config, path, output_exists=True):
    need(set(config) == {'version', 'scope', 'source_commit', 'training_job_id',
        'checkpoint_step', 'tools_manifest_sha256', 'output_directory', 'log_prefixes'},
        'Unexpected request schema')
    need(type(config['version']) is int and config['version'] == 1
        and config['scope'] == SCOPE and config['source_commit'] == SOURCE
        and config['training_job_id'] == '3261' and type(config['checkpoint_step']) is int
        and config['checkpoint_step'] in STEPS, 'Wrong checkpoint request identity')
    need(path == RUN / 'evidence' / ('checkpoint-cpu-' + str(config['checkpoint_step'])
        + '-launch.json'), 'Foreign launch marker')
    tools(config['tools_manifest_sha256'])
    need(type(config['output_directory']) is str, 'Output path must be text')
    output = Path(config['output_directory'])
    need(output.is_absolute() and output.parent == RUN / 'evidence', 'Foreign inspection output')
    need(re.fullmatch('checkpoint-' + str(config['checkpoint_step'])
        + r'-cpu-inspection-[A-Za-z0-9_-]+', output.name),
        'Output must use a checkpoint inspection directory name, never a reserved marker')
    physical(output.parent, True)
    if output_exists:
        physical(output, True)
    else:
        need(not output.exists() and not output.is_symlink(), 'Fresh output required')
    need(set(config['log_prefixes']) == {'training', 'driver'}, 'Two original log prefixes required')
    for name, entry in config['log_prefixes'].items():
        need(set(entry) == {'path', 'bytes', 'sha256'} and type(entry['bytes']) is int
            and entry['bytes'] > 0, 'Exact log prefix required')
        pin(entry['sha256'])
        log = physical(entry['path'])
        if name == 'training':
            need(log == RUN / 'train.log', 'Foreign aggregate log')
        else:
            need(log.is_relative_to(Path('/tmp/o4e-3261/ray'))
                and log.name.endswith('-1533312.out'), 'Foreign driver original')
        before = log.stat()
        with log.open('rb') as handle:
            raw = handle.read(entry['bytes'])
        need(len(raw) == entry['bytes'] and raw.endswith(b'\n')
            and hashlib.sha256(raw).hexdigest() == entry['sha256'], 'Fixed complete prefix changed')
        with physical(log).open('rb') as handle:
            after_raw = handle.read(entry['bytes'])
        after = log.stat()
        need((before.st_dev, before.st_ino, before.st_uid) ==
            (after.st_dev, after.st_ino, after.st_uid) and after_raw == raw,
            'Pinned log identity/prefix changed; append beyond the prefix is allowed')
    return config, output


def load_request(path, expected):
    path = physical(path)
    return validate_request(unique(pinned_bytes(path, expected)), path)


def child(role, request, expected):
    return [sys.executable, str(HERE / 'checkpoint_overlap.py'), role,
        '--request', str(request), '--request-sha256', expected]


def exit_status(code):
    return code if code >= 0 else 128 - code


def controller(config, output, request, expected):
    before = allocation()
    command = cpu_command(request, expected)
    write(output / 'controller-start-original.json', {'utc': now(), 'scope': SCOPE,
        'process': process(os.getpid()), 'before': before, 'command': command})
    with (output / 'srun-stdout-original.txt').open('xb') as stdout, \
            (output / 'srun-stderr-original.txt').open('xb') as stderr:
        proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr)
        try:
            write(output / 'srun-start-original.json', {'utc': now(), 'pid': proc.pid,
                'process': process(proc.pid, require_live=False), 'command': command})
        finally:
            result = proc.wait()
    write(output / 'srun-exit-original.json', {'utc': now(), 'returncode': result,
        'pid': proc.pid, 'waited_by_controller_pid': os.getpid(), 'scope': SCOPE})
    return exit_status(result)


def supervise(config, output, request, expected):
    before = allocation()
    write(output / 'supervisor-start-original.json', {'utc': now(), 'scope': SCOPE,
        'process': process(os.getpid()), 'before': before})
    command = child('controller', request, expected)
    with (output / 'controller-stdout-original.txt').open('xb') as stdout, \
            (output / 'controller-stderr-original.txt').open('xb') as stderr:
        proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr)
        try:
            write(output / 'controller-launch-original.json', {'utc': now(),
                'pid': proc.pid, 'process': process(proc.pid, require_live=False), 'command': command})
        finally:
            result = proc.wait()
    write(output / 'controller-exit-original.json', {'utc': now(), 'returncode': result,
        'pid': proc.pid, 'waited_by_supervisor_pid': os.getpid(), 'scope': SCOPE})
    return exit_status(result)


def start(args):
    tools(args.tools_manifest_sha256)
    before = allocation()
    need(type(args.checkpoint_step) is int and args.checkpoint_step in STEPS,
        'Only later actual checkpoint steps are supported')
    evidence = physical(RUN / 'evidence', True)
    output = Path(args.output_directory)
    need(output.is_absolute() and output.parent == evidence
        and not output.exists() and not output.is_symlink(), 'Fresh physical output required')
    marker = evidence / ('checkpoint-cpu-' + str(args.checkpoint_step) + '-launch.json')
    need(not marker.exists() and not marker.is_symlink(),
        'Existing launch; inspect original results, never delete to retry')
    config = {'version': 1, 'scope': SCOPE, 'source_commit': SOURCE,
        'training_job_id': '3261', 'checkpoint_step': args.checkpoint_step,
        'tools_manifest_sha256': pin(args.tools_manifest_sha256),
        'output_directory': str(output), 'log_prefixes': unique(args.log_prefixes_json)}
    validate_request(config, marker, output_exists=False)
    need(os.readlink('/proc/1533312/fd/1') == config['log_prefixes']['driver']['path'],
        'The pinned driver prefix must be the original process stdout target')
    # No checkpoint or training files are modified. The launch marker remains
    # even if subsequent transport/process creation has an unknown outcome.
    output.mkdir(mode=0o700)
    write(output / 'root-start-context-original.json', before)
    write(marker, config)
    marker_sha = sha(marker)
    load_request(marker, marker_sha)
    command = child('supervise', marker, marker_sha)
    with (output / 'supervisor-stdout-original.txt').open('xb') as stdout, \
            (output / 'supervisor-stderr-original.txt').open('xb') as stderr:
        proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout,
            stderr=stderr, start_new_session=True)
        write(output / 'durable-launch-original.json', {'utc': now(), 'pid': proc.pid,
            'process': process(proc.pid, require_live=False), 'command': command,
            'launcher': process(os.getpid())})
    print(json.dumps({'started': True, 'scope': SCOPE, 'checkpoint_step': args.checkpoint_step,
        'request': str(marker), 'request_sha256': marker_sha, 'supervisor_pid': proc.pid,
        'output_directory': str(output)}, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('start', 'supervise', 'controller'))
    parser.add_argument('--request', type=Path)
    parser.add_argument('--request-sha256')
    parser.add_argument('--checkpoint-step', type=int)
    parser.add_argument('--tools-manifest-sha256')
    parser.add_argument('--output-directory')
    parser.add_argument('--log-prefixes-json')
    args = parser.parse_args()
    if args.mode == 'start':
        start(args)
    else:
        config, output = load_request(args.request, args.request_sha256)
        method = supervise if args.mode == 'supervise' else controller
        sys.exit(method(config, output, args.request, args.request_sha256))


if __name__ == '__main__':
    main()
