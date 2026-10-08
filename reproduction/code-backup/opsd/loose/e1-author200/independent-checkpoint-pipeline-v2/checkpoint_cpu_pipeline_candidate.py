"""Runs pinned observation/native/HF inspectors in a real CPU-only Slurm step.

Only stored content is inspected. No live state is restored, no acceptance is
signed and no training process is started or changed.
"""
import argparse
import datetime
import hashlib
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import types

HERE = Path(__file__).resolve().parent
ROOT = Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct200-author-20261006')
RUN = ROOT / 'runs/formal-3261'
REPO = ROOT / 'repo-188f4e5e7b63'
SOURCE = '188f4e5e7b63cf479290d0888a4a075499886750'
BASE = Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004/assets/qwen2.5-3b-instruct-aa8e725')
MODEL_SHA = '7f53ce6c010ac076b1ec482031bea7c9e6b8d0a096a61b880fe8ad4971bc9e44'
FIRST5_SHA = '8c6e17b2af0026b1267a2422c1437aa00461da46e0a1911caea3633bfef473d3'
RECEIPT_SHA = '812f0baf1d70ee87e847865f62ff03151e1de0c4a4c0eab2e9ae72e0dbb3793b'
PUBLIC_SHA = 'ff53cfde6dd4bf96142b23364ae72da1d9610e1a834663ba70e750f604cee41c'
CLOSURE_SHA = '3842c259b30c524e5ba5e067eb5b37f6dbac1042e162a5d93dd23511c13b5c9d'
NATIVE_HELPER = ROOT / 'tools/first5-native-1f209cab7e02/inspect_payloads.py'
HF_HELPER = ROOT / 'tools/first5-12424d01025e/hf_cpu_reload.py'
SCOPE = 'AUTHOR200_CPU_STORED_CONTENT_INSPECTION_NOT_TRAINING_CONTROL'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def pinned_bytes(path, expected):
    path = Path(path)
    need(type(expected) is str and re.fullmatch('[0-9a-f]{64}', expected), 'Actual SHA pin required')
    need(path.is_absolute() and path.resolve(strict=True) == path and path.is_file()
        and path.stat().st_uid == os.getuid() == 0, 'Owned physical input required')
    fields = ('st_dev', 'st_ino', 'st_uid', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    before = path.stat()
    with path.open('rb') as handle:
        need(tuple(getattr(os.fstat(handle.fileno()), key) for key in fields)
            == tuple(getattr(before, key) for key in fields), 'Input replaced before open')
        raw = handle.read()
        opened_after = os.fstat(handle.fileno())
    need(path.resolve(strict=True) == path, 'Input became a path alias')
    after = path.stat()
    need(tuple(getattr(before, key) for key in fields)
        == tuple(getattr(opened_after, key) for key in fields)
        == tuple(getattr(after, key) for key in fields) and hashlib.sha256(raw).hexdigest() == expected,
        'Immutable input changed or SHA differs')
    return raw


def load_launcher(request_path, request_pin):
    # The request pins the whole tool manifest, which pins the launcher before
    # any of its executable bytes are loaded. No stale pyc or source rewrite.
    request_path = Path(request_path)
    request_raw = pinned_bytes(request_path, request_pin)
    import json
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'Duplicate request key')
            result[key] = value
        return result
    request = json.loads(request_raw, object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite request')))
    expected = request['tools_manifest_sha256']
    manifest_path = HERE / 'manifest.json'
    manifest = json.loads(pinned_bytes(manifest_path, expected), object_pairs_hook=pairs)
    launcher = HERE / 'checkpoint_overlap.py'
    module = types.ModuleType('pinned_author200_cpu_launcher')
    module.__file__ = str(launcher)
    raw = pinned_bytes(launcher, manifest['tools']['checkpoint_overlap.py'])
    exec(compile(raw, str(launcher), 'exec'), module.__dict__)
    config, output = module.load_request(request_path, request_pin)
    return module, config, output


def execution_context(launcher):
    launcher.host()
    job = os.environ.get('SLURM_JOB_ID')
    step = os.environ.get('SLURM_STEP_ID')
    need(job == '3261' and type(step) is str and re.fullmatch('[0-9]+', step),
        'Actual overlap CPU job/step identity required')
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == ''
        and all(os.environ.get(key) == '1' for key in
        ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'))
        and os.environ.get('PYTHONDONTWRITEBYTECODE') == '1'
        and os.environ.get('PYTHONPATH') == str(REPO), 'CPU-only runtime differs')
    p = launcher.process(os.getpid(), training_job=True)
    need(re.search(r'(?:^|/)step_' + re.escape(step) + r'(?:/|$)', p['cgroup_original'], re.M),
        'Observed CPU execution step/cgroup differ')
    memory = Path('/proc/meminfo').read_text()
    available = int(re.search(r'^MemAvailable:\s+([0-9]+) kB', memory, re.M).group(1))
    need(available >= 96 * 1024 ** 2, 'Available node CPU RAM below96GiB')
    return {'job_id': job, 'step_id': step, 'process': p, 'hostname': socket.gethostname(),
        'python_executable': sys.executable, 'python_version': sys.version,
        'meminfo_original': memory, 'available_kib': available,
        'cuda_visible_devices': '', 'scope': SCOPE}


def observer_command(config, output):
    command = [sys.executable, str(HERE / 'observe_author200_checkpoint.py'),
        '--run-root', str(RUN), '--job-id', '3261',
        '--checkpoint-step', str(config['checkpoint_step']),
        '--accepted5-manifest', str(RUN / 'full_state/global_step_5/manifest.json'),
        '--accepted5-manifest-sha256', FIRST5_SHA,
        '--accepted5-receipt', str(RUN / 'acceptance-5-receipt.json'),
        '--accepted5-receipt-sha256', RECEIPT_SHA,
        '--public-key', str(RUN / 'bindings/public.json'), '--public-key-sha256', PUBLIC_SHA,
        '--source-root', str(REPO), '--source-commit', SOURCE,
        '--native-import-closure', str(RUN / 'bindings/import_closure.json'),
        '--native-import-closure-sha256', CLOSURE_SHA,
        '--output-directory', str(output / 'observation')]
    for role, prefix in config['log_prefixes'].items():
        stem = '--training-log' if role == 'training' else '--driver-log'
        command.extend([stem, prefix['path'], stem + '-sha256', prefix['sha256'],
            stem + '-prefix-bytes', str(prefix['bytes'])])
    return command


def native_command(config, output, inventory, inventory_sha, context):
    return [sys.executable, str(HERE / 'inspect_author200_checkpoint_cpu.py'),
        '--inventory', str(inventory), '--inventory-sha256', inventory_sha,
        '--step', str(config['checkpoint_step']), '--repo', str(REPO),
        '--helper', str(NATIVE_HELPER), '--base-model', str(BASE),
        '--base-manifest', str(BASE / 'VERIFIED-MODEL-MANIFEST.json'),
        '--base-manifest-sha256', MODEL_SHA, '--execution-job-id', context['job_id'],
        '--execution-step-id', context['step_id'], '--output', str(output / 'native.json')]


def hf_command(output, inventory, inventory_sha, native_sha, context):
    return [sys.executable, str(HERE / 'hf_author200_cpu.py'),
        '--inventory', str(inventory), '--inventory-sha256', inventory_sha,
        '--native-report', str(output / 'native.json'), '--native-report-sha256', native_sha,
        '--native-inspector', str(HERE / 'inspect_author200_checkpoint_cpu.py'),
        '--native-inspector-sha256', sha(HERE / 'inspect_author200_checkpoint_cpu.py'),
        '--helper', str(HF_HELPER), '--repo', str(REPO),
        '--execution-job-id', context['job_id'], '--execution-step-id', context['step_id'],
        '--output', str(output / 'hf.json')]


def run_one(launcher, output, name, command):
    launcher.write(output / (name + '-command-original.json'),
        {'utc': launcher.now(), 'argv': command, 'scope': SCOPE})
    with (output / (name + '-stdout-original.txt')).open('xb') as stdout, \
            (output / (name + '-stderr-original.txt')).open('xb') as stderr:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout,
            stderr=stderr, env=dict(os.environ))
        try:
            launcher.write(output / (name + '-process-original.json'),
                {'utc': launcher.now(), 'pid': child.pid,
                'process': launcher.process(child.pid, require_live=False)})
        finally:
            code = child.wait()
    launcher.write(output / (name + '-exit-original.json'), {'utc': launcher.now(),
        'returncode': code, 'pid': child.pid, 'waited_by_pipeline_pid': os.getpid(),
        'scope': SCOPE})
    need(code == 0, 'Actual ' + name + ' inspection failed; original exits/streams preserved')


def pipeline(request_path, request_sha):
    launcher, config, output = load_launcher(request_path, request_sha)
    context = execution_context(launcher)
    launcher.write(output / 'cpu-step-original.json', {'utc': launcher.now(),
        'request_sha256': request_sha, 'training_job_id': '3261',
        'checkpoint_step': config['checkpoint_step'], 'execution_context': context})
    run_one(launcher, output, 'observer', observer_command(config, output))
    inventory = launcher.physical(output / 'observation/observed-checkpoint-inventory.json')
    inventory_sha = sha(inventory)
    observed = launcher.unique(inventory.read_bytes())
    need(observed['kind'] == 'OBSERVED_AUTHOR200_CHECKPOINT_INVENTORY_NOT_TRAINING_MANIFEST'
        and observed['source_commit'] == SOURCE and observed['job_id'] == '3261'
        and type(observed['checkpoint_step']) is int
        and observed['checkpoint_step'] == config['checkpoint_step'],
        'Observed inventory identity differs')
    run_one(launcher, output, 'native',
        native_command(config, output, inventory, inventory_sha, context))
    native = launcher.unique(launcher.physical(output / 'native.json').read_bytes())
    need(native['scope'] == 'NATIVE_CPU_STORED_CHECKPOINT_CONTENT_INSPECTION'
        and native['passed'] is True and native['source_commit'] == SOURCE
        and native['job_id'] == '3261' and type(native['checkpoint_step']) is int
        and native['checkpoint_step'] == config['checkpoint_step']
        and native['cuda_is_initialized'] is False, 'Actual native report scope differs')
    native_sha = sha(output / 'native.json')
    whole_hf = None
    if config['checkpoint_step'] == 200:
        run_one(launcher, output, 'hf',
            hf_command(output, inventory, inventory_sha, native_sha, context))
        hf = launcher.unique(launcher.physical(output / 'hf.json').read_bytes())
        need(hf['scope'] == 'HF200_WHOLE_CPU_MODEL_LOAD_NOT_TRAINING_RESTORE'
            and hf['passed'] is True and hf['source_commit'] == SOURCE
            and hf['job_id'] == '3261' and type(hf['checkpoint_step']) is int
            and hf['checkpoint_step'] == 200 and hf['cuda_is_initialized'] is False,
            'Actual HF200 report scope differs')
        whole_hf = sha(output / 'hf.json')
    after_context = execution_context(launcher)
    launcher.load_request(request_path, request_sha)
    need(sha(inventory) == inventory_sha and sha(output / 'native.json') == native_sha,
        'Actual reports changed during pipeline')
    if whole_hf is not None:
        need(sha(output / 'hf.json') == whole_hf, 'HF200 report changed')
    launcher.write(output / 'inspection-completed-review-required.json', {
        'version': 1, 'scope': SCOPE, 'source_commit': SOURCE, 'training_job_id': '3261',
        'checkpoint_step': config['checkpoint_step'], 'request_sha256': request_sha,
        'inventory_sha256': inventory_sha, 'native_report_sha256': native_sha,
        'whole_hf200_report_sha256': whole_hf, 'execution_context': after_context,
        'files': {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob('*')) if p.is_file()},
        'independent_acceptance_pending': True, 'limitations': [
            'CPU stored content inspection; no live restoration or new-process recovery',
            'No training control, signature, scientific effect or terminal/resource acceptance']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True, type=Path)
    parser.add_argument('--request-sha256', required=True)
    args = parser.parse_args()
    pipeline(args.request, args.request_sha256)
