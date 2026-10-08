"""One isolated CPU recheck of the actual failed25 checker, without training control.

The old failure/launch and observed inventory remain immutable. This runner
requires their actual pins, and records a different real CPU execution step.
"""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import types

HERE = Path(__file__).resolve().parent
ROOT = Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct200-author-20261006')
RUN = ROOT / 'runs/formal-3261'
REPO = ROOT / 'repo-188f4e5e7b63'
OLD_TOOLS = ROOT / 'tools/final-acceptance-8441e10fa456'
OLD_MANIFEST = '8441e10fa4565bf53f7baf24a57a60ddbe9844dc979a2c9a0c97250219df7019'
LAUNCHER = '20c9fb1b720bd345f92b3cdfca0b4b19dc6d260e84c2d16429af497f8de45062'
PIPELINE = 'ef6dedc6dfde43c836dd96236d71b280d0c4adf9e53233eb86434d5f8463097a'
ORIGINAL_MARKER = RUN / 'evidence/checkpoint-cpu-25-launch.json'
ORIGINAL_REQUEST = 'feac6e0c56a35e7725307b0d65c6f0437a0ae18bba13bea07e774721b1a2e88f'
ORIGINAL_OUTPUT = RUN / 'evidence/checkpoint-25-cpu-inspection-once-v1'
INVENTORY = ORIGINAL_OUTPUT / 'observation/observed-checkpoint-inventory.json'
INVENTORY_SHA = '32f2c38c121b0c59ab26036c358b4fd749218f8e3980d6700f54c78fabbe6c4e'
FAILURES = {
    'native-exit-original.json': 'acf00633f70bcf95c11e5635df9724235531121177740d81e73768e692a86e46',
    'srun-exit-original.json': 'faec00c6c152b566416d818c2de3fe5b629e27ef4920875ef8f688026c799a37',
    'controller-exit-original.json': '1fbf8672f6873a68c693c6dfcf0ea1263030f5d6ccfdca16bf12e7766c91694a',
}
FAILURE_STDERR = '43277dcc82f583862054290d83e2c608e497ce4815c5d4c5586600652ccc20f8'
MARKER = RUN / 'evidence/checkpoint-cpu-25-repair-v2-launch.json'
OUTPUT = RUN / 'evidence/checkpoint-25-cpu-inspection-repair-v2'
SCOPE = 'CPU25_FAILED_CHECKER_REPAIR_STORED_CONTENT_ONLY'
PYTHON = Path('/mnt/data/shaoshitong/search-r1-rl/opsd/env/bin/python')


def module(path, expected, reader):
    raw = reader(path, expected)
    value = types.ModuleType('pinned_' + path.stem)
    value.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), value.__dict__)
    return value


def bootstrap(manifest_sha):
    # Execute only the exact previously reviewed launcher, after hashing the
    # opened bytes. Its stable reader/owner/source/manifest checks are reused.
    path = HERE / 'checkpoint_overlap.py'
    before = path.stat()
    with path.open('rb') as handle:
        opened = os.fstat(handle.fileno())
        raw = handle.read()
    after = path.stat()
    fields = ('st_dev', 'st_ino', 'st_uid', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    if not (path.is_absolute() and path.resolve(strict=True) == path and
            os.getuid() == 0 and before.st_uid == 0 and
            tuple(getattr(before, k) for k in fields) ==
            tuple(getattr(opened, k) for k in fields) ==
            tuple(getattr(after, k) for k in fields) and
            hashlib.sha256(raw).hexdigest() == LAUNCHER):
        raise ValueError('Reviewed launcher bytes/physical owner changed')
    value = types.ModuleType('current_pinned_launcher')
    value.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), value.__dict__)
    tools = value.tools(manifest_sha)
    value.pinned_bytes(HERE / 'checkpoint_repair_cpu.py',
        tools['tools']['checkpoint_repair_cpu.py'])
    old = module(OLD_TOOLS / 'checkpoint_overlap.py', LAUNCHER, value.pinned_bytes)
    original, old_output = old.load_request(ORIGINAL_MARKER, ORIGINAL_REQUEST)
    value.need(original['tools_manifest_sha256'] == OLD_MANIFEST and
        original['checkpoint_step'] == 25 and old_output == ORIGINAL_OUTPUT,
        'Actual failed launch identity differs')
    value.pinned_bytes(INVENTORY, INVENTORY_SHA)
    for name, pin in FAILURES.items():
        result = value.unique(value.pinned_bytes(ORIGINAL_OUTPUT / name, pin))
        value.need(type(result['returncode']) is int and result['returncode'] == 1,
            'The actual previous inspection must have failed')
    raw_stderr = value.pinned_bytes(ORIGINAL_OUTPUT / 'native-stderr-original.txt', FAILURE_STDERR)
    value.need(raw_stderr.endswith(b'ValueError: Sampler position/config differs: seed\n') and
        not (ORIGINAL_OUTPUT / 'native.json').exists() and
        not (ORIGINAL_OUTPUT / 'inspection-completed-review-required.json').exists(),
        'Different prior failure or unexpected completed report')
    return value, tools


def request_value(manifest_sha):
    return {'version': 1, 'scope': SCOPE, 'training_job_id': '3261',
        'source_commit': '188f4e5e7b63cf479290d0888a4a075499886750',
        'checkpoint_step': 25, 'tools_manifest_sha256': manifest_sha,
        'original_request_sha256': ORIGINAL_REQUEST,
        'original_failed_exit_sha256': FAILURES,
        'original_native_stderr_sha256': FAILURE_STDERR,
        'inventory_path': str(INVENTORY), 'inventory_sha256': INVENTORY_SHA,
        'output_directory': str(OUTPUT)}


def read_request(launcher, manifest_sha, request_sha):
    actual = launcher.unique(launcher.pinned_bytes(MARKER, request_sha))
    launcher.need(launcher.canonical(actual) == launcher.canonical(request_value(manifest_sha)),
        'Repair request differs from fixed actual failure lineage')
    launcher.physical(OUTPUT, True)
    return actual


def self_command(mode, manifest_sha, request_sha):
    return [sys.executable, str(HERE / 'checkpoint_repair_cpu.py'), mode,
        '--tools-manifest-sha256', manifest_sha, '--request-sha256', request_sha]


def start(launcher, manifest_sha):
    before = launcher.allocation()
    launcher.physical(OUTPUT.parent, True)
    launcher.need(not OUTPUT.exists() and not OUTPUT.is_symlink() and
        not MARKER.exists() and not MARKER.is_symlink(),
        'Existing repair marker/output: inspect it, never delete or retry')
    OUTPUT.mkdir(mode=0o700)
    launcher.write(OUTPUT / 'root-start-context-original.json', before)
    launcher.write(MARKER, request_value(manifest_sha))
    request_sha = launcher.sha(MARKER)
    read_request(launcher, manifest_sha, request_sha)
    command = self_command('supervise', manifest_sha, request_sha)
    with (OUTPUT / 'supervisor-stdout-original.txt').open('xb') as stdout, \
            (OUTPUT / 'supervisor-stderr-original.txt').open('xb') as stderr:
        proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout,
            stderr=stderr, start_new_session=True)
        launcher.write(OUTPUT / 'durable-launch-original.json', {'utc': launcher.now(),
            'pid': proc.pid, 'process': launcher.process(proc.pid, require_live=False),
            'command': command, 'scope': SCOPE})
    print(launcher.canonical({'started': True, 'scope': SCOPE, 'checkpoint_step': 25,
        'request': str(MARKER), 'request_sha256': request_sha,
        'output_directory': str(OUTPUT), 'supervisor_pid': proc.pid}).decode().strip())


def wait_child(launcher, role, command):
    launcher.write(OUTPUT / (role + '-command-original.json'),
        {'utc': launcher.now(), 'argv': command, 'scope': SCOPE})
    with (OUTPUT / (role + '-stdout-original.txt')).open('xb') as stdout, \
            (OUTPUT / (role + '-stderr-original.txt')).open('xb') as stderr:
        proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr)
        try:
            launcher.write(OUTPUT / (role + '-process-original.json'),
                {'utc': launcher.now(), 'pid': proc.pid,
                    'process': launcher.process(proc.pid, require_live=False), 'scope': SCOPE})
        finally:
            result = proc.wait()
            launcher.write(OUTPUT / (role + '-exit-original.json'), {'utc': launcher.now(),
                'returncode': result, 'pid': proc.pid, 'waited_by_pid': os.getpid(), 'scope': SCOPE})
    return result


def inspect(launcher, tools, manifest_sha, request_sha):
    pipeline = module(HERE / 'checkpoint_cpu_pipeline.py', PIPELINE, launcher.pinned_bytes)
    context = pipeline.execution_context(launcher)
    launcher.write(OUTPUT / 'cpu-step-original.json', {'utc': launcher.now(),
        'execution_context': context, 'request_sha256': request_sha,
        'original_inventory_sha256': INVENTORY_SHA, 'original_failure_exits': FAILURES,
        'scope': SCOPE})
    command = [sys.executable, str(HERE / 'inspect_author200_checkpoint_cpu.py'),
        '--inventory', str(INVENTORY), '--inventory-sha256', INVENTORY_SHA,
        '--step', '25', '--repo', str(REPO), '--helper', str(pipeline.NATIVE_HELPER),
        '--base-model', str(pipeline.BASE), '--base-manifest',
        str(pipeline.BASE / 'VERIFIED-MODEL-MANIFEST.json'), '--base-manifest-sha256',
        pipeline.MODEL_SHA, '--execution-job-id', context['job_id'],
        '--execution-step-id', context['step_id'], '--output', str(OUTPUT / 'native.json')]
    code = wait_child(launcher, 'native', command)
    launcher.need(code == 0, 'Actual repaired inspection failed; retain original exits/streams')
    pin = launcher.sha(launcher.physical(OUTPUT / 'native.json'))
    report = launcher.unique(launcher.pinned_bytes(OUTPUT / 'native.json', pin))
    launcher.need(report['scope'] == 'NATIVE_CPU_STORED_CHECKPOINT_CONTENT_INSPECTION' and
        report['passed'] is True and report['source_commit'] == tools['source_commit'] and
        report['job_id'] == '3261' and type(report['checkpoint_step']) is int and
        report['checkpoint_step'] == 25 and report['cuda_is_initialized'] is False and
        report['inspector_source_sha256'] == tools['tools']['inspect_author200_checkpoint_cpu.py'],
        'Repaired native report identity/scope differs')
    process_path = OUTPUT / 'native-process-original.json'
    process_record = launcher.unique(launcher.pinned_bytes(process_path, launcher.sha(process_path)))
    identity = report['execution_identity']
    observed_process = process_record['process']
    launcher.need(identity['execution_job_id'] == context['job_id'] and
        identity['execution_step_id'] == context['step_id'] and
        type(identity['pid']) is int and type(process_record['pid']) is int and
        identity['pid'] == process_record['pid'] and
        identity['node'] == context['hostname'] and type(identity['uid']) is int and
        identity['uid'] == 0 and type(identity['start_ticks']) is int and
        identity['start_ticks'] > 0 and observed_process['present'] is True and
        str(identity['start_ticks']) == observed_process['start_ticks'] and
        type(identity['uids']) is list and len(identity['uids']) == 4 and
        all(type(uid) is int and uid == 0 for uid in identity['uids']) and
        type(observed_process['uids']) is list and len(observed_process['uids']) == 4 and
        all(type(uid) is int and uid == 0 for uid in observed_process['uids']) and
        identity['uids'] == observed_process['uids'] and
        identity['cgroup'] == observed_process['cgroup_original'] ==
        context['process']['cgroup_original'],
        'Native child/report execution identity differs')
    bootstrap(manifest_sha)
    read_request(launcher, manifest_sha, request_sha)
    launcher.write(OUTPUT / 'inspection-completed-review-required.json', {
        'utc': launcher.now(), 'scope': SCOPE, 'request_sha256': request_sha,
        'tools_manifest_sha256': manifest_sha, 'inventory_sha256': INVENTORY_SHA,
        'native_report_sha256': pin, 'execution_context': pipeline.execution_context(launcher),
        'original_failure_exits_preserved': FAILURES,
        'independent_acceptance_pending': True,
        'files': {str(p.relative_to(OUTPUT)): launcher.sha(p)
            for p in sorted(OUTPUT.rglob('*')) if p.is_file()},
        'limitations': ['Stored CPU content only; no live/new-process restore',
            'No scientific effect, signature, training control, final200 or terminal acceptance']})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('start', 'supervise', 'controller', 'inspect'))
    parser.add_argument('--tools-manifest-sha256', required=True)
    parser.add_argument('--request-sha256')
    a = parser.parse_args()
    launcher, tools = bootstrap(a.tools_manifest_sha256)
    if a.mode == 'start':
        start(launcher, a.tools_manifest_sha256)
        return
    read_request(launcher, a.tools_manifest_sha256, a.request_sha256)
    if a.mode == 'inspect':
        inspect(launcher, tools, a.tools_manifest_sha256, a.request_sha256)
        return
    launcher.allocation()
    if a.mode == 'supervise':
        command = self_command('controller', a.tools_manifest_sha256, a.request_sha256)
        role = 'controller'
    else:
        command = ['srun', '--jobid=3261', '--overlap', '--nodes=1', '--ntasks=1',
            '--cpus-per-task=1', '--gres=none', '--nodelist=h100-10', 'env',
            'CUDA_VISIBLE_DEVICES=', 'OMP_NUM_THREADS=1', 'MKL_NUM_THREADS=1',
            'OPENBLAS_NUM_THREADS=1', 'PYTHONDONTWRITEBYTECODE=1', 'PYTHONPATH=' + str(REPO),
            str(PYTHON), str(HERE / 'checkpoint_repair_cpu.py'), 'inspect',
            '--tools-manifest-sha256', a.tools_manifest_sha256,
            '--request-sha256', a.request_sha256]
        role = 'srun'
    code = wait_child(launcher, role, command)
    sys.exit(launcher.exit_status(code))


if __name__ == '__main__':
    main()
