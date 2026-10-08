"""Submit one reviewed CPU-only analysis; never retry an uncertain sbatch result."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import time

ROOT = Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
RUN = ROOT / 'runs/tree-safety300-3268'
PACKAGE = ROOT / 'instruct300/candidates/safety110-a632ba9-84e3e186'
TOOLS = ROOT / 'instruct300/context-analysis-28e665eb-v2'
HELPER_SHA = '28e665eb822651ff44559b2e9c6110d01542c5e6433db8dd4d07c659af7131ac'
REVIEW_SHA = 'ab5c5762937279f0a64ef976f53851daf6af11d4f9504d5bab252f7bf641acbb'
PACKAGE_SHA = '84e3e18682afa880f1053a0025eda595fca332d550ec1f6905bc1334049b5a48'
READY_SHA = 'ba948bdec0b71a3c8ff5c61db1360840134e57fccaf71fb154449f30c300c468'
INDEPENDENT_PROBES = {
    'raw_stats_cpu_v2.py': '56d00286a0640d75cac0c299be3144d7e6b46d0c1f03902de51208f6ba0114af',
    'tree_trajectory_cpu.py': 'b96f6b90614817c4330411104bf6b62fe59c9c4417239d8d411d0ab25dbd6152'}

def bound(path, expected=None):
    with Path(path).open('rb') as stream:
        payload = stream.read()
    actual = hashlib.sha256(payload).hexdigest()
    if expected is not None:
        assert actual == expected, str(path)
    return json.loads(payload), actual

assert socket.gethostname().split('.')[0] == 'h100-9'
assert hashlib.sha256((TOOLS / 'analyze_tree_sampling_calibration.py').read_bytes()).hexdigest() == HELPER_SHA
compile((TOOLS / 'analyze_tree_sampling_calibration.py').read_text(), str(TOOLS / 'analyze_tree_sampling_calibration.py'), 'exec')
for name, expected in INDEPENDENT_PROBES.items():
    assert hashlib.sha256((TOOLS / name).read_bytes()).hexdigest() == expected
    compile((TOOLS / name).read_text(), str(TOOLS / name), 'exec')
review, _ = bound(TOOLS / 'CONTEXT_HELPER_RECEIPT.json', REVIEW_SHA)
assert review['status'] == 'READY_FOR_READ_ONLY_SAME_NODE_CPU_CONTEXT_ANALYSIS'
assert review['helper_sha256'] == HELPER_SHA and review['expected_run'] == str(RUN)
scope = review['execution_scope']
assert scope['gpu_count'] == 0 and scope['cpu_count'] == 4 and scope['memory_gib'] == 64 and scope['time_minutes'] == 45
package, _ = bound(PACKAGE / 'manifest.json', PACKAGE_SHA)
assert package['stage'] == 'calibration110' and package['code_commit'] == review['expected_calibration_source_commit']
ready_path = RUN / 'output/probability-calibration/ready.json'
ready, ready_sha = bound(ready_path, READY_SHA)
assert ready['status'] == 'CALIBRATION_OBSERVED_ZERO_WRITES' and ready['optimizer_committed'] == 0
assert ready['parent_completed_outer'] == 110 and ready['observed_outer'] == 111 and ready['new_completed_outer'] == 0
assert ready['code_commit'] == package['code_commit'] and ready['job_id'] == '3268'
result, _ = bound(RUN / 'run-result.json')
assert result['status'] == 'CALIBRATION_OBSERVED_ZERO_WRITES_UNREVIEWED' and result['cleanup_residual'] == {}
assert len(ready['workers']) == 8 and {worker['rank'] for worker in ready['workers']} == set(range(8))
rows = 0
for worker in ready['workers']:
    for entry in worker['diagnostic_entries']:
        path = Path(entry['path']).resolve()
        assert path.is_relative_to(RUN / 'output/sampling-consistency')
        record, _ = bound(path, entry['sha256'])
        assert record['rank'] == worker['rank'] and record['saved_rows'] == record['local_rows']
        rows += record['saved_rows']
assert 0 < rows * 500 <= 20_000_000, 'Re-estimate same-node CPU resources before execution'
job_name = 'tree-context-analysis-3268-v2'
active = subprocess.check_output(['squeue', '-h', '-w', 'h100-9', '-o', '%i %j %T'], text=True)
assert 'tree-context-analysis-3268' not in active, 'Existing own CPU analysis; do not duplicate'
assert not any(line.split() and line.split()[0] == '3270' for line in active.splitlines()), 'Previous own CPU validation is still active'
journal = ROOT / 'instruct300/CPU_ANALYSIS_ONCE-3268-28e665eb-v2.json'
intent = dict(status='SUBMITTING_DO_NOT_RETRY_ON_UNKNOWN', epoch=time.time(), parent_job=3268,
              helper_sha256=HELPER_SHA, review_sha256=REVIEW_SHA, package_sha256=PACKAGE_SHA,
              ready_sha256=ready_sha, raw_rows=rows, raw_token_upper_bound=rows * 500,
              independent_probes=INDEPENDENT_PROBES,
              cpu_count=4, memory_gib=64, gpu_count=0, time_minutes=45)
with journal.open('x') as stream:
    json.dump(intent, stream); stream.flush(); os.fsync(stream.fileno())
args = [str(ROOT / 'env/bin/python'), str(TOOLS / 'analyze_tree_sampling_calibration.py'),
        '--run', str(RUN), '--package', str(PACKAGE),
        '--expected-package-manifest-sha256', PACKAGE_SHA, '--expected-ready-sha256', ready_sha]
verify = 'import hashlib,pathlib;files=' + repr({str(TOOLS / 'analyze_tree_sampling_calibration.py'): HELPER_SHA,
          **{str(TOOLS / name): value for name, value in INDEPENDENT_PROBES.items()}}) + ';' + \
          'assert all(hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()==s for p,s in files.items())'
compile(verify, '<bound-code-sha-verification>', 'exec')
probe_commands = ''
for probe_name in INDEPENDENT_PROBES:
    prefix = 'raw-stats' if probe_name == 'raw_stats_cpu_v2.py' else 'tree-trajectory'
    probe_commands += shlex.join([str(ROOT / 'env/bin/python'), str(TOOLS / probe_name)]) + ' > ' + \
        shlex.quote(str(RUN)) + '/' + prefix + '-${SLURM_JOB_ID}.json\n'
body = ('#!/usr/bin/env bash\nset -euo pipefail\nset -o noclobber\n'
        'export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4\n'
        + shlex.join([str(ROOT / 'env/bin/python'), '-c', verify]) + '\n' + probe_commands +
        'exec ' + shlex.join(args) + '\n')
try:
    proc = subprocess.run(['sbatch', '--parsable', '--job-name=' + job_name, '--nodes=1', '--nodelist=h100-9',
           '--gres=gpu:0', '--cpus-per-task=4', '--mem=64G', '--time=00:45:00', '--no-requeue',
           '--output=' + str(RUN / 'context-analysis-%j.json'), '--error=' + str(RUN / 'context-analysis-%j.err')],
           input=body, capture_output=True, text=True, timeout=60)
    intent.update(returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
    if proc.returncode == 0:
        job = proc.stdout.strip().split(';')[0]
        assert job.isdigit(), proc.stdout
        intent.update(status='SUBMITTED', job_id=int(job))
    else:
        intent['status'] = 'SUBMISSION_FAILED_NO_AUTOMATIC_RETRY'
except BaseException as error:
    intent.update(status='SUBMISSION_UNKNOWN_DO_NOT_RETRY', exception=repr(error))
    raise
finally:
    temporary = journal.with_suffix('.final.tmp')
    with temporary.open('x') as stream:
        json.dump(intent, stream, indent=2); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, journal)
    print(json.dumps(intent, allow_nan=False), flush=True)
