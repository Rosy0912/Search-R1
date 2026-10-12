"""Owned new prefix-only official3B native; no retained formal training."""
import fcntl
import inspect
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from owned_runtime_resume_v1 import ROOT, require, sha, cgroup_scope, OwnedProcesses, gpu_sample
from action_prefix_native_control_v1 import SCOPE, validate_binding, validate_result
from native_guard_binding_v2 import validate_guard_helper_binding

require(len(sys.argv) == 3, 'Expected bound artifact path and SHA')
binding = Path(sys.argv[1])
require(sha(binding) == sys.argv[2], 'Native binding SHA mismatch')
value = json.loads(binding.read_text())
helpers = Path(__file__).resolve().parent
validate_binding(value, helpers, sha)
for item in value['files']:
    require(sha(item['path']) == item['sha256'], 'Full source/environment/asset/helper closure changed')
guard_identity = validate_guard_helper_binding(helpers, value, require, sha)
require(Path(os.environ['GIGPO_RESUME_REPO']).resolve() == Path(value['repo']).resolve(), 'Bound source path differs')
require(subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == value['code_commit'], 'Source commit differs')
require(socket.gethostname().split('.')[0] == 'h100-8', 'Wrong node')
job = os.environ['SLURM_JOB_ID']
scheduler = subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True)
require('Requeue=0' in scheduler.split() and 'Restarts=0' in scheduler.split(), 'Restart/requeue forbidden')
require(os.environ['SLURM_JOB_NODELIST'] == 'h100-8' and len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) == 8,
        'Not full8 node allocation')
from search_r1.baselines.preflight import validate_inherited_gpu_leases, gpu_snapshot, check_selected_gpus
validate_inherited_gpu_leases(8)
lock = (ROOT / 'gpu-job.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
    os.environ.pop(key, None)
os.environ.update(NO_PROXY='localhost,127.0.0.1,::1', no_proxy='localhost,127.0.0.1,::1',
    OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NCCL_NVLS_ENABLE='0',
    GIGPO_POLICY_NATIVE_BINDING=str(binding), GIGPO_POLICY_NATIVE_BINDING_SHA256=sys.argv[2],
    SEARCH_R1_BASELINE_COMMIT=value['code_commit'], ACTION_PREFIX_NATIVE_BINDING=str(binding))
os.environ['CPATH'] = '/mnt/data/shaoshitong/search-r1-rl/gigpo/tools/python310-headers-v1/extracted/usr/include:/mnt/data/shaoshitong/search-r1-rl/gigpo/tools/python310-headers-v1/extracted/usr/include/python3.10:/mnt/data/shaoshitong/search-r1-rl/gigpo/tools/python310-headers-v1/extracted/usr/include/x86_64-linux-gnu/python3.10'
run = ROOT / 'runs' / f'gigpo-e1-action-prefix-native-job{job}'
run.mkdir()
for key in list(os.environ):
    if key.startswith('SEARCH_R1_BASELINE_AUDIT'):
        os.environ.pop(key, None)
os.environ['NUMERIC_FIXTURE_OUTPUT'] = str(run / 'gpu-records')
scope, relative = cgroup_scope(job)
owned = OwnedProcesses(scope, relative)
inventory, processes = gpu_snapshot()
selected = check_selected_gpus(os.environ['CUDA_VISIBLE_DEVICES'].split(','), inventory, processes)
(ROOT / 'tools/expected-gpu-uuids.json').write_text(json.dumps([gpu['uuid'] for gpu in selected]))
import torch
import transformers
import search_r1.baselines.action_prefix_forward as actual_prefix
require(inspect.getfile(actual_prefix) == str(Path(value['repo']) / 'search_r1/baselines/action_prefix_forward.py'),
        'Wrong imported new prefix module')
result = dict(status='FAILED', native_scope=SCOPE, validation_scope=value['validation_scope'],
    code_commit=value['code_commit'], binding_sha256=sha(binding), job=job, node='h100-8', cgroup=str(scope),
    native_initialization='OFFICIAL_MODEL_INITIALIZATION', native_actual_restore=False,
    formal_updates_contributed=0, guard_identity=guard_identity, initial_gpu=gpu_sample(),
    runtime_identity=dict(executable=sys.executable, prefix=sys.prefix, python=sys.version,
        torch=torch.__version__, transformers=transformers.__version__, prefix_module=inspect.getfile(actual_prefix)))


def interrupted(sig, frame):
    raise RuntimeError('Own action-prefix native interrupted')


signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGINT, interrupted)
try:
    guard = owned.spawn([sys.executable, guard_identity['resource_guard']['path'], '--output', str(run / 'guard-resources.jsonl')], run / 'guard.log')
    deadline = time.monotonic() + 30
    while not (run / 'guard-resources.ready').exists():
        require(guard.poll() is None and time.monotonic() < deadline, 'Owned resource guard failed')
        time.sleep(1)
    cpu = owned.spawn([sys.executable, '-m', 'pytest', str(helpers / 'test_action_prefix_native_control_v1.py'),
                       '-q', '-p', 'no:cacheprovider'], run / 'control-cpu.log',
                      env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
    while cpu.poll() is None:
        require(guard.poll() is None, 'Own resource guard failed')
        time.sleep(1)
    require(cpu.returncode == 0, 'New control-only CPU tests failed')
    gpu = owned.spawn([sys.executable, '-m', 'torch.distributed.run', '--nproc_per_node=8',
                       '--master_addr=127.0.0.1', '--master_port=26381', str(helpers / 'native_action_prefix_gpu_v1.py')],
                      run / 'gpu-prefix.log', env=dict(os.environ))
    while gpu.poll() is None:
        require(guard.poll() is None, 'Own resource guard failed')
        time.sleep(1)
    require(gpu.returncode == 0, 'Actual official3B prefix GPU native failed')
    records = [json.loads((run / f'gpu-records/rank-{rank}.json').read_text()) for rank in range(8)]
    for rank, record in enumerate(records):
        validate_result(record, rank)
        require(record['code_commit'] == value['code_commit'] and record['binding_sha256'] == sha(binding),
                'Actual rank source/binding identity differs')
    result.update(fixtures_passed=True, cpu_exit=cpu.returncode, gpu_exit=gpu.returncode, rank_records=records,
                  production_probability_acceptance=False)
except BaseException as error:
    result['error'] = repr(error)
    raise
finally:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        owned.cleanup()
        require(not gpu_sample()['processes'], 'Residual GPU compute after own strict cleanup')
        if result.get('fixtures_passed'):
            result['status'] = 'PASS'
    finally:
        result.update(cleanup_residual_states=getattr(owned, 'cleanup_residual', {}), final_gpu=gpu_sample())
        (run / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False))
print(json.dumps(result, allow_nan=False), flush=True)
