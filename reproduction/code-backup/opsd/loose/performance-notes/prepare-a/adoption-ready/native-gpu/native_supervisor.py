"""UNRUN adopted-source micro2 capacity only; no Ray/vLLM/production iterations."""
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time
import shutil
from search_r1.baselines.preflight import gpu_snapshot, validate_inherited_gpu_leases

ROOT = Path(os.environ['SEARCH_R1_BASE'])
OUT = Path(os.environ['OPSD_EVIDENCE'])
JOB = os.environ['SLURM_JOB_ID']
COMMIT = os.environ['OPSD_EXPECTED_COMMIT']
ACTOR_ALLOCATOR = os.environ.get('OPSD_ACTOR_ALLOCATOR_CONF', '')
assert ACTOR_ALLOCATOR == '' and os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '') == ''
ACTOR_ENV = dict(os.environ)
HERE = Path(__file__).resolve().parent
assert COMMIT != 'e43fbfa73088ed70ed3efef79ba561507c0afc33'
assert hashlib.sha256((Path(os.environ['SEARCH_R1_REPO'])/'search_r1/baselines/distillation.py').read_bytes()).hexdigest() == '2131315b6770779f0c7579bf2a6139e742cf75de03b7ee9f44662f8c32a870cf'
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == COMMIT
assert socket.gethostname().split('.')[0] == os.environ['SLURM_JOB_NODELIST'] == 'h100-10'
assert re.search(r'(?:^|/)job_' + re.escape(JOB) + r'(?:/|$)', Path('/proc/self/cgroup').read_text(), re.M)
LEASE_FDS = tuple(int(record['fd']) for record in json.loads(os.environ['SEARCH_R1_GPU_LEASES']))
assert len(set(LEASE_FDS)) == 8
RETRIEVER_PID = int((OUT / 'retriever.pid').read_text())


def own_service_guard():
    os.kill(RETRIEVER_PID, 0)
    assert re.search(r'(?:^|/)job_' + re.escape(JOB) + r'(?:/|$)',
                     Path(f'/proc/{RETRIEVER_PID}/cgroup').read_text(), re.M)
    listeners = subprocess.check_output(['ss', '-ltnp', 'sport = :18084'], text=True)
    assert f'pid={RETRIEVER_PID},' in listeners and '127.0.0.1:18084' in listeners, listeners
    rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                   '--format=csv,noheader,nounits'], text=True).splitlines()
    parsed = [(row.split(',')[0].strip(), int(row.split(',')[1])) for row in rows if row.strip()]
    inventory, _ = gpu_snapshot()
    selected = [device.strip() for device in os.environ['CUDA_VISIBLE_DEVICES'].split(',') if device.strip()]
    expected = {record['uuid'] for index, record in inventory.items()
                if index in selected or record['uuid'] in selected}
    assert len(expected) == 8 and len(parsed) == 8
    assert {uuid for uuid, _ in parsed} == expected and all(pid == RETRIEVER_PID for _, pid in parsed), parsed
    return dict(selected_uuids=sorted(expected), only_gpu_service_pid=RETRIEVER_PID,
                listener= listeners, hostname=socket.gethostname(), job_id=JOB)


def wait_for_idle():
    errors = []
    for attempt in range(60):
        try:
            service = own_service_guard()
            validate_inherited_gpu_leases(8)  # Both thresholds; no own-service exemption.
            return dict(attempts=attempt + 1, service=service, previous_errors=errors)
        except (RuntimeError, AssertionError) as error:
            errors.append(dict(attempt=attempt + 1, type=type(error).__name__, message=str(error)))
            if attempt == 59:
                raise
            time.sleep(2)


summary = dict(job_id=JOB, source_commit=COMMIT, passed=False, actual_training_iterations=0,
               local_micro=2, global_micro=16, input_width=6144, response_width=4096,
               numeric_equivalence_proof='Previous external2990 short paired only; not this capacity job',
               vllm_coexistence_tested=False, production_five_or_hf_tested=False)
try:
    for fd in LEASE_FDS: os.fstat(fd)
    summary['before_idle'] = wait_for_idle()
    command = ['torchrun', '--standalone', '--nproc_per_node=8', str(HERE/'microbatch_probe.py'),
               '--micro', '2', '--input-width', '6144', '--response-width', '4096',
               '--model', os.environ['OPSD_TEACHER'], '--output-dir', str(OUT)]
    with (OUT/'micro2-native.log').open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                pass_fds=LEASE_FDS, env=ACTOR_ENV)
    path = OUT/'micro-2.json'
    record = json.loads(path.read_text()) if path.exists() else None
    ranks = record.get('ranks', []) if record else []
    passed = (result.returncode == 0 and record is not None and record.get('passed') is True
              and len(ranks) == 8 and {r['rank'] for r in ranks} == set(range(8))
              and all(r['passed'] and r['job_id'] == JOB and r['local_micro'] == 2
                      and r['optimizer_steps'] == 2 and r['input_width'] == 6144
                      and r['response_width'] == 4096 for r in ranks))
    summary.update(returncode=result.returncode, result_sha256=hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None)
    if not passed: raise RuntimeError('Actual adopted-source native eight-rank capacity failed')
    summary['after_idle'] = wait_for_idle()
    summary['passed'] = True
finally:
    (OUT/'adopted-native-capacity.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2), flush=True)
