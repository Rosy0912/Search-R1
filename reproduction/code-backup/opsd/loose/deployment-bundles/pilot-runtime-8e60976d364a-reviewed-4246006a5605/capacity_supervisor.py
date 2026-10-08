"""Capacity-only gate: inherited leases, live local retrieval, isolated eight-rank candidates.

No numerical diagnostic, Ray, vLLM, rollout, or actual training is launched here.
A passing synthetic update is not a completed production training iteration.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time
from search_r1.baselines.preflight import gpu_snapshot, validate_inherited_gpu_leases

ROOT = Path(os.environ['SEARCH_R1_BASE'])
OUT = Path(os.environ['OPSD_EVIDENCE'])
JOB = os.environ['SLURM_JOB_ID']
COMMIT = os.environ['OPSD_EXPECTED_COMMIT']
ACTOR_ALLOCATOR = os.environ.get('OPSD_ACTOR_ALLOCATOR_CONF', '')
assert ACTOR_ALLOCATOR in ('', 'backend:native,max_split_size_mb:128')
ACTOR_ENV = dict(os.environ)
if ACTOR_ALLOCATOR: ACTOR_ENV['PYTORCH_CUDA_ALLOC_CONF'] = ACTOR_ALLOCATOR
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


summary = dict(actor_allocator_configuration=ACTOR_ALLOCATOR, job_id=JOB, source_commit=COMMIT, passed=False, max_safe_local_micro=None,
               actual_training_iterations=0, numeric_probe_launched=False, ray_launched=False,
               vllm_coexistence_tested=False, candidates=[],
               scope='Synthetic worst-width 6144 input / 4096 response eight-rank full backward and optimizer with live same-node GPU E5/FAISS. Not production outer training, numerical equivalence, or vLLM coexistence acceptance.')
try:
    for micro in (1, 2, 4, 8):
        idle = wait_for_idle()
        command = ['torchrun', '--standalone', '--nproc_per_node=8',
                   str(ROOT / 'tools/pilot-runtime-8e60976d364a-reviewed-4246006a5605/microbatch_probe.py'), '--micro', str(micro),
                   '--input-width', '6144', '--response-width', '4096',
                   '--model', os.environ['OPSD_TEACHER'], '--output-dir', str(OUT)]
        start = time.perf_counter()
        with (OUT / f'micro-{micro}.log').open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, pass_fds=LEASE_FDS, env=ACTOR_ENV)
        record_path = OUT / f'micro-{micro}.json'
        record = json.loads(record_path.read_text()) if record_path.exists() else None
        passed = result.returncode == 0 and record is not None and record.get('passed') is True
        if passed:
            ranks = record.get('ranks', [])
            passed = (len(ranks) == 8 and {row['rank'] for row in ranks} == set(range(8))
                      and all(row.get('passed') is True and row['job_id'] == JOB
                              and row['local_micro'] == micro for row in ranks))
        summary['candidates'].append(dict(local_micro=micro, returncode=result.returncode,
            passed=passed, wall_seconds=time.perf_counter()-start, before_idle=idle,
            aggregate_json=str(record_path) if record_path.exists() else None,
            aggregate_sha256=hashlib.sha256(record_path.read_bytes()).hexdigest() if record_path.exists() else None))
        (OUT / 'micro-capacity.json').write_text(json.dumps(summary, indent=2))
        print('CAPACITY_CANDIDATE', micro, 'PASS' if passed else 'FAIL', flush=True)
        if not passed:
            break
    summary['after_idle'] = wait_for_idle()
    safe = [row['local_micro'] for row in summary['candidates'] if row['passed']]
    if not safe:
        raise RuntimeError('No eight-rank full-backward capacity candidate passed')
    summary.update(passed=True, max_safe_local_micro=max(safe), max_safe_global_micro=8*max(safe))
finally:
    (OUT / 'micro-capacity.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2), flush=True)
