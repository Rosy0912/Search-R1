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
from proposal_a_fsdp_gate import pinned_files
pinned_files()
assert COMMIT == 'e43fbfa73088ed70ed3efef79ba561507c0afc33'
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


summary = dict(job_id=JOB, source_commit=COMMIT, passed=False, external_proposal_applied=False,
               production_outer_iterations=0, worst_width_capacity_tested=False,
               vllm_coexistence_tested=False, groups=[])
try:
    for fd in LEASE_FDS: os.fstat(fd)
    # Peak retained baseline snapshots about 136GB/case; reserve extra scratch.
    assert shutil.disk_usage(OUT).free >= 256 * 1024**3, 'Insufficient snapshot scratch disk'
    for case in ('nonempty', 'rank0-empty', 'global-empty'):
        for variant in ('baseline', 'proposal'):
            idle = wait_for_idle()
            command = ['torchrun', '--standalone', '--nproc_per_node=8',
                       str(HERE / 'proposal_a_fsdp_gate.py'), '--case', case, '--variant', variant,
                       '--model', os.environ['OPSD_TEACHER'], '--output-dir', str(OUT)]
            with (OUT / f'{case}-{variant}.log').open('w') as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                        pass_fds=LEASE_FDS, env=ACTOR_ENV)
            path = OUT / f'{case}-{variant}.json'
            record = json.loads(path.read_text()) if path.exists() else None
            ranks = record.get('ranks', []) if record else []
            passed = (result.returncode == 0 and record is not None and record.get('passed') is True
                      and len(ranks) == 8 and {r['rank'] for r in ranks} == set(range(8))
                      and all(r['passed'] and r['case'] == case and r['variant'] == variant
                              and r['allocation']['job_id'] == JOB for r in ranks))
            summary['groups'].append(dict(case=case, variant=variant, passed=passed,
                returncode=result.returncode, before_idle=idle, result_sha256=hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None))
            (OUT / 'proposal-a-gate.json').write_text(json.dumps(summary, indent=2))
            if not passed: raise RuntimeError('Proposal A group failed; no continuation/retry')
        # Every baseline shard comparison/hash remains in result JSON; delete
        # ONLY this successful case's owned temporary snapshot directory.
        snapshot_dir = OUT / ('snapshots-' + case)
        assert snapshot_dir.is_dir() and not snapshot_dir.is_symlink()
        shutil.rmtree(snapshot_dir)
    summary['after_idle'] = wait_for_idle()
    summary['passed'] = True
finally:
    (OUT / 'proposal-a-gate.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2), flush=True)
