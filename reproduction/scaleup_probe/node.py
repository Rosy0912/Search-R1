"""One bounded orchestrator per allocated node. No submit or production IO."""
import json
import os
from pathlib import Path
import socket
import signal
import subprocess
import sys
import time

from .contract import node_coordinates, read_plan, sha, validate_gate


def main():
    pp = Path(sys.argv[1]); plan = read_plan(pp); root = Path(plan['root'])
    node, master, port = node_coordinates(os.environ)
    directory = root / ('node-' + str(node)); directory.mkdir(exist_ok=False)
    with (directory / 'entry-once.json').open('x') as f:
        json.dump(dict(at=time.time(), node=node, hostname=socket.gethostname(),
                       plan_sha256=sha(pp), master=master, port=port,
                       allowlisted_env={k: os.environ.get(k) for k in (
                           'NODE_RANK', 'NODE_COUNT', 'WORLD_SIZE', 'RANK',
                           'MASTER_ADDR', 'MASTER_PORT', 'CUDA_VISIBLE_DEVICES',
                           'NVIDIA_VISIBLE_DEVICES')}), f, indent=2)
    deadline = time.monotonic() + 300
    while not (root / 'scheduler-release-once.json').exists():
        assert time.monotonic() < deadline, 'No independently registered scheduler gate'
        time.sleep(2)
    submission = json.loads((root / 'submission.json').read_text())
    gate = json.loads((root / 'scheduler-release-once.json').read_text())
    validate_gate(gate, submission, sha(pp))
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,memory.total',
                                    '--format=csv,noheader,nounits'], text=True, timeout=20)
    devices = [line.split(',') for line in rows.splitlines() if line.strip()]
    assert len(devices) == 8 and len({d[0].strip() for d in devices}) == 8
    assert all('A100' in d[1] and int(d[2].strip()) >= 79000 for d in devices)
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
                                        '--format=csv,noheader'], text=True, timeout=20).strip()
    assert not processes, 'Allocated GPUs already have compute processes; do not interfere'
    with (directory / 'hardware.json').open('x') as f:
        json.dump(dict(devices=devices, no_prior_gpu_processes=True,
                       scheduler_uid=submission['uid']), f, indent=2)
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               NCCL_DEBUG='WARN', NCCL_NVLS_ENABLE='0', TORCH_NCCL_ASYNC_ERROR_HANDLING='1')
    for name in ('RAY_ADDRESS', 'ROCR_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES'):
        env.pop(name, None)
    cmd = [plan['runtime'], '-m', 'torch.distributed.run', '--nnodes=2',
           '--nproc-per-node=8', '--node-rank=' + str(node), '--master-addr=' + master,
           '--master-port=' + str(port), '--max-restarts=0',
           '-m', 'reproduction.scaleup_probe.worker', str(pp)]
    with (directory / 'torchrun.log').open('xb') as log:
        process = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            rc = process.wait(timeout=plan['timeout_seconds'])
        except subprocess.TimeoutExpired:
            # Only this newly created probe process group; never other jobs.
            os.killpg(process.pid, signal.SIGTERM)
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait(timeout=10)
            rc = 124
    (directory / 'exit.code').open('x').write(str(rc) + '\n')
    assert rc == 0, 'Unique probe failed; preserve evidence, never auto retry'


if __name__ == '__main__':
    main()
