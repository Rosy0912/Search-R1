"""User-approved trained step85 teacher; reuse the existing student allocation."""
import hashlib
import json
import os
import pathlib
import socket
import subprocess
import sys
import time

ROOT = pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd')
RUN = ROOT / 'logs/topd-instruct300-teacher85-approved-v1'
TEACHER_RUN = ROOT / 'logs/r1-standard32b-grpo100-common-v5-r1'


def atomic_new(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())


def main():
    if socket.gethostname() != 'h100-11' or os.environ.get('SLURM_JOB_ID') != '3224':
        raise ValueError('Only the existing authorized student job3224 on h100-11')
    RUN.mkdir(mode=0o700, exist_ok=False)
    actual_path = TEACHER_RUN / 'actual-outer.json'
    actual_bytes = actual_path.read_bytes()
    actual = json.loads(actual_bytes)
    if actual['actual_completed_outer_steps'] != list(range(1, 90)):
        raise ValueError('Original teacher actual89 provenance changed')
    teacher = TEACHER_RUN / 'checkpoints/actor/global_step_85'
    if not (teacher / 'model.safetensors.index.json').is_file():
        raise ValueError('Saved trained teacher85 is absent')
    approval = RUN / 'teacher85-user-approval.json'
    atomic_new(approval, {
        'status': 'USER_APPROVED_TRAINED_CHECKPOINT',
        'checkpoint_step': 85,
        'original_teacher_job': 3228,
        'original_teacher_completed_outer': 89,
        'original_teacher_run_status': 'FAILED',
        'human_instruction': '可以直接进行TOPD，这个teacher的reward应该足够好了',
        'export': {'directory': str(teacher), 'files': None},
        'identity_scope': 'Full HF inventory and hashes are read once by student supervisor before model startup',
        'actual_outer_ref': {'path': str(actual_path), 'sha256': hashlib.sha256(actual_bytes).hexdigest()},
    })
    config = RUN / 'static-config.json'
    subprocess.run([
        str(ROOT / 'env/bin/python'), str(ROOT / 'tools/bind_topd_instruct300_v8.py'),
        '--infrastructure', str(TEACHER_RUN / 'evidence/effective-runtime-config.json'),
        '--teacher-result', str(approval),
        '--teacher-result-sha256', hashlib.sha256(approval.read_bytes()).hexdigest(),
        '--run-directory', str(RUN / 'evidence'), '--output', str(config),
    ], check=True)
    # External work outside Slurm may occupy these cards. Do not initialize CUDA
    # until all eight cards are actually free; never signal another owner's PID.
    previous = None
    while True:
        gpu = subprocess.run([
            'nvidia-smi', '--query-gpu=uuid,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=20)
        apps = subprocess.run([
            'nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=20)
        rows = [line.split(',') for line in gpu.stdout.splitlines() if line.strip()]
        free = (gpu.returncode == apps.returncode == 0 and len(rows) == 8
                and not apps.stdout.strip()
                and all(float(row[1]) < 4096 and float(row[2]) <= 15 for row in rows))
        status = 'GPU_READY' if free else 'WAITING_EXTERNAL_GPU_WORK'
        state = {'status': status, 'epoch': time.time(), 'gpu': gpu.stdout,
                 'processes': apps.stdout, 'gpu_returncode': gpu.returncode,
                 'apps_returncode': apps.returncode, 'no_cuda_initialized': True}
        temp = RUN / 'resource-wait.tmp'
        temp.write_text(json.dumps(state, indent=2) + '\n')
        os.replace(temp, RUN / 'resource-wait.json')
        if status != previous:
            print(json.dumps(state), flush=True)
            previous = status
        if free:
            break
        time.sleep(60)
    os.environ.update(
        AGENT_ROOT=str(ROOT), PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        PYTORCH_CUDA_ALLOC_CONF='backend:native,max_split_size_mb:128',
        PYTHONPATH=str(ROOT / 'repo-terminal-fixed413') + ':' + str(ROOT / 'tools/topd-instruct300-launch-v6/runtime'))
    argv = [str(ROOT / 'env/bin/python'),
            str(ROOT / 'repo-terminal-fixed413/search_r1/baselines/preflight.py'), '--',
            str(ROOT / 'env/bin/python'), str(ROOT / 'tools/topd-instruct300-launch-v6/runtime/supervise.py'),
            '--config', str(config)]
    os.execv(argv[0], argv)


if __name__ == '__main__':
    main()
