"""Read-only deployment checks and process-scoped GPU leases; never stop other jobs."""
import argparse
import fcntl
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def parse_gpu_inventory(text):
    result = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        index, uid, used, total, utilization = [v.strip() for v in line.split(',')]
        result[index] = dict(uuid=uid, memory_used=float(used), memory_total=float(total),
                             utilization=float(utilization))
    return result


def check_selected_gpus(selected, inventory, processes):
    if not selected or len(set(selected)) != len(selected):
        raise ValueError('CUDA_VISIBLE_DEVICES must select distinct GPUs')
    if os.environ.get('RECURSIVE_BRAN_STRICT_GPU_GATE')=='1':
        from .recursive_admission import exact_gpu_gate
        exact_gpu_gate(selected,inventory)
    records = []
    for device in selected:
        matches = [v for index, v in inventory.items() if device == index or device == v['uuid']]
        if len(matches) != 1:
            raise ValueError(f'Cannot resolve allocated GPU {device}')
        record = matches[0]
        if (not all(math.isfinite(record[k]) for k in ('utilization', 'memory_used', 'memory_total'))
                or record['memory_total'] <= 0 or record['memory_used'] < 0
                or not 0 <= record['utilization'] <= 100):
            raise RuntimeError(f'Allocated GPU {device} has unknown/invalid occupancy')
        if record['utilization'] > 15 or 100 * record['memory_used'] / record['memory_total'] > 15:
            raise RuntimeError(f'Allocated GPU {device} exceeds the 15% idle thresholds')
        # Existing processes are recorded, never stopped; eligibility uses both thresholds.
        records.append(record)
    if len({r['uuid'] for r in records}) != len(records):
        raise ValueError('GPU indices and UUIDs resolve to duplicate devices')
    return records


def paths(value):
    return [value] if isinstance(value, str) else list(value)


def validate_gpu_lease(config):
    """Validate final Hydra resource counts and inherited launcher leases."""
    if config.trainer.nnodes != 1:
        raise ValueError('This launcher supports one node; multi-node requires per-node scheduler/lease gates')
    validate_inherited_gpu_leases(config.trainer.n_gpus_per_node)


def validate_inherited_gpu_leases(expected_count=None):
    """Reuse a parent-held lease without reacquiring and deadlocking its own flock."""
    selected = [v.strip() for v in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if v.strip()]
    if not selected:
        raise ValueError('Inherited GPU leases require selected allocated GPUs')
    if expected_count is not None and len(selected) != expected_count:
        raise ValueError('Final n_gpus_per_node does not match CUDA_VISIBLE_DEVICES')
    records = json.loads(os.environ.get('SEARCH_R1_GPU_LEASES', '[]'))
    if os.environ.get('SEARCH_R1_GPU_PREFLIGHT') != '1' or len(records) != len(selected):
        raise ValueError('Start training through reproduction/baselines/run.sh for GPU leases')
    inventory, processes = gpu_snapshot()
    allocated = check_selected_gpus(selected, inventory, processes)
    if {r['uuid'] for r in records} != {r['uuid'] for r in allocated}:
        raise ValueError('Inherited GPU leases differ from selected hardware')
    for record in records:
        fd = int(record['fd'])
        path = Path('/tmp') / f'search-r1-gpu-leases-{os.getuid()}' / (record['uuid'] + '.lock')
        if os.fstat(fd).st_ino != path.stat().st_ino:
            raise ValueError('GPU lease descriptor does not identify the lease file')
        # A new open file description must fail to acquire the already-held lease.
        probe = os.open(str(path), os.O_RDWR)
        try:
            try:
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                raise ValueError('GPU lease is not held by the launcher')
            # A blocked independent probe alone could be a foreign holder.
            # This is idempotent for the inherited locked open description;
            # an unlocked lookalike FD cannot acquire a foreign-held lease.
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError('Inherited descriptor does not own the GPU lease') from error
        finally:
            os.close(probe)


def validate_assets_and_retriever(config):
    """Run before Ray/model allocation; no downloads or foreign service management."""
    from transformers import AutoConfig, AutoTokenizer
    import requests
    for module in ('torch', 'vllm', 'flash_attn', 'ray', 'tensordict'):
        importlib.import_module(module)
    model_paths = [config.actor_rollout_ref.model.path]
    method = config.baselines.method
    if method in ('opsd', 'topd'):
        model_paths.append(config.baselines.distillation.teacher_path)
    if method == 'cosearch':
        model_paths.append(config.baselines.ranker.model_path)
    for value in model_paths:
        path = Path(value).expanduser()
        if not path.is_dir():
            raise ValueError(f'Missing offline model directory: {path}')
        AutoConfig.from_pretrained(str(path), local_files_only=True)
        AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        if not any(path.glob('*.safetensors')) and not any(path.glob('pytorch_model*.bin')):
            raise ValueError(f'Missing model weights: {path}')
    dataset_paths = paths(config.data.train_files) + paths(config.data.val_files)
    if config.data.get('final_val_files'):
        dataset_paths += paths(config.data.final_val_files)
    if method == 'bapo':
        dataset_paths += paths(config.baselines.bapo.stage_validation_files)
    for value in dataset_paths:
        if not Path(value).expanduser().is_file():
            raise ValueError(f'Missing dataset: {value}')
    count = config.baselines.ranker.candidates if method == 'cosearch' else config.retriever.topk
    response = requests.post(config.retriever.url,
                             json={'queries': ['What is the capital of France?'], 'topk': count,
                                   'return_scores': True}, timeout=(5, 30))
    response.raise_for_status()
    payload = response.json()
    results = payload.get('result') if isinstance(payload, dict) else None
    if not isinstance(results, list) or len(results) != 1 or not isinstance(results[0], list):
        raise ValueError('Retriever returned an incompatible response')
    if len(results[0]) < count:
        raise ValueError(f'Retriever returned fewer than {count} candidates')
    for doc in results[0]:
        if not isinstance(doc, dict) or not isinstance(doc.get('document'), dict) or not isinstance(doc['document'].get('contents'), str):
            raise ValueError('Retriever response lacks document.contents')
    print('BASELINE_ASSET_RETRIEVER_PREFLIGHT_OK', flush=True)


def gpu_snapshot():
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu',
                                          '--format=csv,noheader,nounits'], text=True)
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                    '--format=csv,noheader,nounits'], text=True)
    processes = {line.split(',')[0].strip() for line in apps.splitlines() if ',' in line}
    print('BASELINE_GPU_OCCUPANCY_JSON=' + json.dumps(dict(inventory=parse_gpu_inventory(inventory),
          compute_processes=apps.splitlines()), sort_keys=True), flush=True)
    return parse_gpu_inventory(inventory), processes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reuse-inherited', action='store_true')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('Specify a command after --')
    if args.reuse_inherited:
        validate_inherited_gpu_leases()
        for record in json.loads(os.environ['SEARCH_R1_GPU_LEASES']):
            os.set_inheritable(int(record['fd']), True)
        print('BASELINE_INHERITED_GPU_LEASES_OK', flush=True)
        os.execvp(command[0], command)
        return
    selected = [v.strip() for v in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if v.strip()]
    inventory, processes = gpu_snapshot()
    records = check_selected_gpus(selected, inventory, processes)
    locks = []
    # Shared across all baseline checkouts for this OS user, unlike per-run leases.
    lock_dir = Path('/tmp') / f'search-r1-gpu-leases-{os.getuid()}'
    lock_dir.mkdir(mode=0o700, exist_ok=True)
    for record in sorted(records, key=lambda r: r['uuid']):
        fd = os.open(str(lock_dir / (record['uuid'] + '.lock')), os.O_CREAT | os.O_RDWR, 0o600)
        locks.append(fd)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Re-check after acquiring leases. Other users/schedulers do not honor these leases;
    # an externally allocated exclusive GPU job is still required.
    time.sleep(.2)
    inventory, processes = gpu_snapshot()
    check_selected_gpus(selected, inventory, processes)
    os.environ['SEARCH_R1_GPU_PREFLIGHT'] = '1'
    for fd in locks:
        os.set_inheritable(fd, True)
    os.environ['SEARCH_R1_GPU_LEASES'] = json.dumps([dict(uuid=r['uuid'], fd=fd) for r, fd in zip(sorted(records, key=lambda r: r['uuid']), locks)])
    print('BASELINE_GPU_PREFLIGHT_OK', json.dumps([r['uuid'] for r in records]), flush=True)
    os.execvp(command[0], command)


if __name__ == '__main__':
    main()
