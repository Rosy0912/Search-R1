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
import stat
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


def check_selected_gpus(selected, inventory, processes, require_idle=True):
    if not selected or len(set(selected)) != len(selected):
        raise ValueError('CUDA_VISIBLE_DEVICES must select distinct GPUs')
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
        if require_idle and (record['utilization'] > 15 or 100 * record['memory_used'] / record['memory_total'] > 15):
            raise RuntimeError(f'Allocated GPU {device} exceeds the 15% idle thresholds')
        # Existing processes are recorded, never stopped; eligibility uses both thresholds.
        records.append(record)
    if len({r['uuid'] for r in records}) != len(records):
        raise ValueError('GPU indices and UUIDs resolve to duplicate devices')
    return records


def paths(value):
    return [value] if isinstance(value, str) else list(value)


def _process_identity(pid):
    """Linux identity, including ppid so reparenting cannot fake an ancestor."""
    fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
    if fields[0] in ('Z', 'X'):
        raise ValueError('Resource guard ancestor is not alive')
    return int(fields[19]), int(fields[1])


def validate_resource_guard():
    """Verify a real live same-run resource_watch ancestor, not an env assertion.

    Watcher exports SEARCH_R1_GPU_RESOURCE_GUARD={pid,start_ticks,run} only to
    supervised children. BAPO_RUN_ID must already be in its startup environment.
    """
    try:
        guard = json.loads(os.environ['SEARCH_R1_GPU_RESOURCE_GUARD'])
        pid, ticks = int(guard['pid']), int(guard['start_ticks'])
        run = Path(os.environ['BAPO_RUN_ID'])
        if (not run.is_absolute() or run.parent.name != 'runs' or
                str(run.resolve()) != str(run) or guard['run'] != str(run) or
                pid <= 1 or ticks <= 0 or pid == os.getpid()):
            raise ValueError('Invalid resource guard run/identity')
        expected_script = run.parent.parent / 'tools' / 'resource_watch.py'
        chain, current = [], os.getpid()
        for _ in range(256):
            before = _process_identity(current)
            after = _process_identity(current)
            if before != after:
                raise ValueError('Resource guard ancestor identity changed')
            chain.append((current, before))
            if current == pid:
                if before[0] != ticks:
                    raise ValueError('Resource guard PID has been reused')
                proc = Path('/proc') / str(pid)
                argv = [v.decode() for v in (proc / 'cmdline').read_bytes().split(b'\0') if v]
                env = (proc / 'environ').read_bytes().split(b'\0')
                if (len(argv) < 2 or argv[1] != str(expected_script) or
                        not expected_script.is_file() or
                        not (proc / 'exe').resolve().name.startswith('python')):
                    raise ValueError('Ancestor is not the exact own resource_watch script')
                run_arguments = [argv[i + 1] for i, v in enumerate(argv[:-1]) if v == '--run']
                run_arguments += [v.split('=', 1)[1] for v in argv if v.startswith('--run=')]
                if run_arguments != [str(run)] or ('BAPO_RUN_ID=' + str(run)).encode() not in env:
                    raise ValueError('Resource guard is supervising another run')
                # Recheck the complete parent chain after cmdline/environment reads.
                if any(_process_identity(p) != identity for p, identity in chain):
                    raise ValueError('Resource guard ancestry changed during validation')
                return guard
            current = before[1]
            if current <= 1 or current in {p for p, _ in chain}:
                break
        raise ValueError('Resource guard is not an actual ancestor')
    except (KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
        raise ValueError('Cannot verify live resource guard') from exc


def validate_inherited_gpu_leases(selected, require_guard=False):
    """Busy own cards require a verified supervisor; all paths verify real locks."""
    records = json.loads(os.environ.get('SEARCH_R1_GPU_LEASES', '[]'))
    if os.environ.get('SEARCH_R1_GPU_PREFLIGHT') != '1' or not isinstance(records, list) or len(records) != len(selected):
        raise ValueError('Start training through reproduction/baselines/run.sh for GPU leases')
    supervised = 'SEARCH_R1_GPU_RESOURCE_GUARD' in os.environ
    if require_guard and not supervised:
        raise ValueError('GPU lease reuse requires a live resource_watch guard')
    if supervised:
        validate_resource_guard()
    inventory, processes = gpu_snapshot()
    allocated = check_selected_gpus(selected, inventory, processes, require_idle=not supervised)
    if {r['uuid'] for r in records} != {r['uuid'] for r in allocated}:
        raise ValueError('Inherited GPU leases differ from selected hardware')
    if len({int(r['fd']) for r in records}) != len(records):
        raise ValueError('Duplicate GPU lease descriptors')
    for record in records:
        fd = int(record['fd'])
        path = Path('/tmp') / f'search-r1-gpu-leases-{os.getuid()}' / (record['uuid'] + '.lock')
        descriptor, named = os.fstat(fd), path.lstat()
        if (not stat.S_ISREG(descriptor.st_mode) or not stat.S_ISREG(named.st_mode) or
                descriptor.st_uid != os.getuid() or
                (descriptor.st_dev, descriptor.st_ino) != (named.st_dev, named.st_ino)):
            raise ValueError('GPU lease descriptor does not identify the lease file')
        probe = os.open(str(path), os.O_RDWR)
        try:
            # A shared probe must be blocked by the existing EXCLUSIVE lease.
            try:
                fcntl.flock(probe, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                raise ValueError('GPU lease is not held exclusively by the launcher')
            # Same open-file-description succeeds; an unrelated fd for a lock
            # held elsewhere fails. Do not accept merely "somebody holds it".
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError('Inherited descriptor does not hold the GPU lease') from exc
        finally:
            os.close(probe)
        os.set_inheritable(fd, True)
    return allocated


def validate_gpu_lease(config):
    """Validate final Hydra counts plus idle or verified supervised leases."""
    if config.trainer.nnodes != 1:
        raise ValueError('This launcher supports one node; multi-node requires per-node scheduler/lease gates')
    selected = [v.strip() for v in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if v.strip()]
    if len(selected) != config.trainer.n_gpus_per_node:
        raise ValueError('Final n_gpus_per_node does not match CUDA_VISIBLE_DEVICES')
    validate_inherited_gpu_leases(selected)


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
    if config.get('baseline_aligned', False):
        from .baseline_alignment_retrieval import validate_retriever_identity
        actor_tokenizer = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path, local_files_only=True)
        receipt = validate_retriever_identity(config, actor_tokenizer)
        print('BASELINE_ALIGNED_RETRIEVER_IDENTITY_JSON=' + json.dumps(receipt, sort_keys=True), flush=True)
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
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('Specify a command after --')
    selected = [v.strip() for v in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if v.strip()]
    if os.environ.get('SEARCH_R1_GPU_PREFLIGHT') or os.environ.get('SEARCH_R1_GPU_LEASES') or os.environ.get('SEARCH_R1_GPU_RESOURCE_GUARD'):
        records = validate_inherited_gpu_leases(selected, require_guard=True)
        print('BASELINE_GPU_LEASE_REUSED', json.dumps([r['uuid'] for r in records]), flush=True)
        os.execvp(command[0], command)
        return
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
