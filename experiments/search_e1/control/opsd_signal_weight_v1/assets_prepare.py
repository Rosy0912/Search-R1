"""HKUST4 CPU asset binding; big inputs never move to local disk."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

STUDY = '/weka_3/shaoshitong/search-e1-opsd-signal-weight-4gpu-20261010'
ARCHIVE = '/weka_3/shaoshitong/search-e1-config-study-20261004'
RUNTIME_SHA = '3c05be8e2ae52278eaa7f28f40f4fc07a87c847fd67a7bb2217777f4f78467bc'
TRAIN_SHA = 'a334d768230fd85d3465e251aee906060150628775b78ab7ce15e5e789f7489f'
VALIDATION_SHA = '75785f9c9b32c1b35878f6922be57923f431c52d11aa410c380f159d69febd76'
INITIAL_SHA = '8cbebc6cb10203bf93a2c8b7271d31020ab1d1e7f1958a6f94f5890af3206c3d'
INITIAL_CONFIG_SHA = 'b30b7758c00a5010d0de1fe6aae667443406b676860c00b994b1f6cfe0974ccb'
INITIAL_TENSOR_SHA = '836590b798434f21160418c6471cbce3bdfac5bf4a9947c1e3e7723d057baf2f'


def proof(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('Require actual regular remote file: ' + str(path))
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 << 20), b''):
            digest.update(block)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError('File changed while binding')
    return dict(path=str(path.resolve()), sha256=digest.hexdigest(), bytes=before.st_size)


def write_once(path, value):
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode()
    path = Path(path)
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError('Existing asset binding changed: ' + str(path))
    else:
        with path.open('xb') as handle:
            handle.write(raw); handle.flush(); os.fsync(handle.fileno())
    return proof(path)


def verify_assets():
    if os.getuid() != 1001 or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('HKUST4 authorized UID1001, CPU-only visibility required')
    root = Path(STUDY)
    if root.is_symlink() or not root.resolve().is_relative_to(Path('/weka_3/shaoshitong')):
        raise ValueError('Study output escaped approved remote disk')
    root.mkdir(parents=True, exist_ok=True)
    for directory in ('receipts', 'assets/common-initial', 'cache', 'tmp', 'logs', 'runs'):
        (root / directory).mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < 100 << 30:
        raise ValueError('Remote reserve below100GiB')
    expected = json.loads((Path(ARCHIVE) / 'receipts/archive100-transfer-manifest.json').read_text())
    source = Path(ARCHIVE) / 'runs/archive/round-1-grpo/actor/global_step_100'
    if len(expected) != 12 or {p.name for p in source.iterdir() if p.is_file()} != set(expected):
        raise ValueError('Source100 actual twelve-file inventory differs')
    files = []
    for name, wanted in sorted(expected.items()):
        actual = proof(source / name)
        if any(actual[k] != wanted[k] for k in ('sha256', 'bytes')):
            raise ValueError('Source100 differs from accepted original: ' + name)
        files.append(actual)
    model = write_once(root / 'receipts/source100-model.json', dict(accepted=True, source_grpo_step=100,
        files=files, original_acceptance=proof(Path(ARCHIVE) / 'receipts/archive100-transfer-manifest.json')))
    data = {}
    for name, wanted in (('train', TRAIN_SHA), ('validation', VALIDATION_SHA)):
        actual = proof(Path(ARCHIVE) / 'data' / (name + '.jsonl'))
        if actual['sha256'] != wanted:
            raise ValueError('Original data mismatch: ' + name)
        data[name] = actual
    manifest_ref = proof(root / 'receipts/frozen-runtime-manifest.json')
    if manifest_ref['sha256'] != RUNTIME_SHA:
        raise ValueError('Frozen305 original manifest bytes differ')
    manifest = json.loads(Path(manifest_ref['path']).read_text())
    entries = manifest.get('files', manifest)
    if len(entries) != 305:
        raise ValueError('Frozen305 inventory missing')
    runtime = root / 'runtime/frozen-305'
    for name, wanted in entries.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe runtime manifest path')
        actual = proof(runtime / relative)
        sha = wanted['sha256'] if isinstance(wanted, dict) else wanted
        if actual['sha256'] != sha:
            raise ValueError('Frozen runtime changed: ' + name)
    adapter_dir = root / 'assets/common-initial'
    initial_file = proof(adapter_dir / 'adapter_model.safetensors')
    if initial_file['sha256'] != INITIAL_SHA:
        raise ValueError('Common original zero-effect LoRA bytes mismatch')
    initial_config = proof(adapter_dir / 'adapter_config.json')
    if initial_config['sha256'] != INITIAL_CONFIG_SHA:
        raise ValueError('Common original adapter configuration bytes mismatch')
    import torch
    from safetensors.torch import load_file
    from trainer import tensor_digest
    if torch.cuda.is_initialized():
        raise ValueError('CPU asset reader initialized CUDA')
    tensors = load_file(initial_file['path'], device='cpu')
    if len(tensors) != 504 or not all(t.dtype == torch.float32 and torch.isfinite(t).all().item() for t in tensors.values()):
        raise ValueError('Common504 actual tensors not finiteFP32')
    zero_B = [t for name, t in tensors.items() if 'lora_B' in name]
    if len(zero_B) != 252 or any(torch.count_nonzero(t).item() for t in zero_B):
        raise ValueError('Common252B not zero')
    if tensor_digest(tensors) != INITIAL_TENSOR_SHA:
        raise ValueError('Actual shared LoRA tensor identity mismatch')
    initial = write_once(root / 'receipts/common-initial-adapter.json', dict(schema='opsd-signal-common-initial-v1',
        accepted=True, adapter_model=initial_file, payload_files=[initial_config, initial_file],
        adapter_tensor_sha256=INITIAL_TENSOR_SHA, tensors=504, zero_B_tensors=252, dtype='float32',
        fresh_adam_required=True, source_model_identity=model['sha256']))
    from cohort_v1 import freeze_from_files
    frozen = freeze_from_files(data['train'], data['validation'])
    cohorts = write_once(root / 'receipts/common-cohorts.json', frozen)
    if torch.cuda.is_initialized():
        raise ValueError('CPU preparation initialized CUDA')
    storage = dict(accepted=True, output_root=str(root.resolve()), remote_node=subprocess.check_output(['hostname'], text=True).strip(),
        mount=subprocess.check_output(['findmnt', '-J', '-T', str(root)], text=True),
        policy='Latest human explicitly authorizes HKUST node-local output; no Mac weights/checkpoints/raw outputs',
        retention_date='UNKNOWN; preserve originals and remote backups, no proactive node release',
        shared_storage_claim=False)
    storage_ref = write_once(root / 'receipts/storage-admission.json', storage)
    result = dict(schema='opsd-signal-assets-v1', accepted=True, source_model_manifest=model,
        runtime_manifest=manifest_ref, common_initial_adapter_manifest=initial, common_cohort_manifest=cohorts,
        train_identity=data['train'], validation_identity=data['validation'], storage=storage_ref,
        runtime_files=305, local_large_product_bytes=0, cuda_initialized=False)
    ready = write_once(root / 'receipts/assets-ready.json', result)
    fresh = write_once(root / 'receipts' / ('assets-check-' + str(time.time_ns()) + '.json'),
        dict(accepted=True, checked_unix=time.time(), immutable_assets=ready,
             current_available_bytes=shutil.disk_usage(root).free,
             source12_runtime305_data2_initial504_fresh_readback=True,
             cuda_initialized=False))
    return dict(accepted=True, immutable_assets=ready, fresh_readback=fresh)


if __name__ == '__main__':
    print(json.dumps(verify_assets(), sort_keys=True))
