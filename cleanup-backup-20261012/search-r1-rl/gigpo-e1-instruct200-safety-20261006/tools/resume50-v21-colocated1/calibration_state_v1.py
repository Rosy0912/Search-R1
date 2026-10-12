"""Typed complete-state digests and a zero-Adam pause check; never restores state."""
import hashlib
import re
import struct

import numpy as np
import torch


BF16_ACCUMULATION_SCOPE = 'BF16_REDUCTION_DISABLED_BEFORE_BASE_WORKER_INITIALIZATION_V1'


def validate_bf16_accumulation_identity(row, *, role):
    """Require the actual process readback; never set or repair a live flag."""
    policy = row['bf16_accumulation']
    if (policy['scope'] != BF16_ACCUMULATION_SCOPE
            or policy['allow_bf16_reduced_precision_reduction'] is not False
            or policy['configured_before_base_worker_initialization'] is not True
            or policy['constructor_pid'] != row['pid']
            or policy['worker_role'] != role):
        raise ValueError('BF16 accumulation process policy differs; no live flag repair is permitted')
    return policy


def state_digest(value):
    """Hash actual values, types, shapes and mapping keys without pickle/storage IDs."""
    digest = hashlib.sha256()

    def blob(tag, content):
        digest.update(tag)
        digest.update(struct.pack('!Q', len(content)))
        digest.update(content)

    def visit(item):
        if item is None:
            blob(b'N', b'')
        elif isinstance(item, np.generic):
            blob(b'G', item.dtype.str.encode())
            blob(b'D', item.tobytes())
        elif isinstance(item, bool):
            blob(b'B', b'1' if item else b'0')
        elif isinstance(item, int):
            blob(b'I', str(item).encode())
        elif isinstance(item, float):
            blob(b'F', struct.pack('!d', item))
        elif isinstance(item, str):
            blob(b'S', item.encode('utf-8'))
        elif isinstance(item, bytes):
            blob(b'Y', item)
        elif isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            blob(b'T', str((tuple(tensor.shape), str(tensor.dtype))).encode())
            view = memoryview(tensor.reshape(-1).view(torch.uint8).numpy()).cast('B')
            digest.update(struct.pack('!Q', len(view)))
            for start in range(0, len(view), 8 * 1024 * 1024):
                digest.update(view[start:start + 8 * 1024 * 1024])
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            blob(b'A', str((array.shape, array.dtype.str)).encode())
            if array.dtype.hasobject:
                visit(array.tolist())
            else:
                blob(b'D', array.tobytes())
        elif isinstance(item, dict):
            blob(b'M', str(len(item)).encode())
            # Canonical ordering includes key type. Values stay fully typed.
            for key in sorted(item, key=state_digest):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            blob(b'L' if isinstance(item, list) else b'U', str(len(item)).encode())
            for entry in item:
                visit(entry)
        else:
            raise TypeError('Unsupported complete state value: ' + repr(type(item)))

    visit(value)
    return digest.hexdigest()


WORKER_KEYS = ('rank', 'pid', 'start_ticks', 'scheduler_step', 'learning_rates',
               'model_shards_sha256', 'optimizer_tensors_sha256', 'rng_tensors_sha256',
               'local_shards_sha256', 'buffers_sha256', 'complete_state_sha256')


def validate_pause_workers(workers):
    if len(workers) != 8 or {row.get('rank') for row in workers} != set(range(8)):
        raise ValueError('Missing actual eight-worker pause identity; zero Adam')
    if len({row.get('pid') for row in workers}) != 8:
        raise ValueError('Duplicate worker process identity; zero Adam')
    for row in workers:
        if any(key not in row for key in WORKER_KEYS) or not row['start_ticks'] or row['scheduler_step'] != 0:
            raise ValueError('Incomplete actual post-sampling worker pause state; zero Adam')
        if not isinstance(row['pid'], int) or row['pid'] <= 0 or not str(row['start_ticks']).isdecimal():
            raise ValueError('Invalid actual worker PID/start; zero Adam')
        for key in WORKER_KEYS:
            if key.endswith('_sha256') and not re.fullmatch(r'[0-9a-f]{64}', row[key]):
                raise ValueError('Invalid actual complete worker state SHA: ' + key + '; zero Adam')
    return sorted(workers, key=lambda row: row['rank'])


def assert_complete_pause_unchanged(before_workers, after_workers, before_driver, after_driver):
    """No mutation/restart is allowed between sampled evidence and first Adam."""
    before = validate_pause_workers(before_workers)
    after = validate_pause_workers(after_workers)
    if before != after:
        raise ValueError('Worker PID/start/model/internal/Adam/scheduler/RNG changed during review; zero Adam')
    if state_digest(before_driver) != state_digest(after_driver):
        raise ValueError('Driver PID/start/RNG/data/GiGPO/batch changed during review; zero Adam')
    return dict(status='UNCHANGED_BEFORE_FIRST_ADAM', actual_adam_writes=0,
                worker_pause_sha256=state_digest(before), driver_complete_state_sha256=state_digest(before_driver))
