"""Explicit human-authorized cohosting; never signal or alter foreign processes."""
import hashlib
import json
import os
from pathlib import Path

def authorize():
    path = Path(os.environ['GIGPO_COHOST_BINDING'])
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != os.environ['GIGPO_COHOST_BINDING_SHA256']:
        raise RuntimeError('Cohost binding SHA differs')
    value = json.loads(raw)['cohosting']
    if value['authorization'] != 'HUMAN_ALLOW_SAME_GPU_WITH_VIDEO_20261009' or value['preserve_foreign'] is not True:
        raise RuntimeError('Missing latest human same-GPU authorization')
    return value

def check_selected_gpus(selected, inventory, processes):
    import math
    authorize()
    if len(selected) != 8 or len(set(selected)) != 8:
        raise RuntimeError('Cohosting still requires eight distinct allocated GPUs')
    result = []
    for device in selected:
        matches = [row for index, row in inventory.items() if device in (index, row['uuid'])]
        if len(matches) != 1:
            raise RuntimeError('Unknown allocated GPU identity')
        row = matches[0]
        if not all(math.isfinite(row[key]) for key in ('memory_used','memory_total','utilization')):
            raise RuntimeError('Unknown GPU occupancy')
        if row['memory_total'] <= 0 or not 0 <= row['memory_used'] <= row['memory_total'] or not 0 <= row['utilization'] <= 100:
            raise RuntimeError('Invalid GPU occupancy')
        result.append(row)
    if len({row['uuid'] for row in result}) != 8:
        raise RuntimeError('Duplicate GPU UUIDs')
    return result
