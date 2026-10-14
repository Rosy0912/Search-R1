"""Pinned, private CPU probe dependencies; never change reused model runtime."""
import hashlib
import json
from pathlib import Path

ROOT=Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-20261014')
LOCK=Path(__file__).with_name('probe_cpu_wheels.json')


def lock():return json.loads(LOCK.read_text())


def root():
    fingerprint=hashlib.sha256(json.dumps(lock(),sort_keys=True).encode()).hexdigest()
    return ROOT/'runtime'/('probe-cpu-'+fingerprint[:16])


def verify():
    target=root();manifest=json.loads((target/'manifest.json').read_text())
    if manifest['wheels']!=lock() or manifest['status']!='CPU_PROBE_WHEELS_FULL_SHA_PASS':
        raise ValueError('Required CPU probe dependency binding differs')
    for n,h in manifest['files'].items():
        p=target/'site-packages'/n
        if not p.is_file() or p.is_symlink() or hashlib.sha256(p.read_bytes()).hexdigest()!=h:
            raise ValueError('Private CPU probe dependency changed: '+n)
    return target/'site-packages',manifest
