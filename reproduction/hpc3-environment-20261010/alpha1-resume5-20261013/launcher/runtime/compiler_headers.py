"""Bind the actual HPC3 Python headers and verify a compiled native ABI."""
import hashlib
import json
import pathlib
import sys
import sysconfig

BASE = pathlib.Path('/data/user/jxu957/experiments/topd-20261008/runtime/assets/hpc3-compiler-headers')


def binding():
    p = BASE / 'inventory.json'
    d = json.loads(p.read_text())
    return {'base': str(BASE), 'inventory_sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
            'cpath': d['include'], 'provider': 'HPC3_CONDA_PY310_NATIVE_ABI'}


def verify(config):
    if config != binding() or sys.version_info[:2] != (3, 10):
        raise ValueError('HPC3 header/Python identity changed')
    d = json.loads((BASE / 'inventory.json').read_text())
    include = pathlib.Path(sysconfig.get_path('include')).resolve()
    if str(include) != d['include']:
        raise ValueError('Actual Python include path differs')
    actual = {str(p.relative_to(include)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in include.rglob('*') if p.is_file()}
    if actual != d['files']:
        raise ValueError('Actual compiler header inventory changed')
    import torch
    import flash_attn_3_cuda
    return {'scope': 'HPC3_HEADERS_AND_COMPILED_NATIVE_ABI_PASS_NO_GPU_EXECUTION',
            'python': sys.version.split()[0], 'regular_files': len(actual),
            'native_extension': flash_attn_3_cuda.__file__, 'cpath': str(include)}
