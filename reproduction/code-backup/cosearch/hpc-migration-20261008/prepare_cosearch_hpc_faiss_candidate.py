"""Test a separate ELF alignment compatibility candidate; frozen FAISS stays intact."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import time

BASE = Path('/data/user/jxu957/experiments/cosearch-continue300-x203-20261008')
SOURCE = BASE / 'source/retriever-gpu-lib'
ROOT = BASE / 'runtime/gpu-faiss-alignment-candidate'
NAME = '_swigfaiss.cpython-310-x86_64-linux-gnu.so'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    assert not ROOT.exists(), 'Inspect previous candidate before retry'
    src = SOURCE / 'faiss' / NAME
    original = src.read_bytes()
    assert original[:6] == b'\x7fELF\x02\x01'
    assert hashlib.sha256(original).hexdigest() == '1992d0aa5d8c2ac443a03d1e8728a2c783ed00032aaf0778113001044dd7ae45'
    phoff = struct.unpack_from('<Q', original, 32)[0]
    entsize, count = struct.unpack_from('<HH', original, 54)
    assert entsize == 56 and count == 12
    patched = bytearray(original)
    edits = []
    for i in range(count):
        pos = phoff + i * entsize
        kind, flags, offset, virtual, physical, filesz, memsz, align = struct.unpack_from('<IIQQQQQQ', original, pos)
        if kind == 1 and align > 4096 and (virtual - offset) % align:
            assert (virtual - offset) % 4096 == 0
            struct.pack_into('<Q', patched, pos + 48, 4096)
            edits.append({'program_header': i, 'byte_offset': pos + 48, 'old_align': align, 'new_align': 4096})
    assert len(edits) == 1, edits
    allowed = set(range(edits[0]['byte_offset'], edits[0]['byte_offset'] + 8))
    differences = [i for i, (a, b) in enumerate(zip(original, patched)) if a != b]
    assert differences and set(differences).issubset(allowed) and len(original) == len(patched)
    ROOT.mkdir(parents=True)
    shutil.copytree(str(SOURCE / 'faiss'), str(ROOT / 'faiss'),
                    ignore=shutil.ignore_patterns('__pycache__'))
    # Wheel-relative native dependencies remain immutable source files.
    (ROOT / 'faiss_gpu_cu12.libs').symlink_to(SOURCE / 'faiss_gpu_cu12.libs', target_is_directory=True)
    dst = ROOT / 'faiss' / NAME
    assert sha(dst) == sha(src)
    dst.write_bytes(patched)
    assert sha(src) == hashlib.sha256(original).hexdigest()
    runtime = BASE / 'runtime/retriever-python310-candidate'
    env = dict(os.environ)
    for k in ('PYTHONPATH', 'PYTHONHOME', 'LD_PRELOAD'):
        env.pop(k, None)
    env.update(LD_LIBRARY_PATH=str(runtime / 'lib'), CUDA_VISIBLE_DEVICES='',
               PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    paths = [str(ROOT), str(BASE / 'source/env/lib/python3.10/site-packages')]
    code = ('import sys,json;sys.path[:0]=' + repr(paths) + ';import torch,faiss;'
            'assert hasattr(faiss,"StandardGpuResources");assert not torch.cuda.is_initialized();'
            'print(json.dumps({"faiss_file":faiss.__file__,"version":faiss.__version__}))')
    r = subprocess.run([str(runtime / 'bin/python3.10'), '-I', '-B', '-c', code], env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=60)
    result = {'epoch': time.time(), 'source_sha256': sha(src), 'candidate_sha256': sha(dst),
              'header_edits': edits, 'changed_byte_offsets': differences, 'returncode': r.returncode,
              'stdout': r.stdout, 'stderr': r.stderr[-5000:], 'training_started': False,
              'runtime_acceptance': False, 'gpu_execution_parity': 'PENDING',
              'status': 'SEPARATE_ALIGNMENT_CANDIDATE_IMPORT_PASS' if r.returncode == 0 else 'CANDIDATE_IMPORT_FAILED'}
    (ROOT / 'candidate-result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))

if __name__ == '__main__':
    main()
