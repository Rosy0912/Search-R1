"""Check exact migrated package overlays without model loading or CUDA execution."""
import json
import os
from pathlib import Path
import subprocess
import time

BASE = Path('/data/user/jxu957/experiments/cosearch-continue300-x203-20261008')
SOURCE = BASE / 'source'
OUT = BASE / 'migration-hf-20261009/runtime-scoped-imports.json'

def main():
    env = dict(os.environ)
    for key in ('PYTHONHOME', 'PYTHONPATH', 'LD_PRELOAD'):
        env.pop(key, None)
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    candidate = BASE / 'runtime/retriever-python310-candidate'
    python311 = SOURCE / 'author-native/python/cpython-3.11.17-linux-x86_64-gnu'
    cases = [
        ('retriever_gpu_faiss_import', candidate / 'bin/python3.10', candidate / 'lib',
         [SOURCE / 'retriever-gpu-lib', SOURCE / 'env/lib/python3.10/site-packages', SOURCE / 'repo'],
         'import torch,faiss;assert hasattr(faiss,"StandardGpuResources");'
         'assert faiss.__file__.startswith(' + repr(str(SOURCE / 'retriever-gpu-lib')) + ');'
         'detail={"faiss_file":faiss.__file__,"faiss_version":faiss.__version__,"torch":torch.__version__}'),
        ('training_verl_import', python311 / 'bin/python3.11', python311 / 'lib',
         [SOURCE / 'author-native/repo', SOURCE / 'author-native/repo/verl', SOURCE / 'author-native/env/lib/python3.11/site-packages'],
         'import torch,verl;detail={"verl_file":verl.__file__,"torch":torch.__version__}'),
        ('training_vllm_import', python311 / 'bin/python3.11', python311 / 'lib',
         [SOURCE / 'author-native/env/lib/python3.11/site-packages'],
         'import torch,vllm;detail={"vllm_file":vllm.__file__,"vllm_version":vllm.__version__}'),
        ('training_flash_attn_import', python311 / 'bin/python3.11', python311 / 'lib',
         [SOURCE / 'author-native/env/lib/python3.11/site-packages'],
         'import torch,flash_attn;detail={"file":flash_attn.__file__,"version":flash_attn.__version__}')]
    rows = []
    for name, binary, lib, paths, body in cases:
        assert binary.is_file(), str(binary)
        env['LD_LIBRARY_PATH'] = str(lib)
        code = ('import sys,json;sys.path[:0]=' + repr([str(p) for p in paths]) + ';' + body +
                ';assert not torch.cuda.is_initialized();print(json.dumps(detail))')
        start = time.time()
        try:
            r = subprocess.run([str(binary), '-I', '-B', '-c', code], env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               universal_newlines=True, timeout=90)
            row = {'name': name, 'returncode': r.returncode, 'stdout': r.stdout[-3000:], 'stderr': r.stderr[-5000:]}
        except subprocess.TimeoutExpired as exc:
            row = {'name': name, 'error': repr(exc)}
        row['seconds'] = time.time() - start
        rows.append(row)
        tmp = OUT.with_suffix('.tmp')
        tmp.write_text(json.dumps({'epoch': time.time(), 'rows': rows,
            'status': 'SCOPED_IMPORT_DIAGNOSTICS_ONLY', 'cuda_initialized': False,
            'runtime_acceptance': False, 'training_started': False}, indent=2) + '\n')
        os.replace(tmp, OUT)

if __name__ == '__main__':
    main()
