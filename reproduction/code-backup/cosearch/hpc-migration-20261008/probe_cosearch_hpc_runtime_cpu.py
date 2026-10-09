"""Read-only native dependency imports; no models, retrieval queries, or CUDA."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import sys

BASE = Path('/data/user/jxu957/experiments/cosearch-continue300-x203-20261008')
OUT = BASE / 'migration-hf-20261009/runtime-cpu-probe.json'
SOURCE = BASE / 'source'

def main():
    env = dict(os.environ)
    for key in ('PYTHONPATH', 'PYTHONHOME', 'LD_PRELOAD'):
        env.pop(key, None)
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    candidate = BASE / 'runtime/retriever-python310-candidate'
    rows = []
    configs = [
        ('retriever310', candidate / 'bin/python3.10',
         SOURCE / 'env/lib/python3.10/site-packages', candidate / 'lib'),
        ('training311', SOURCE / 'author-native/python/cpython-3.11-linux-x86_64-gnu/bin/python3.11',
         SOURCE / 'author-native/env/lib/python3.11/site-packages',
         SOURCE / 'author-native/python/cpython-3.11-linux-x86_64-gnu/lib')]
    for role, binary, packages, lib in configs:
        if len(sys.argv) > 1 and role != sys.argv[1]:
            continue
        for module in ('numpy', 'torch', 'faiss', 'transformers', 'datasets',
                       'fastapi', 'uvicorn', 'pyarrow', 'ray', 'verl'):
            env['LD_LIBRARY_PATH'] = str(lib)
            code = ('import sys,json,importlib;sys.path.insert(0,' + repr(str(packages)) + ');'
                    'm=importlib.import_module(' + repr(module) + ');'
                    'print(json.dumps({"version":getattr(m,"__version__",None),'
                    '"file":getattr(m,"__file__",None),"python":sys.version}));'
                    'assert "torch" not in sys.modules or not sys.modules["torch"].cuda.is_initialized()')
            started = time.time()
            try:
                r = subprocess.run([str(binary), '-I', '-B', '-c', code], env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   universal_newlines=True, timeout=60)
                row = {'role': role, 'module': module, 'returncode': r.returncode,
                       'stdout': r.stdout[-2000:], 'stderr': r.stderr[-3000:]}
            except (OSError, subprocess.TimeoutExpired) as exc:
                row = {'role': role, 'module': module, 'error': repr(exc)}
            row['elapsed_seconds'] = time.time() - started
            rows.append(row)
            result = {'status': 'CPU_DEPENDENCY_DIAGNOSTICS_ONLY', 'epoch': time.time(),
                      'rows': rows, 'gpu_used': False, 'training_started': False,
                      'runtime_acceptance': False}
            destination = OUT if len(sys.argv) == 1 else OUT.with_name('runtime-cpu-probe-' + role + '.json')
            tmp = destination.with_suffix('.tmp')
            tmp.write_text(json.dumps(result, indent=2) + '\n')
            os.replace(tmp, destination)
    print(destination.read_text())

if __name__ == '__main__':
    main()
