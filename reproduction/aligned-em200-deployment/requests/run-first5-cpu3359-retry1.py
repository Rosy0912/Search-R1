"""Run the existing first-five CPU extractor once into a fresh evidence directory."""
import os, json, subprocess
from pathlib import Path

ROOT = Path('/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v11-20261009')
OUT = ROOT / 'evidence/first5-cpu3359-20261009-2050'
assert os.uname().nodename == 'h100-11'
assert ROOT.is_dir() and ROOT.resolve() == ROOT
assert not OUT.exists()
env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(ROOT/'repo'),
           HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
           OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
env.pop('LD_LIBRARY_PATH', None); env.pop('PYTHONOPTIMIZE', None)
p = subprocess.run([str(ROOT/'env/bin/python'), '-B',
                    str(ROOT/'tools/deployment/first5-node11-3359/aligned_first5_cpu_evidence.py'),
                    '--output', str(OUT)], cwd=ROOT/'repo', env=env,
                   capture_output=True, text=True, timeout=1500)
print(json.dumps({'exit': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr,
                  'output': str(OUT)}))
if (OUT/'cpu-evidence.json').exists():
    print((OUT/'cpu-evidence.json').read_text())
raise SystemExit(p.returncode)
