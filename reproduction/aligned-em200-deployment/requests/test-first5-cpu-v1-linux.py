import os,subprocess,json,hashlib
from pathlib import Path
r=Path('/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-baseline-aligned-em200-20261009');d=r/'tools/deployment/first5-cpu-v1'
expected={'aligned_first5_cpu_evidence.py': '78d47e2653dd235f952c50b01cc1fc9baca525e11ecd17fa1a4e8202de9c8223', 'aligned_first5_math.py': '32c037c9e81bb9b0f3c30396f09623db57a3d763754a5d29b3c28e37964e3815', 'aligned-first5-math-provenance.json': 'a59a0503e2cbb66448e9ddf3b1ef3bc0e7cf7e0d716b68f273f3cfd664435e13', 'test_aligned_first5_cpu_evidence.py': '74114d12e8c580d402075fc7beba84a8262e9cdf8868de11981d8b75f752910d', 'FIRST5-CPU-CANDIDATE.md': '0f3eabf0305ff9f429625a0471d47a1c91165b22f199a24366c8125b4c0965e5'}
for n,h in expected.items():
 p=d/n
 if p.is_symlink() or p.resolve()!=p or hashlib.sha256(p.read_bytes()).hexdigest()!=h:raise RuntimeError('candidate file changed '+n)
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(r/'repo'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
v=subprocess.run([str(r/'env/bin/python'),'-B','-m','pytest','-q','-p','no:cacheprovider',str(d/'test_aligned_first5_cpu_evidence.py')],cwd=r/'repo',env=env,capture_output=True,text=True,timeout=90)
print(json.dumps(dict(exit=v.returncode,stdout=v.stdout,stderr=v.stderr,scope='ACTUAL_LINUX_CPU_FIXTURE_TOOL_TESTS_NOT_OFFICIAL3B_OR_FIRST5_ACCEPTANCE')))
raise SystemExit(v.returncode)
