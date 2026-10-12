import os,subprocess,json,hashlib
from pathlib import Path
r=Path('/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-baseline-aligned-em200-20261009');d=r/'tools/deployment/first5-cpu-v1'
for n,h in {'publish_first5_release.py': '6f323917cf04cccd6a6d71528d1ff336d6638250159e1d406b80ced865d3006a', 'test_publish_first5_release.py': '7be8ba579e030d19eff59240991faced23840b383a69b850575fa9aeef82131a', 'FIRST5-RELEASE-CANDIDATE.md': 'cfbb6a6fce3055154022bd38c5bff8d9ae94688c75a7ad81f7a7fcc65480aebc', 'FIRST5-RELEASE-CANDIDATE-HASHES.json': '1c3c5d88f1ee8f057edd4632edfeba9f43dded2a5c80e41f90c8ebbd620aad4e', 'aligned_first5_cpu_evidence.py': '78d47e2653dd235f952c50b01cc1fc9baca525e11ecd17fa1a4e8202de9c8223', 'aligned_first5_math.py': '32c037c9e81bb9b0f3c30396f09623db57a3d763754a5d29b3c28e37964e3815', 'aligned-first5-math-provenance.json': 'a59a0503e2cbb66448e9ddf3b1ef3bc0e7cf7e0d716b68f273f3cfd664435e13'}.items():
 p=d/n
 if p.is_symlink() or p.resolve()!=p or hashlib.sha256(p.read_bytes()).hexdigest()!=h:raise RuntimeError('fixed candidate changed '+n)
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(r/'repo'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
v=subprocess.run([str(r/'env/bin/python'),'-B','-m','pytest','-q','-p','no:cacheprovider',str(d/'test_publish_first5_release.py')],cwd=r/'repo',env=env,capture_output=True,text=True,timeout=90)
print(json.dumps(dict(exit=v.returncode,stdout=v.stdout,stderr=v.stderr,scope='ACTUAL_LINUX_CPU_BOUNDARY_TESTS_NOT_REAL_FIRST5_RELEASE_OR_ACCEPTANCE')))
raise SystemExit(v.returncode)
