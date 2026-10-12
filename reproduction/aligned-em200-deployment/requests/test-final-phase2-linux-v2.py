"""Actual target CPU fixture suite only, no official assets or training."""
import os,subprocess,json,hashlib
from pathlib import Path
r=Path('/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-baseline-aligned-em200-20261009');d=r/'tools/deployment/final-evidence-phase2-v1'
expected={'aligned_final_phase2.py': 'c7dc6c94896b292cd066481d76fa2eed071d46a3de83df7498d7f4d6595b77f3', 'aligned_final_evidence.py': '128ce70a0be27213b4bedb232b030e6bdf094279aa4d55dbc983d553c0bd8a8a', 'aligned_first5_cpu_evidence.py': '78d47e2653dd235f952c50b01cc1fc9baca525e11ecd17fa1a4e8202de9c8223', 'aligned_first5_math.py': '32c037c9e81bb9b0f3c30396f09623db57a3d763754a5d29b3c28e37964e3815', 'aligned-first5-math-provenance.json': 'a59a0503e2cbb66448e9ddf3b1ef3bc0e7cf7e0d716b68f273f3cfd664435e13', 'collect_hardware_evidence.py': 'be052023211b4ff49a60a3e3682c7a919ca84140badabc611092b128f2fefab2', 'test_aligned_final_phase2.py': 'e650eddcacd63ba30e5e9ea1bce9413f11646b0db5fb944c65a04afd510bf5e9', 'FINAL-EVIDENCE-PHASE2-CANDIDATE.md': '7b7942cd73c0b95a5c2d2ad154df493e4f92202ac1ef755dcd91b36a2cbdb70d', 'FINAL-EVIDENCE-PHASE2-INDEPENDENT-REVIEW.md': '8616241897d0c33d17588bf2914a7e978efda3c51a1f516c143eb241d9d0e7a0', 'test_aligned_final_phase2_linux_v2.py': 'f14dd2bee0fa8e42464112a25d4bff9c7dc9928558bf70684c1c63a4e2bb30a4', 'PHASE2-LINUX-FIXTURE-COMPATIBILITY-V2.md': '1965ccde8a7e7dc198382a7e70502d776b226684cadea1cf545f8886b6a92c20'}
assert r.resolve()==r and d.resolve()==d and not d.is_symlink()
assert subprocess.check_output(['git','-C',str(r/'repo'),'rev-parse','HEAD'],text=True).strip()=='080ebbc46b89d61567cd33e1a5e2f1b40afd8b84'
def closure():
 for n,h in expected.items():
  p=d/n
  if p.is_symlink() or p.resolve()!=p or hashlib.sha256(p.read_bytes()).hexdigest()!=h:raise RuntimeError('isolated candidate bytes changed '+n)
closure()
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(r/'repo'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
v=subprocess.run([str(r/'env/bin/python'),'-B','-m','pytest','-q','-p','no:cacheprovider',str(d/'test_aligned_final_phase2_linux_v2.py')],cwd=r/'repo',env=env,capture_output=True,text=True,timeout=90)
closure()
print(json.dumps(dict(exit=v.returncode,stdout=v.stdout,stderr=v.stderr,scope='ACTUAL_LINUX_TINY_CPU_FIXTURES_NOT_OFFICIAL3B_OR_FINAL200_ACCEPTANCE')))
raise SystemExit(v.returncode)
