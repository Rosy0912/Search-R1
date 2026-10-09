import os,json,subprocess
from pathlib import Path
r=Path('/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v11-20261009')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(r/'repo'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1');env.pop('LD_LIBRARY_PATH',None)
tests=['test_aligned_complete5_resume.py','test_complete5_gate_lineage.py','test_first5_empty_bound_diagnostic.py','test_complete5_startup_claim.py','test_baseline_aligned_first5_gate.py','test_fresh200_gate_skip.py']
p=subprocess.run([str(r/'env/bin/python'),'-B','-m','pytest','-q']+['tests/baselines/'+t for t in tests],cwd=r/'repo',env=env,capture_output=True,text=True,timeout=90)
print(json.dumps({'exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'scope':'ACTUAL_LINUX_TARGETED_CPU_TESTS_NOT_NATIVE_RESTORE_NOT_TRAINING_PASS'}));raise SystemExit(p.returncode)
