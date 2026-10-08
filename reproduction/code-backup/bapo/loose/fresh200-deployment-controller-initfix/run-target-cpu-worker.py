from pathlib import Path
import os,subprocess,json,sys
R=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-controller-initfix-20261006'); S='60f566b3ddabc88ce9de56ead5ad30a3d6795dd2'
assert os.uname().nodename=='h100-5'and R.resolve()==R
assert subprocess.check_output(['git','-C',str(R/'repo'),'rev-parse','HEAD'],text=True).strip()==S
assert not subprocess.check_output(['git','-C',str(R/'repo'),'status','--porcelain'],text=True).strip()
env=dict(os.environ,PYTHONPATH=str(R/'repo'),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
for name in ['capture-runtime-executable.py','run-target-cpu.py']:
 subprocess.run([str(R/'env/bin/python'),'-B',str(R/'tools'/name)],env=env,cwd=str(R/'repo'),check=True)
