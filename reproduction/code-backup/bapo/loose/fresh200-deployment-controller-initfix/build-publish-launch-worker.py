from pathlib import Path
import os,subprocess,hashlib,json
R=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-controller-initfix-20261006')
assert os.uname().nodename=='h100-5'and R.resolve()==R
p=R/'publish-launch-approval-reviewed.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=='b65c377a567d16bc51d537372d73a42c186fc5c7b6bd91dfe09cd78074e2c379'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(R/'repo'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1');env.pop('PYTHONOPTIMIZE',None)
cmd=[str(R/'env/bin/python'),'-B',str(p),'--expected-human-goal-sha','a3c57fdc1a2a74f226dd194d1e7e7a4325e0721c716769fce2647f26ff52be16']
with(R/'logs/conditional-launch-build.actual.log').open('x')as f:subprocess.run(cmd,env=env,cwd=str(R/'repo'),stdout=f,stderr=subprocess.STDOUT,check=True)
with(R/'logs/conditional-launch-publish.actual.log').open('x')as f:subprocess.run(cmd+['--publish'],env=env,cwd=str(R/'repo'),stdout=f,stderr=subprocess.STDOUT,check=True)
print(json.dumps({'scope':'ACTUAL_BOUND_FIRST5_ONLY_LAUNCH_PUBLICATION_NOT_NATIVE','approval_sha256':hashlib.sha256((R/'logs/launch-approval.json').read_bytes()).hexdigest(),'publish_log':(R/'logs/conditional-launch-publish.actual.log').read_text()}))
