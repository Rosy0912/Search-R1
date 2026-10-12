"""CPU evidence collector after an own successful Tree stage; no acceptance decision."""
import argparse, hashlib, json, os, re, socket, subprocess, sys, time
from pathlib import Path
ROOT=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--stage',choices=['five','formal'],required=True);a=p.parse_args()
assert a.job.isdigit();run=ROOT/'runs'/f'tree-{a.stage}-{a.job}'
job=os.environ.get('SLURM_JOB_ID',''); assert job.isdigit()
allocation=subprocess.check_output(['scontrol','show','job',job],text=True)
assert socket.gethostname().split('.')[0]=='h100-9' and 'NodeList=h100-9' in allocation and 'JobState=RUNNING' in allocation
assert 'TresPerNode=gres:gpu:0' in allocation and not os.environ.get('SLURM_JOB_GPUS')
allocated_cpus=int(re.search(r'\bNumCPUs=(\d+)',allocation).group(1))
with (run/'cpu-check-allocation.txt').open('x') as stream:stream.write(allocation)
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',HF_HUB_OFFLINE='1')
expected={'validate_tree_stage.py':'2c2590a7830e2433d79122d329f488607f1e5697ca110348945383aff2dd0fb9','check_real_checkpoint.py':'b3075d7b9346bb7682e0ea05962de0624c117b354590b728ed8a18b7e8ba3c97'}
for n,v in expected.items():assert hashlib.sha256((ROOT/'tools'/n).read_bytes()).hexdigest()==v
result=json.loads((run/'run-result.json').read_text());assert result['status']=='ACTUAL_STAGE_COMPLETED_UNREVIEWED' and result['training_body_exit_code']==0 and result['entry_config_verified'] is True
assert all(v=='Z' for v in result['cleanup_residual'].values())
start=time.time();code=1
try:
 subprocess.run([sys.executable,str(ROOT/'tools/validate_tree_stage.py'),'--run',str(run),'--stage',a.stage,'--base',str(ROOT/'assets-root/assets/qwen2.5-3b'),'--output',str(run/'stage-evidence.json'),'--require-tree-audit'],check=True)
 code=0
finally:
 with (run/'cpu-check-cost.json').open('x') as stream:json.dump(dict(job_id=os.environ['SLURM_JOB_ID'],training_job_id=a.job,allocated_gpus=0,allocated_cpus=allocated_cpus,requested_cpus_per_task=int(os.environ['SLURM_CPUS_PER_TASK']),start_epoch=start,end_epoch=time.time(),exit_code=code,scope='CPU validator including actual full HF weight reload; source pin verification precedes cost start'),stream,indent=2)
