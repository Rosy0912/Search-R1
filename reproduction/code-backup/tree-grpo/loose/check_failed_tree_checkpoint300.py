import os,json,hashlib,subprocess,time,re,socket
from pathlib import Path
root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo');run=root/'runs/tree-formal-2971';job=os.environ['SLURM_JOB_ID'];allocation=subprocess.check_output(['scontrol','show','job',job],text=True)
assert socket.gethostname().split('.')[0]=='h100-9' and 'JobState=RUNNING' in allocation and 'TresPerNode=gres:gpu:0' in allocation and not os.environ.get('SLURM_JOB_GPUS')
assert json.loads((run/'run-result.json').read_text())['status']=='FAILED'
checker=root/'tools/check_real_checkpoint.py';assert hashlib.sha256(checker.read_bytes()).hexdigest()=='b3075d7b9346bb7682e0ea05962de0624c117b354590b728ed8a18b7e8ba3c97'
(run/'failed-checkpoint300-cpu-allocation.txt').write_text(allocation);start=time.time();code=1
try:
 subprocess.run([str(root/'env/bin/python'),str(checker),'--base',str(root/'assets-root/assets/qwen2.5-3b'),'--checkpoint',str(run/'output/checkpoints/tree_grpo-qwen2.5-3b/actor/global_step_300'),'--output',str(run/'hf300-after-failure.json')],check=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'8','HF_HUB_OFFLINE':'1'});code=0
finally:
 with (run/'failed-checkpoint300-cpu-cost.json').open('x') as f:json.dump(dict(job_id=job,parent_job_id='2971',scope='Only saved HF300 full CPU reload after failed old-protocol job; not500 or corrected reward acceptance',allocated_gpus=0,allocated_cpus=int(re.search(r'NumCPUs=(\d+)',allocation).group(1)),start_epoch=start,end_epoch=time.time(),exit_code=code),f,indent=2)
