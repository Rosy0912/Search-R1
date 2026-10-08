"""Owned same-node CPU terminal retention/fullHF2 reload; never resume training."""
import hashlib,json,os,re,socket,subprocess,sys,time
from pathlib import Path
root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo');parent=sys.argv[1];assert parent.isdigit()
run=root/'runs'/f'tree-fixed-two-{parent}';job=os.environ['SLURM_JOB_ID']
assert socket.gethostname().split('.')[0]=='h100-9' and not os.environ.get('SLURM_JOB_GPUS')
allocation=subprocess.check_output(['scontrol','show','job',job],text=True)
assert 'JobState=RUNNING' in allocation and 'TresPerNode=gres:gpu:0' in allocation
evidence=root/'runs'/f'tree-fixed-two-cpu-{parent}-{job}';evidence.mkdir()
(evidence/'allocation.txt').write_text(allocation)
terminal=subprocess.run(['scontrol','show','job',parent],text=True,capture_output=True)
(evidence/'slurm-parent-terminal.txt').write_text(terminal.stdout+terminal.stderr)
if run.exists():(run/'slurm-parent-terminal.txt').write_text(terminal.stdout+terminal.stderr)
result=json.loads((run/'run-result.json').read_text()) if (run/'run-result.json').exists() else {'status':'NO_SUPERVISOR_RESULT'};start=time.time();code=1
record=dict(parent_job_id=parent,cpu_job_id=job,parent_state=result['status'],allocated_gpus=0,allocated_cpus=int(re.search(r'NumCPUs=(\d+)',allocation).group(1)),parent_terminal_capture_exit=terminal.returncode,source='296217a4b676a26a92fc729f5d9882b785dbfa96',scope='Parent terminal retention and HF2 CPU reload only; not2 independent acceptance or500')
try:
 assert terminal.returncode==0,'Parent terminal unknown'
 if result['status']!='ACTUAL_STAGE_COMPLETED_UNREVIEWED':
  record.update(status='PARENT_FAILED_HF_ACCEPTANCE_NOT_ATTEMPTED');code=0
 else:
  assert 'JobState=COMPLETED' in terminal.stdout and 'ExitCode=0:0' in terminal.stdout
  assert result['code_commit']==record['source'] and result['cleanup_residual']=={}
  checker=root/'tools/check_real_checkpoint.py'
  assert hashlib.sha256(checker.read_bytes()).hexdigest()=='b3075d7b9346bb7682e0ea05962de0624c117b354590b728ed8a18b7e8ba3c97'
  subprocess.run([str(root/'env/bin/python'),str(checker),'--base',str(root/'assets-root/assets/qwen2.5-3b'),'--checkpoint',str(run/'output/checkpoints/tree_grpo-qwen2.5-3b/actor/global_step_2'),'--output',str(evidence/'hf2-full-reload.json')],check=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'8','HF_HUB_OFFLINE':'1'})
  record['status']='HF2_FULL_RELOAD_COMPLETE_PENDING_REVIEW';code=0
finally:
 record.update(start_epoch=start,end_epoch=time.time(),exit_code=code)
 with (evidence/'result.json').open('x') as f:json.dump(record,f,indent=2)
if code:raise SystemExit(code)
