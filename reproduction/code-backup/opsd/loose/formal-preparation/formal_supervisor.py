"""Hold parent GPU leases across real capacity gates and trainer child."""
import hashlib,json,os,subprocess,time
from pathlib import Path
from omegaconf import OmegaConf
from search_r1.baselines.preflight import validate_inherited_gpu_leases
from search_r1.baselines.holdout import prepare_common_holdout
from search_r1.baselines.controller import validate_config
root=Path(os.environ['SEARCH_R1_BASE']); out=Path(os.environ['OPSD_EVIDENCE'])
actor_allocator=os.environ.get('OPSD_ACTOR_ALLOCATOR_CONF','')
assert actor_allocator == '', 'Experimental allocator tuning is capacity-only; production uses default native allocator'
actor_environment=dict(os.environ)
if actor_allocator: actor_environment['PYTORCH_CUDA_ALLOC_CONF']=actor_allocator
(out/'actor-runtime-environment.json').write_text(json.dumps(dict(job_id=os.environ['SLURM_JOB_ID'],source_commit=os.environ['OPSD_EXPECTED_COMMIT'],driver_environment_only=True,actor_actual_allocator_verification_required=True,PYTORCH_CUDA_ALLOC_CONF=actor_environment.get('PYTORCH_CUDA_ALLOC_CONF','')),indent=2))
lease_fds=tuple(int(r['fd']) for r in json.loads(os.environ['SEARCH_R1_GPU_LEASES']))
# Wait for the completed health check to become idle, preserving the double15% guard.
for attempt in range(30):
    try: validate_inherited_gpu_leases(8);break
    except RuntimeError:
        if attempt==29: raise
        time.sleep(2)
subprocess.run(['python','-u',str(root/'tools/runtime/semantic_device_gate.py')],check=True,pass_fds=lease_fds)
# The numerical child exited; allow NVML sampling to reflect cleanup before
# the next child repeats the unchanged allocation occupancy checks.
for attempt in range(30):
    try: validate_inherited_gpu_leases(8);break
    except RuntimeError:
        if attempt==29: raise
        time.sleep(2)
subprocess.run(['python','-u',str(root/'tools/runtime/gpu_loss_checkpoint_equivalence.py'),'--scope','production-eight'],check=True,pass_fds=lease_fds)
# Reuse the same independently reviewed exact-UUID, per-candidate double15%
# and all-eight-rank capacity supervisor in this fresh owned allocation.
subprocess.run(['python','-u',str(root/'tools/runtime/capacity_supervisor.py')],check=True,pass_fds=lease_fds)
capacity=json.loads((out/'micro-capacity.json').read_text())
assert capacity['passed'] is True and capacity['job_id']==os.environ['SLURM_JOB_ID']
assert capacity['actual_training_iterations']==0 and capacity['vllm_coexistence_tested'] is False
capacity_micro=capacity['max_safe_local_micro']
assert type(capacity_micro) is int and capacity_micro >= 2, 'Formal protocol requires accepted local micro2; do not silently downsize'
local_micro=2
assert type(local_micro) is int and local_micro in (1,2,4,8)
# Keep margin; actual rollout/FSDP/optimizer residency is verified in the fresh trainer.
os.environ['OPSD_GLOBAL_MICROBATCH']=str(local_micro*8)
args=[f'actor_rollout_ref.actor.ppo_micro_batch_size={local_micro*8}',
'baselines.distillation.teacher_device=cuda','baselines.distillation.max_student_logits_gib=16.0',
'baselines.distillation.loss_checkpointing=true',
'baselines.local_runtime_only=true','baselines.local_runtime_host=h100-10',
'retriever.url=http://127.0.0.1:18084/retrieve','trainer.total_training_steps=501',
'data.val_batch_size=64','trainer.smoke_only=false','trainer.val_before_train=true','trainer.test_freq=50','trainer.save_freq=100','trainer.acceptance_step=5',
 'trainer.experiment_name=opsd-gpu-formal500','actor_rollout_ref.rollout.log_prob_micro_batch_size=64',
 'actor_rollout_ref.ref.log_prob_micro_batch_size=64']
base=['bash','reproduction/baselines/run.sh','opsd']
resolved=subprocess.check_output(base+args+['--cfg','job','--resolve'],text=True,pass_fds=lease_fds)
(out/'resolved.yaml').write_text(resolved)
config=OmegaConf.create(resolved)
assert config.trainer.total_training_steps==501 and config.trainer.smoke_only is False
assert config.trainer.test_freq==50 and config.trainer.save_freq==100 and config.trainer.acceptance_step==5
assert config.actor_rollout_ref.model.path==os.environ['OPSD_TEACHER'], 'Fresh initialbase required; no pilot checkpoint resume'
assert config.actor_rollout_ref.actor.optim.lr_warmup_steps_ratio==.285
assert int(config.trainer.total_training_steps*config.actor_rollout_ref.actor.optim.lr_warmup_steps_ratio)==142
assert config.data.train_batch_size==512 and config.actor_rollout_ref.actor.ppo_mini_batch_size==256
assert config.actor_rollout_ref.actor.ppo_micro_batch_size==16 and config.data.val_batch_size==64
assert config.data.val_data_num is None and config.data.train_data_num is None
manifest=prepare_common_holdout(config);validate_config(config)
assert manifest['content_identity']['sha256']=='fac19c74fe28df2016bdff12f74f035fe272bed0d65d37e9740071e674195524'
import pandas as pd
expected_final=dict(pd.read_parquet(config.data.final_val_files,columns=['data_source'])['data_source'].value_counts())
expected_final={name:int(count) for name,count in expected_final.items()}
assert sum(expected_final.values())==51713 and len(expected_final)==7
(out/'expected-final-counts.json').write_text(json.dumps(expected_final,sort_keys=True,indent=2))
text=OmegaConf.to_yaml(OmegaConf.create(OmegaConf.to_container(config,resolve=True)))
(out/'reviewed-entry-config.yaml').write_text(text)
os.environ.update(SEARCH_R1_BASELINE_AUDIT_DIR=str(out/'audit'),SEARCH_R1_BASELINE_RUN_ID=os.environ['SLURM_JOB_ID'],
 SEARCH_R1_BASELINE_COMMIT=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
 SEARCH_R1_BASELINE_CONFIG_SHA256=hashlib.sha256(text.encode()).hexdigest(),
 SEARCH_R1_BASELINE_DATA_IDENTITY=manifest['content_identity']['sha256'])
with (out/'environment-freeze.txt').open('w') as log:
 subprocess.run([str(root/'tools/uv-env/bin/uv'),'pip','freeze','--python',str(root/'env/bin/python')],check=True,stdout=log)
for attempt in range(30):
    try: validate_inherited_gpu_leases(8);break
    except RuntimeError:
        if attempt==29: raise
        time.sleep(2)
with (out.parent/'training.log').open('w') as log:
 result=subprocess.run(base+args,stdout=log,stderr=subprocess.STDOUT,pass_fds=lease_fds,env=dict(os.environ,PYTORCH_CUDA_ALLOC_CONF=actor_allocator))
if result.returncode:
 raise SystemExit(result.returncode)
# External completeness checks add no model/data/training math changes.
import re
log=re.sub(r'\x1b\[[0-9;]*m','',(out.parent/'training.log').read_text())
from verify_formal_prerequisites import verify_outers, loads
verify_outers(log,500)
marker='SEARCH_R1_FINAL_METRICS_JSON='
records=[loads(line.split(marker,1)[1]) for line in log.splitlines() if marker in line]
assert len(records)==1, 'Require exactly one complete terminal final-evaluation record'
metrics=records[0]
assert metrics['val/qa_count/all']==51713
for source,count in expected_final.items():
 assert metrics[f'val/qa_count/{source}']==count and metrics[f'val/evaluated_count/{source}']==count
for step in (5,100,200,300,400,500):
 checkpoint=out.parent/'output/checkpoints/opsd-gpu-formal500/actor'/f'global_step_{step}'
 assert (checkpoint/'config.json').is_file() and (checkpoint/'tokenizer.json').is_file()
 assert (checkpoint/'model.safetensors.index.json').is_file() or len(list(checkpoint.glob('*.safetensors')))==1
(out/'formal-completeness.json').write_text(json.dumps(dict(passed=True,actual_outer_updates=500,final_questions=51713,final_per_source_counts=expected_final,checkpoint_steps=[5,100,200,300,400,500],fresh_hf_reload_pending=True,source_commit=os.environ['OPSD_EXPECTED_COMMIT']),indent=2))
