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
local_micro=capacity['max_safe_local_micro']
assert type(local_micro) is int and local_micro in (1,2,4,8)
# Keep margin; actual rollout/FSDP/optimizer residency is verified in the fresh trainer.
os.environ['OPSD_GLOBAL_MICROBATCH']=str(local_micro*8)
args=[f'actor_rollout_ref.actor.ppo_micro_batch_size={local_micro*8}',
'baselines.distillation.teacher_device=cuda','baselines.distillation.max_student_logits_gib=16.0',
'baselines.distillation.loss_checkpointing=true',
'baselines.local_runtime_only=true','baselines.local_runtime_host=h100-10',
'retriever.url=http://127.0.0.1:18084/retrieve','trainer.total_training_steps=6',
'data.val_batch_size=64','trainer.smoke_only=true','trainer.val_before_train=true','trainer.test_freq=5',
 'trainer.experiment_name=opsd-gpu-acceptance-5','actor_rollout_ref.rollout.log_prob_micro_batch_size=64',
 'actor_rollout_ref.ref.log_prob_micro_batch_size=64']
base=['bash','reproduction/baselines/run.sh','opsd']
resolved=subprocess.check_output(base+args+['--cfg','job','--resolve'],text=True,pass_fds=lease_fds)
(out/'resolved.yaml').write_text(resolved)
config=OmegaConf.create(resolved); manifest=prepare_common_holdout(config);validate_config(config)
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
raise SystemExit(result.returncode)
