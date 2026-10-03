"""Hold parent GPU leases across real capacity gates and trainer child."""
import hashlib,json,os,subprocess,time
from pathlib import Path
from omegaconf import OmegaConf
from search_r1.baselines.preflight import validate_inherited_gpu_leases
from search_r1.baselines.holdout import prepare_common_holdout
from search_r1.baselines.controller import validate_config
root=Path(os.environ['SEARCH_R1_BASE']); out=Path(os.environ['OPSD_EVIDENCE'])
lease_fds=tuple(int(r['fd']) for r in json.loads(os.environ['SEARCH_R1_GPU_LEASES']))
# Wait for the completed health check to become idle, preserving the double15% guard.
for attempt in range(30):
    try: validate_inherited_gpu_leases(8);break
    except RuntimeError:
        if attempt==29: raise
        time.sleep(2)
subprocess.run(['python','-u',str(root/'tools/runtime/gpu_teacher_probe.py')],check=True,pass_fds=lease_fds)
capacity=[]
for micro in (1,2,4,8):
    command=['torchrun','--standalone','--nproc_per_node=8',str(root/'tools/runtime/microbatch_probe.py'),'--micro',str(micro)]
    with (out/f'micro-{micro}.log').open('w') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,pass_fds=lease_fds)
    record=out/f'micro-{micro}.json'
    passed=result.returncode==0 and record.exists() and json.loads(record.read_text()).get('passed') is True
    capacity.append(dict(local_micro=micro,returncode=result.returncode,passed=passed))
    (out/'micro-capacity.json').write_text(json.dumps(capacity,indent=2))
    if not passed: break
safe=[r['local_micro'] for r in capacity if r['passed']]
if not safe: raise RuntimeError('No full backward GPU microbatch passed')
local_micro=max(safe)
# Keep margin; actual rollout/FSDP/optimizer residency is verified in the fresh trainer.
os.environ['OPSD_GLOBAL_MICROBATCH']=str(local_micro*8)
args=[f'actor_rollout_ref.actor.ppo_micro_batch_size={local_micro*8}',
'baselines.distillation.teacher_device=cuda','baselines.distillation.max_student_logits_gib=16.0',
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
 result=subprocess.run(base+args,stdout=log,stderr=subprocess.STDOUT,pass_fds=lease_fds)
raise SystemExit(result.returncode)
