"""A new allocation only; never modify or stop the original live allocation."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .contract import sha, validate_plan


def verify(planfile, check_cpu=True):
    from reproduction.official_branpo.entry import verify as verify_parent
    plan = json.loads(Path(planfile).read_text())
    parent = json.loads(Path(plan['parent_plan']).read_text())
    assert sha(plan['parent_plan']) == plan['parent_plan_sha256']
    validate_plan(plan, parent)
    verify_parent(parent)
    for basekey, fileskey in [('adapter','adapter_files'),('upstream','upstream_files')]:
        for name, digest in plan[fileskey].items():
            assert sha(Path(plan[basekey])/name) == digest, name
    assert sha(plan['native_proof']) == plan['native_proof_sha256']
    proof = json.loads(Path(plan['native_proof']).read_text())
    assert proof['size'] == plan['size'] and proof['parent_job_id'] == plan['parent_job_id']
    assert proof['parent_plan_sha256'] == plan['parent_plan_sha256']
    assert proof['native']['actual_updates'] == 140 and proof['native']['world_size'] == 8
    assert proof['dataloader']['next_label'] == 241 and len(proof['dataloader']['next_ids']) == 256
    for row in proof['native']['files'].values():
        p = Path(row['path']);st = p.stat()
        assert st.st_size == row['bytes'] and st.st_mtime_ns == row['mtime_ns']
        assert sha(p) == row['sha256']
    for row in proof['small_files'].values():
        assert sha(row['path']) == row['sha256']
    if check_cpu:
        assert sha(plan['cpu_preflight']) == plan['cpu_preflight_sha256']
        cpu = json.loads(Path(plan['cpu_preflight']).read_text())
        assert cpu['status'] == 'PASSED' and cpu['actual_gpu_training'] is False
        assert cpu['overrides'] == plan['overrides']
    return plan, parent, proof


def run(planfile):
    plan,parent,proof = verify(planfile)
    from reproduction.official_branpo.entry import (sanitize_nvidia_visibility, build_trainer,
                                                    ray_init_settings, ray_runtime_env)
    root = Path(plan['root']);sub = json.loads((root/'submission.json').read_text())
    jid = os.environ['SLURM_JOB_ID']
    assert sub['job_id'] == jid and sub['job'] == plan['job']
    assert sub['plan_sha256'] == sha(planfile)
    row = subprocess.check_output(['scontrol','show','job',jid,'-o'],text=True)
    assert 'JobState=RUNNING' in row and 'JobName='+plan['job']+' ' in row
    assert 'UserId=jmzhang(1066)' in row and 'gres/gpu=8' in row and 'mem=600G' in row
    inventory = subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.total',
                                        '--format=csv,noheader,nounits'],text=True).splitlines()
    assert len(inventory) == 8 and len({s.split(',')[0] for s in inventory}) == 8
    assert all(float(s.split(',')[-1]) >= 44000 for s in inventory)
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                                        '--format=csv,noheader,nounits'],text=True).strip()
    assert not (root/'checkpoints').exists()
    removed = sanitize_nvidia_visibility(os.environ,inventory)
    with (root/'entry-once.json').open('x') as f:
        json.dump(dict(at=time.time(),job_id=jid,plan_sha256=sha(planfile),gpus=inventory,
                       parent_job_id=plan['parent_job_id'],resume_actual=140,final_actual=150,
                       removed_amd_visibility_masks=removed),f,indent=2)
    audit = root/'audit';audit.mkdir(exist_ok=False)
    os.environ['BRANPO_AUDIT_DIR'] = str(audit)
    os.environ['BRANPO150_NATIVE_PROOF'] = plan['native_proof']
    os.environ['PYTHONPATH'] = plan['adapter']+':'+plan['upstream']
    sys.path[:0] = [plan['adapter'],plan['upstream']];os.chdir(plan['upstream'])
    from hydra import compose,initialize_config_module
    from omegaconf import OmegaConf
    if not OmegaConf.has_resolver('mul'):
        OmegaConf.register_new_resolver('mul',lambda x,y:int(x)*int(y))
    with initialize_config_module(config_module='rllm.trainer.config',version_base=None):
        cfg = compose(config_name='agent_ppo_trainer',overrides=plan['overrides'])
    trainer = build_trainer(parent,cfg)
    (root/'resolved.yaml').write_text(OmegaConf.to_yaml(cfg,resolve=True))
    import ray
    from rllm.trainer.verl.ray_runtime_env import get_ppo_ray_runtime_env
    env = ray_runtime_env(get_ppo_ray_runtime_env(),os.environ)
    env['env_vars']['BRANPO150_NATIVE_PROOF'] = plan['native_proof']
    assert not ray.is_initialized()
    ray.init(runtime_env=env,**ray_init_settings(cfg))
    try:trainer.train()
    finally:ray.shutdown()
    progress = json.loads((audit/'progress.json').read_text())
    assert progress['completed_actual_iterations'] == 150 and progress['completed_upstream_label'] == 250
    assert (root/'checkpoints/global_step_250/data.pt').is_file()
    with (root/'training-returned.json').open('x') as f:
        json.dump(dict(at=time.time(),job_id=jid,actual_updates=150,upstream_label=250,
                       parent_job_id=plan['parent_job_id'],new_updates=10),f,indent=2)


if __name__ == '__main__':
    run(sys.argv[1])
