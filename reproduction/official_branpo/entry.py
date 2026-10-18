"""Independent BranPO allocation, immutable inputs, and official training driver."""
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .contract import UPSTREAM, DATA_SHA, REVISIONS, job_name, sha, training_overrides

ROOT = '/home/jmzhang/Workspace/Recursive-Tree-GRPO-20261014-v1/official-branpo'
VERSIONS = {'torch':'2.8.0','vllm':'0.11.0','verl':'0.6.1','transformers':'4.57.1',
            'ray':'2.59.0','flash-attn':'2.8.3','pyarrow':'21.0.0','rllm':'0.2.1'}


def actor_patch(text):
    assert 'from reproduction.official_branpo.observe import observe_adam' not in text
    anchor = '                self.actor_optimizer.step()\n'
    assert text.count(anchor) == 1
    extra = ("                import os as _branpo_os\n"
             "                if _branpo_os.environ.get('BRANPO_AUDIT_DIR'):\n"
             "                    from reproduction.official_branpo.observe import observe_adam\n"
             "                    observe_adam(self.actor_optimizer, grad_norm)\n")
    return text.replace(anchor, anchor + extra), extra


def iteration_patch(text):
    assert 'from reproduction.official_branpo.observe import observe_iteration' not in text
    anchor = '                logger.log(data=metrics, step=self.global_steps)\n'
    assert text.count(anchor) == 1
    extra = ("                import os as _branpo_os\n"
             "                if _branpo_os.environ.get('BRANPO_AUDIT_DIR'):\n"
             "                    from reproduction.official_branpo.observe import observe_iteration\n"
             "                    observe_iteration(self.global_steps, metrics)\n")
    return text.replace(anchor, anchor + extra), extra


def validate_plan(plan, script):
    size=plan['size'];assert size in REVISIONS
    assert plan['protocol']=='official_branpo_common_initialization_v1'
    assert plan['job']==job_name(size) and plan['root']==ROOT+'/runs/'+size+('-v5' if size=='3b' else '-v3')
    assert plan['upstream_commit']==UPSTREAM and plan['model_revision']==REVISIONS[size]
    assert plan['versions']==VERSIONS and plan['data_manifest_sha256']==DATA_SHA
    assert plan['expected_actual_iterations']==200 and plan['upstream_labels']==[101,300]
    assert plan['gpus']==8 and plan['no_auto_retry'] is True and plan['no_additional_sft'] is True
    previous=plan['prior_attempt']
    assert previous['job_id']==('139127' if size=='3b' else '139124')
    assert previous['scientific_updates']==0
    assert previous['raw_policy_tokens'] is None if size=='3b' else previous['raw_policy_tokens']==0
    assert previous['status']==('FAILED_AFTER_FIRST_ROLLOUT_POLICY_LOSS_SCHEMA_MISSING_BEFORE_ADAM' if size=='3b' else 'CANCELLED_NEVER_EXECUTED_KNOWN_GPU_VISIBILITY_CONFLICT')
    assert plan['overrides']==training_overrides(script,size,plan['model'],plan['root'])
    assert plan['allocation']==dict(gpus=8,cpus=64,memory_gib=600)
    assert plan['minimum_gpu_memory_mib']==44000
    assert plan['partition']==('ADA6000,L40S' if size=='3b' else 'ADA6000')
    assert plan['node'] is None
    return Path(plan['root'])


def verify(plan):
    upstream=Path(plan['upstream']);root=validate_plan(plan,(upstream/'agent/search/start_branpo_trainer.sh').read_text())
    assert sys.version_info[:3]==(3,12,9)
    for n,v in VERSIONS.items():assert metadata.version(n).split('+')[0]==v,n
    for group,base in [('upstream_files',upstream),('adapter_files',Path(plan['adapter']))]:
        for n,h in plan[group].items():assert sha(base/n)==h,n
    assert len(plan['upstream_files'])==286
    for n,h in plan['model_files'].items():assert sha(Path(plan['model'])/n)==h,n
    previous=plan['prior_attempt'];assert sha(previous['receipt'])==previous['receipt_sha256']
    inspected=json.loads(Path(previous['receipt']).read_text())
    for key in ['job','status','scientific_updates','raw_policy_tokens']:assert inspected[key]==previous[key],key
    assert sha(Path(plan['common'])/'manifest.json')==DATA_SHA
    for n,h in plan['evidence_files'].items():assert sha(n)==h,n
    patch=json.loads(Path(plan['observer_deployment']).read_text())
    for f,h in patch['backend_files'].items():assert sha(f)==h,f
    assert patch['only_post_adam_and_post_completed_iteration'] is True
    assert patch['actual_patched_optimizer_cpu_bit_exact'] is True and patch['rng_unchanged'] is True
    data=json.loads(Path(plan['converted_data_receipt']).read_text())
    assert data['common_manifest_sha256']==DATA_SHA
    for split,count in [('train',169028),('validation',512)]:
        s=data['splits'][split];assert s['rows']==count and s['full_round_trip_exact'] is True
        assert sha(s['tasks'])==s['tasks_sha256'] and sha(s['verl'])==s['verl_sha256']
    preflight=json.loads(Path(plan['cpu_preflight']).read_text())
    assert preflight['status']=='PASSED' and preflight['actual_gpu_training'] is False
    assert preflight['actual_worker_visibility_method_checked'] is True
    assert preflight['actual_rollout_correction_helper_cpu_equivalence'] is True
    assert preflight['overrides']==plan['overrides'] and preflight['model_files']==plan['model_files']
    return root


def ray_runtime_env(base, environ):
    import copy
    result=copy.deepcopy(base)
    for name in ['PYTHONPATH','BRANPO_AUDIT_DIR','SLURM_JOB_ID','SLURM_JOB_NAME']:
        assert environ.get(name), name
        result['env_vars'][name]=environ[name]
    # The assigned devices have been independently verified as NVIDIA. Empty AMD
    # masks prevent an inherited Slurm/image variable from narrowing CUDA devices.
    result['env_vars']['ROCR_VISIBLE_DEVICES']=''
    result['env_vars']['HIP_VISIBLE_DEVICES']=''
    return result


def sanitize_nvidia_visibility(environ, inventory):
    assert len(inventory)==8 and all('NVIDIA' in row for row in inventory)
    assert environ.get('CUDA_VISIBLE_DEVICES'), 'Preserve the scheduler-assigned CUDA mask'
    removed={k:environ.pop(k) for k in ['ROCR_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES'] if k in environ}
    return removed


def ray_init_settings(config):
    return {k:v for k,v in config.ray_init.items() if v is not None} if hasattr(config,'ray_init') else {}

def prepare_policy_loss_schema(config):
    # The official helper attaches its unchanged algorithm config dynamically.
    # Relax only this mapping's schema, leaving every effective value intact.
    from omegaconf import OmegaConf
    assert config.algorithm.rollout_correction.bypass_mode is True
    assert config.algorithm.rollout_correction.get('use_policy_gradient',False) is False
    OmegaConf.set_struct(config.actor_rollout_ref.actor.policy_loss,False)


def build_trainer(plan,cfg):
    prepare_policy_loss_schema(cfg)
    from rllm.data import DatasetRegistry
    data=Path(plan['converted_data_root'])
    DatasetRegistry._REGISTRY_DIR=str(data/'registry');DatasetRegistry._REGISTRY_FILE=str(data/'registry/dataset_registry.json');DatasetRegistry._DATASET_DIR=str(data/'datasets')
    train=DatasetRegistry.load_dataset('branpo_common','train');val=DatasetRegistry.load_dataset('branpo_common','validation')
    assert len(train)==169028 and len(val)==512
    from rllm.agents.search_r1_agent import SearchR1Agent
    from rllm.environments.search_r1.search_r1 import SearchR1Env
    from rllm.rewards.reward_fn import search_r1_reward_fn
    from rllm.trainer.agent_branch_trainer import AgentTrainer
    from .retrieval import make_tool
    env_args=dict(cfg.rllm.env.get('env_args',{}));env_args['max_steps']=env_args.get('max_steps',4)
    env_args['tool_map']={'local_search':make_tool(plan['retriever'])};env_args['reward_fn']=search_r1_reward_fn
    trainer=AgentTrainer(agent_class=SearchR1Agent,env_class=SearchR1Env,config=cfg,
                        train_dataset=train,val_dataset=val,agent_args={},env_args=env_args)
    return trainer


def run(planfile):
    plan=json.loads(Path(planfile).read_text());root=verify(plan)
    jid=os.environ['SLURM_JOB_ID'];sub=json.loads((root/'submission.json').read_text())
    assert sub['job_id']==jid and sub['job']==plan['job'] and sub['plan_sha256']==sha(planfile)
    line=subprocess.check_output(['scontrol','show','job',jid,'-o'],text=True)
    assert 'JobName='+plan['job']+' ' in line and 'UserId=jmzhang(1066)' in line and 'gres/gpu=8' in line
    assert 'JobState=RUNNING' in line
    if plan['node']:assert os.environ['SLURM_JOB_NODELIST']==plan['node']
    inv=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.total','--format=csv,noheader,nounits'],text=True).splitlines()
    assert len(inv)==8 and len({s.split(',')[0] for s in inv})==8
    assert all(float(s.split(',')[-1])>=plan['minimum_gpu_memory_mib'] for s in inv)
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    assert not (root/'checkpoints').exists()
    removed=sanitize_nvidia_visibility(os.environ,inv)
    with (root/'entry-once.json').open('x') as f:json.dump(dict(at=time.time(),job_id=jid,gpus=inv,plan_sha256=sha(planfile),removed_amd_visibility_masks=removed,cuda_mask_preserved=os.environ['CUDA_VISIBLE_DEVICES']),f,indent=2)
    os.environ['BRANPO_AUDIT_DIR']=str(root/'audit')
    os.environ['PYTHONPATH']=plan['adapter']+':'+plan['upstream']
    sys.path[:0]=[plan['adapter'],plan['upstream']];os.chdir(plan['upstream'])
    import verl
    assert '/official-branpo/env/' in verl.__file__, verl.__file__
    from hydra import compose,initialize_config_module
    from omegaconf import OmegaConf
    if not OmegaConf.has_resolver('mul'):
        OmegaConf.register_new_resolver('mul',lambda x,y:int(x)*int(y))
    with initialize_config_module(config_module='rllm.trainer.config',version_base=None):
        cfg=compose(config_name='agent_ppo_trainer',overrides=plan['overrides'])
    trainer=build_trainer(plan,cfg)
    (root/'resolved.yaml').write_text(OmegaConf.to_yaml(cfg,resolve=True))
    import ray
    from rllm.trainer.verl.ray_runtime_env import get_ppo_ray_runtime_env
    settings=ray_init_settings(cfg)
    assert not ray.is_initialized()
    ray.init(runtime_env=ray_runtime_env(get_ppo_ray_runtime_env(),os.environ),**settings)
    try:trainer.train()
    finally:ray.shutdown()
    with (root/'training-returned.json').open('x') as f:json.dump(dict(at=time.time(),job_id=jid),f)


if __name__=='__main__':run(sys.argv[1])
