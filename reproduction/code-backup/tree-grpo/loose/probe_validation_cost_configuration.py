"""CPU-only Hydra inspection: observer entry must preserve the complete config."""
import hashlib, json, os, pathlib, subprocess, sys
import yaml
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
assert os.environ['SLURM_JOB_NODELIST']=='h100-9' and os.environ.get('CUDA_VISIBLE_DEVICES')==''
out=root/'logs'/('validation-cost-config-'+os.environ['SLURM_JOB_ID']); out.mkdir(exist_ok=False)
env=dict(os.environ,SEARCH_R1_ASSETS=str(root/'assets-root'),SEARCH_R1_BASELINE_OUTPUT=str(out/'output'),SEARCH_R1_BASELINE_RUN_ROOT=str(out/'runtime'),PYTHONPATH=str(root/'tools')+':'+str(root/'repo'))
args=['baselines.method=tree_grpo','trainer.n_gpus_per_node=8','actor_rollout_ref.rollout.n_agent=2','actor_rollout_ref.actor.ppo_micro_batch_size=8','retriever.url=http://10.32.0.11:18082/retrieve','baselines.holdout.output_dir='+str(root/'common-data'),'trainer.total_training_steps=501','trainer.smoke_only=false','trainer.val_before_train=true','trainer.logger=[console,wandb]','--cfg','job','--resolve']
reports=[]
for name,entry in [('original',['-m','search_r1.baselines.main']),('observed',[str(root/'tools/validation_cost_entry.py')])]:
    result=subprocess.run([sys.executable]+entry+args,cwd=root/'repo',env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=False)
    (out/(name+'.yaml')).write_bytes(result.stdout); (out/(name+'.stderr')).write_bytes(result.stderr)
    assert result.returncode==0, name+' config inspection failed'
    reports.append(yaml.safe_load(result.stdout))
assert reports[0]==reports[1], 'External observer changes resolved training configuration'
cfg=reports[0]
assert cfg['trainer']['total_training_steps']==501 and cfg['trainer']['n_gpus_per_node']==8
assert cfg['actor_rollout_ref']['actor']['ppo_micro_batch_size']==8 and cfg['retriever']['topk']==3
files=[out/'original.yaml',out/'observed.yaml',root/'tools/validation_cost_hooks.py',root/'tools/validation_cost_entry.py',root/'repo/verl/trainer/main_ppo.py',root/'repo/verl/trainer/ppo/ray_trainer.py',pathlib.Path(__file__)]
r={'status':'CPU_HYDRA_CONFIGURATION_IDENTICAL','no_gpu_allocation_or_training':True,'semantic_config_identical':True,'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
(out/'result.json').write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r),flush=True)
