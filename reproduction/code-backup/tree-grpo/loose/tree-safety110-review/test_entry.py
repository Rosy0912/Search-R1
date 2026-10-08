import os,subprocess,json,hashlib
from pathlib import Path
from omegaconf import OmegaConf
W=Path.cwd();P=W/'work/tree-safety110-a632ba9-v2';O=W/'work/tree-safety110-review';PY='/Users/shaoshitong/Documents/Search-R1-RL-Baselines/.venv-cpu/bin/python'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',TREE_INSTRUCT_MODEL='/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/assets/qwen2.5-3b-instruct-aa8e725',TREE_COMMON_DATA='/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/common-data',TREE_PPO_CALIBRATION_ONLY='1',PYTHONPATH=str(P)+':'+str(P/'source/repo'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
args=['baselines.method=tree_grpo','trainer.n_gpus_per_node=8','actor_rollout_ref.rollout.n_agent=2','--config-path',str(O),'--config-name','input-config','--cfg','job','--resolve']
results=[]
for name,entry in [('observer',[str(P/'validation_cost_entry.py')]),('module',['-m','search_r1.baselines.main'])]:
 p=subprocess.run([PY,*entry,*args],cwd=P/'source/repo',env=env,text=True,capture_output=True,timeout=90);(O/(name+'-entry.stdout')).write_text(p.stdout);(O/(name+'-entry.stderr')).write_text(p.stderr);assert p.returncode==0,(name,p.stderr)
 # entry source prints no prefix until actual main_task; YAML starts data:
 text=p.stdout[p.stdout.index('data:\n'):];c=OmegaConf.to_container(OmegaConf.create(text),resolve=True);results.append(c)
assert results[0]==results[1]
print(json.dumps(dict(status='ACTUAL_CPU_HYDRA_ENTRY_EQUIVALENCE_PASS',canonical_config_sha256=hashlib.sha256(json.dumps(results[0],sort_keys=True,separators=(',',':')).encode()).hexdigest(),actual_ray_task_started=False,actual_cuda_used=False)))
