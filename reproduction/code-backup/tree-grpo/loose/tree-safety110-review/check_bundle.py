import hashlib,json,subprocess,os,ast,sys
from pathlib import Path
from hydra import initialize_config_dir,compose
from omegaconf import OmegaConf
W=Path.cwd(); P=W/'work/tree-safety110-a632ba9-v2'; O=W/'work/tree-safety110-review'; H=lambda b:hashlib.sha256(b).hexdigest()
m=json.loads((P/'manifest.json').read_bytes()); assert H((P/'manifest.json').read_bytes())=='84e3e18682afa880f1053a0025eda595fca332d550ec1f6905bc1334049b5a48'
assert len(m['files'])==717
for n,s in m['files'].items():assert H((P/n).read_bytes())==s,n
repo=W/'work/tree-ppo-three-guards'
files=subprocess.check_output(['git','ls-tree','-r','--name-only',m['code_commit']],cwd=repo,text=True).splitlines();assert len(files)==699
for f in files:
 b=subprocess.check_output(['git','show',m['code_commit']+':'+f],cwd=repo);assert H(b)==m['files']['source/repo/'+f],f
assert H((P/'SOURCE_INTERFACE_RECEIPT.json').read_bytes())==m['probability_calibration']['source_interface_receipt_sha256']
for n,s in m['probability_calibration']['source_contract_files'].items():assert m['files']['source/repo/'+n]==s
assert H((W/'work/submit_tree_safety110_once.py').read_bytes())=='35b609d693435166f25eec742bbb722a4c80ca5216ac120d425a48a5426f38a6'
os.environ.update(TREE_INSTRUCT_MODEL='/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/assets/qwen2.5-3b-instruct-aa8e725',TREE_COMMON_DATA='/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/common-data',SEARCH_R1_BASELINE_OUTPUT='/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-safety300-999/output')
with initialize_config_dir(config_dir=str(P/'source/repo/verl/trainer/config'),version_base=None):c=compose(config_name='ppo_tree_safety300',overrides=m['configuration_overrides'])
c=OmegaConf.create(OmegaConf.to_container(c,resolve=True)); OmegaConf.save(c,O/'input-config.yaml')
assert c.trainer.total_training_steps==301 and c.trainer.actual_update_limit==300 and c.trainer.acceptance_step==115
assert c.data.train_batch_size==640 and c.actor_rollout_ref.actor.ppo_mini_batch_size==384 and c.actor_rollout_ref.actor.ppo_micro_batch_size==16
assert c.actor_rollout_ref.ref.model_path==c.actor_rollout_ref.model.path==os.environ['TREE_INSTRUCT_MODEL']
assert c.actor_rollout_ref.actor.optim.lr_warmup_steps==57 and c.actor_rollout_ref.rollout.temperature==1 and c.actor_rollout_ref.rollout.top_p==1 and c.actor_rollout_ref.rollout.top_k==-1
assert c.baselines.tree.roots==2 and c.baselines.tree.leaves==3 and c.retriever.topk==3 and c.max_turns==4 and not c.force_final_answer
assert c.actor_rollout_ref.rollout.sampling_consistency.enabled and not c.actor_rollout_ref.rollout.sampling_consistency.enforce
assert m['probability_calibration']['environment']=={'TREE_PPO_CALIBRATION_ONLY':'1'}
# Actual source fit path: zero-write return is before actor update and budget credit.
s=(P/'source/repo/verl/trainer/ppo/ray_trainer.py').read_text(); ix=s.index("if os.environ.get('TREE_PPO_CALIBRATION_ONLY') == '1':"); assert s.index('return finalize_calibration(self, batch)',ix)<s.index('actor_output = self.actor_rollout_wg.update_actor(batch)',ix)<s.index('self.actual_outer_updates =',ix)
for path in P.glob('*.py'):ast.parse(path.read_text())
subprocess.run(['bash','-n',str(P/'launch.sbatch')],check=True);subprocess.run(['bash','-n',str(P/'run_tree.sh')],check=True)
print(json.dumps(dict(status='LOCAL_FROZEN_BUNDLE_CONFIG_PASS',manifest_sha256=H((P/'manifest.json').read_bytes()),files=717,git_source_files=699,historical_source_manifests=sum(Path(f).name=='manifest.json' for f in files),config_sha256=H((O/'input-config.yaml').read_bytes()),actual_gpu_execution=False)))
