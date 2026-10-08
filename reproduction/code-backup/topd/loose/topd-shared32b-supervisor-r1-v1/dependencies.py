"""All approvals external. Bootstrap explicitly has no preclaimed coexist proof."""
import importlib.util,json,os,pathlib,subprocess,sys
from contracts import *
def load(name,ref):
 p=own(ref['path']);require(sha(p)==ref['sha256'],'module source drift '+name);spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def gate(config,mode):
 require(mode in ('profile','formal500'),'unknown mode')
 review=artifact(config['source_review']);unsigned={k:v for k,v in config.items() if k!='source_review'}
 require(review['status']=='PASS' and review['approval_scope']=='SOURCE_NATIVE_CPU_SHARED32B_BOOTSTRAP' and review['reviewed_payload_sha256']==digest(unsigned),'whole static recipe/source native approval missing')
 require(config['sealed_runtime_source']['sha256']=='9e41464d49054a1765fed29f5839f9fd8c568f0040aedbf6cf36b3c096adb3c6' and config['telemetry_source']['sha256']=='d340c77f54c8c744816da416c91016e59bd5a21bd42262b2ae672a3432f6588b','Current WindowV4 exact source pair required')
 window=own(config['telemetry_source']['path']).parent
 require(window==ROOT/'tools/qwen25-32b-window-safety-candidate-v4' and sha(window/'files.json')=='9835bc499bcb10da1959ae0e3bce88cf7a7966ad51ed21dc80fbe03368eaba8c','Current WindowV4 full manifest required')
 require(own(config['teacher_gate_source']['path'])==ROOT/'tools/topd-32b-native-score-gate-r1-v1/score_gate.py' and sha(ROOT/'tools/topd-32b-native-score-gate-r1-v1/tool-files.json')=='da88d6d6b9141effb7cec2ac36d7e729e09007facc8d7d12445dbccb34b9e78a','Current scorecaller V7 full source manifest required')
 require(config['code_commit']==D23 and len(config['telemetry_source']['sha256'])==64,'clean d23/explicit telemetry bytes required')
 telemetry=artifact(config['telemetry_native_review']);telemetry_pair(telemetry,config['telemetry_source'],config['sealed_runtime_source'])
 repo=own(config['repo']);require(repo==ROOT/'repo-shared32b','wrong source checkout')
 require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==D23 and not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip(),'dirty/wrong checkout')
 for ref in config['module_sources'].values():require(sha(own(ref['path']))==ref['sha256'],'external module bytes drift')
 require(any(own(t['root'])==pathlib.Path(__file__).resolve().parent for t in config['source_trees']),'Actual executed supervisor whole source tree must be bound')
 for tree in config['source_trees']:
  manifest=artifact(tree['manifest']);verify_tree(own(tree['root']),manifest['files'],own(tree['manifest']['path']))
 gate_module=load('canonical_teacher_gate',config['teacher_gate_source'])
 require(config['teacher_gate_source']['sha256']=='894f26d736e02464e1db272ce292cebdf4e697b154add4fbb6eb91af32924200','canonical teacher gate required')
 provenance=artifact(config['teacher_provenance']);require(provenance['teacher_kind']=='SEARCH_R1_GRPO100_32B','Current fallback requires actual Search-R1-100 trained teacher');model=gate_module.check_identity(provenance,ROOT/'tools/topd-32b-pipeline-candidate')
 numeric=artifact(config['native_numerical_release']);numerical(numeric,provenance['expected_fp32_parameter_sha256'],config['expected_uuids'])
 independent=artifact(config['native_numerical_review']);require(independent['status']=='PASS' and independent['approval_scope']=='SELECTED32B_NUMERICAL_RELEASE_NOT_ALGORITHM_ACCEPTANCE' and independent['reviewed_payload_sha256']==digest(numeric),'native numerical independent review missing')
 # Canonical provenance checks selected64 or verified same-family Search-R1 actual100 quality.
 require(provenance['model_inventory']['sha256']==config['hf_inventory_sha256'],'selected teacher inventory drift')
 if mode=='formal500':
  coexist=artifact(config['resource_proof']);creview=artifact(config['resource_review'])
  require(creview['status']=='PASS' and creview['reviewed_payload_sha256']==digest(coexist),'actual resource independent review missing')
  require(coexist['status']=='PROFILE_COMPLETE_UNREVIEWED_NOT_ALGORITHM_TRAINING' and coexist['actor_global_microbatch']==64 and coexist['actor_ranks']==8 and coexist['teacher_fp32_parameter_sha256']==provenance['expected_fp32_parameter_sha256'],'resource proof wrong teacher/protocol')
  for k in ('actual_actor_backward_observed','teacher_scoring_kv_released','next_rollout_kv_rebuilt','retrieval_http_during_score','all8_teacher_release_zero','original_baseline_preserved','own_cleanup_passed'):require(coexist.get(k) is True,'missing actual resource '+k)
  for ref in coexist['raw_refs']:file_ref(ref)
  ipc=load('import_gate',config['owner_import_gate']);backend=ipc.verified_ipc()
  template=artifact(config['formal_manifest']);backend.validate_manifest(template)
  from topd32b_shared_pipeline.readiness import validate_training_readiness
  validate_training_readiness(template)
 return dict(provenance=provenance,model=str(model),numerical=numeric)

def recipe(mode,run,teacher,ipc):
 # Parent common source composes and freezes resolved Hydra + shared holdout identity.
 return ['baselines.method=topd','baselines.topd.teacher_backend=local_gpu_fp32_shared_pipeline32b',f'baselines.topd.socket_manifest={ipc}/manifest.json',f'baselines.distillation.teacher_path={teacher}','baselines.distillation.privileged_answers=false','baselines.distillation.chunk_size=16','baselines.topd.alpha=0.1','baselines.local_runtime_only=true','baselines.local_runtime_host=h100-11','data.train_batch_size=512','data.val_batch_size=64','data.val_data_num=64','actor_rollout_ref.actor.ppo_mini_batch_size=256','actor_rollout_ref.actor.ppo_micro_batch_size=64','actor_rollout_ref.actor.ppo_epochs=2','actor_rollout_ref.actor.use_kl_loss=false','actor_rollout_ref.actor.entropy_coeff=0','algorithm.kl_ctrl.kl_coef=0','actor_rollout_ref.rollout.n_agent=5','actor_rollout_ref.rollout.tensor_model_parallel_size=1','actor_rollout_ref.rollout.gpu_memory_utilization=0.6','trainer.n_gpus_per_node=8','trainer.nnodes=1','trainer.total_training_steps=501','trainer.acceptance_step=5','trainer.smoke_only=false','trainer.val_before_train=true','trainer.save_freq=100','trainer.test_freq=50',f'trainer.default_local_dir={run}/checkpoints','baselines.bapo.stage_validation_num=64',f'baselines.holdout.output_dir={ROOT}/assets-root/common-data','retriever.url=http://127.0.0.1:18085/retrieve']
