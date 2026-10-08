"""Concrete standard GRPO config, from existing data/asset refs; no calibration inputs."""
import argparse,copy,json,pathlib
from compiler_headers import binding
from checkpoint_inventory import sha
from pipeline import atomic
from allocator_policy import POLICY
from source_binding import UPSTREAM,NATIVE,check
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd')
SOURCE=ROOT/'tools/search-r1-32b-grpo100-source-v4'
TOOLS=ROOT/'tools/search-r1-32b-grpo100-tools-v12'

def derive(prior,mode,directory):
 directory=pathlib.Path(directory);steps=2 if mode=='profile2' else 100
 keys=('base_model','train_file','val_file','preflight_source','retrieval_asset_binding','telemetry_policy','base_model_inventory','base_model_inventory_sha256','base64_result_sha256','base64_quality_independent_review','input_bindings')
 c={k:copy.deepcopy(prior[k]) for k in keys}
 c.update(candidate=str(SOURCE),mode=mode,evidence_root=str(directory/'evidence'),checkpoint_root=str(directory/'checkpoints'),actual_outer_receipt_path=str(directory/'actual-outer.json'),final_checkpoint=str(directory/'checkpoints/actor'/('global_step_'+str(steps))),expected_outer_steps=steps,compiler_headers=binding(),support={'path':str(TOOLS/'owner_support.py'),'sha256':'d4de70880869abbec64f81ccd86f6f8dd8755b01d9db2cb93e2ec112d3031a09'},tools=str(TOOLS))
 # Source exact manifest is created when source snapshot is frozen and deployed.
 c['source_manifest']={'path':str(SOURCE/'r1-files.json'),'sha256':None};c['tools_manifest']={'path':str(TOOLS/'files.json'),'sha256':None}
 c['independent_gate']={'path':str(directory/'source-allocation-gate.json'),'sha256':None}
 c['retriever_command']=copy.deepcopy(prior['retriever_command']);c['retriever_command'][3]=str(TOOLS/'retrieval_server.py')
 c['retriever_url']='http://127.0.0.1:18085/retrieve';c['retriever_health_url']='http://127.0.0.1:18085/docs'
 overrides={
 'algorithm.adv_estimator':'grpo','data.train_files':c['train_file'],'data.val_files':c['val_file'],'data.train_batch_size':'512','data.val_batch_size':'64',
 'data.max_prompt_length':'4096','data.max_response_length':'500','data.max_start_length':'2048','data.max_obs_length':'500','data.shuffle_train_dataloader':'true',
 'actor_rollout_ref.model.path':c['base_model'],'actor_rollout_ref.model.enable_gradient_checkpointing':'true','actor_rollout_ref.model.use_remove_padding':'true',
 'actor_rollout_ref.actor.optim.lr':'1e-6','actor_rollout_ref.actor.optim.lr_warmup_steps_ratio':'0.285','+actor_rollout_ref.actor.optim.foreach':'false',
 'actor_rollout_ref.actor.use_kl_loss':'true','actor_rollout_ref.actor.kl_loss_coef':'0.001','actor_rollout_ref.actor.kl_loss_type':'low_var_kl',
 'actor_rollout_ref.actor.ppo_mini_batch_size':'256','actor_rollout_ref.actor.ppo_micro_batch_size':'64','actor_rollout_ref.actor.ppo_epochs':'1','actor_rollout_ref.actor.use_dynamic_bsz':'false','actor_rollout_ref.actor.ulysses_sequence_parallel_size':'1','actor_rollout_ref.actor.state_masking':'true',
 'actor_rollout_ref.actor.fsdp_config.param_offload':'true','actor_rollout_ref.actor.fsdp_config.grad_offload':'true','actor_rollout_ref.actor.fsdp_config.optimizer_offload':'true','actor_rollout_ref.ref.fsdp_config.param_offload':'true',
 'actor_rollout_ref.rollout.log_prob_micro_batch_size':'64','actor_rollout_ref.ref.log_prob_micro_batch_size':'64','actor_rollout_ref.rollout.tensor_model_parallel_size':'8','actor_rollout_ref.rollout.name':'vllm','actor_rollout_ref.rollout.gpu_memory_utilization':'0.4','actor_rollout_ref.rollout.temperature':'1','actor_rollout_ref.rollout.n_agent':'5','actor_rollout_ref.rollout.n':'1',
 'algorithm.no_think_rl':'false','trainer.critic_warmup':'0','trainer.logger':'[console]','trainer.project_name':'Search-R1-32B-Standard','trainer.experiment_name':'grpo100-'+mode,'+trainer.val_before_train':'true','trainer.default_hdfs_dir':'null','trainer.default_local_dir':c['checkpoint_root'],'trainer.n_gpus_per_node':'8','trainer.nnodes':'1','trainer.save_freq':str(steps),'trainer.test_freq':'50','trainer.total_epochs':'15','trainer.total_training_steps':str(steps+1),'+trainer.r1_actual_outer_receipt':c['actual_outer_receipt_path'],'+trainer.weight_continuation_step':'0','max_turns':'4','retriever.url':c['retriever_url'],'retriever.topk':'3'}
 c['input_bindings']=[r for r in c['input_bindings'] if 'calibration' not in r['path']]
 c['training_command']=[str(ROOT/'env/bin/python'),'-m','verl.trainer.main_ppo']+[k+'='+v for k,v in overrides.items()]
 c['cuda_allocator']=copy.deepcopy(POLICY)
 c['telemetry_policy']={'directory':str(ROOT/'tools/qwen25-32b-window-safety-candidate-v5'),'runtime_sha256':'9e41464d49054a1765fed29f5839f9fd8c568f0040aedbf6cf36b3c096adb3c6','guard_sha256':'65165ac7c8f5c9042ba254fd36d26e222096e6764fa83c167061e25530d5ad39','manifest_sha256':'c8506109114ac45d2f18ac7f4eaf7c1f4d411ce8c0fabfabb0e78d82f689dac0','actual_independent_native_review':None}
 c['source_upstream_commit']=UPSTREAM;c['source_native_cpu_evidence']=copy.deepcopy(NATIVE)
 c['source_device_correctness']=None # Actual raw parent + independent CUDA review bound separately.
 c['profile_resource_proof']=None # Must be filled with actual independent profile2 refs for train100.
 return c

def main():
 p=argparse.ArgumentParser();p.add_argument('--existing-asset-config',required=True);p.add_argument('--mode',choices=['profile2','train100'],required=True);p.add_argument('--directory',required=True);a=p.parse_args();d=pathlib.Path(a.directory).resolve()
 if ROOT/'logs' not in d.parents or d.exists():raise ValueError('fresh own logs directory required')
 c=derive(json.loads(pathlib.Path(a.existing_asset_config).read_text()),a.mode,d)
 for key in ('source_manifest','tools_manifest'):c[key]['sha256']=sha(c[key]['path'])
 if c['source_manifest']['sha256']!='12da9b40250360f5a9448d546c93949186eed2e35bf80be9250a1b5e1db338be':raise ValueError('Frozen streamed sourceV4 required')
 check(c,read_native=True)
 d.mkdir(mode=0o700);atomic(d/'static-config.json',c)
 print(json.dumps({'config':str(d/'static-config.json'),'scope':'SOURCE_ASSEMBLED_ALLOCATION_GATE_PENDING_NO_GPU'}))
if __name__=='__main__':main()
