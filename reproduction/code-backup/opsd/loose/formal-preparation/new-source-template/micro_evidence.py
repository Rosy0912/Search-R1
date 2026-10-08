"""Read actual all-eight capacity and pilot YAML; declarations cannot select a batch."""
import math
from pathlib import Path
from verify_formal_prerequisites import loads,own_file,digest

def validate_capacity_record(summary,record,job,commit,micro):
 assert type(micro) is int and micro in (2,4,8)
 assert summary['passed'] is True and summary['job_id']==job and summary['source_commit']==commit
 assert summary['max_safe_local_micro']==micro
 assert summary['actor_allocator_configuration']==''
 assert summary['actual_training_iterations']==0 and summary['vllm_coexistence_tested'] is False
 candidates=summary['candidates'];assert [c['local_micro'] for c in candidates]==[1,2,4,8][:len(candidates)]
 assert candidates and max(c['local_micro'] for c in candidates if c['passed'])==micro
 selected=[c for c in candidates if c['local_micro']==micro]
 assert len(selected)==1 and selected[0]['passed'] is True and selected[0]['returncode']==0
 service=selected[0]['before_idle']['service']
 assert service['job_id']==job and service['hostname'].split('.')[0]=='h100-10'
 assert len(set(service['selected_uuids']))==8 and service['only_gpu_service_pid']>0
 assert '127.0.0.1:18084' in service['listener']
 assert record['passed'] is True and record['local_micro']==micro
 ranks=record['ranks'];assert len(ranks)==8 and {r['rank'] for r in ranks}==set(range(8))
 for row in ranks:
  assert row['passed'] is True and row['job_id']==job and row['world_size']==8 and row['local_micro']==micro
  assert row['hostname'].split('.')[0]=='h100-10'
  assert row['input_width']==6144 and row['response_width']==4096
  assert row['optimizer_steps']==2 and len(row['optimizer_step_records'])==2
  assert row['minimum_required_free_fraction']==.15
  assert row['estimated_peak_device_used_bytes']==row['external_baseline_used_bytes']+row['peak_reserved_bytes']+1024**3
  assert row['external_baseline_used_bytes']>=0 and row['headroom_estimate_includes_nccl_slack_bytes']==1024**3
  assert row['estimated_peak_device_used_bytes']<=.85*row['gpu_total_bytes']
  assert row['student_master_parameter_dtypes']==['torch.float32'] and row['student_mixed_precision_parameter_dtype']=='torch.bfloat16'
  assert row['allocator_configuration']=='' and row['allocator_backend']=='native'
  assert row['peak_reserved_bytes']>=row['peak_allocated_bytes']>0
  for step,item in enumerate(row['optimizer_step_records'],1):
   assert item['step']==step and item['tracked_parameter_delta']>0
   assert math.isfinite(item['loss']) and math.isfinite(item['grad_norm']) and item['grad_norm']>0
  for key in ('teacher/trainable_parameters_per_rank','teacher/gradient_tensors_per_rank','teacher/student_optimizer_membership_per_rank'):assert row['teacher_audit'][key]==0
  for key,device in (('teacher_active_runtime',f'cuda:{row["local_rank"]}'),('teacher_returned_runtime','cpu')):
   runtime=row[key];assert runtime['parameter_dtypes']==['torch.float32'] and runtime['attention']=='eager'
   assert runtime['parameter_devices']==[device] and runtime['active_device']==device
   assert runtime['matmul_tf32'] is False and runtime['cudnn_tf32'] is False
  flags=row['student_checkpoint_training_runtime']
  for key in ('student_wrapper_training','student_base_training','student_decoder_container_training','student_gradient_checkpointing','all_decoder_training'):assert flags[key] is True
  assert flags['checkpoint_use_reentrant'] is False and flags['decoder_count']>0
  calls=row['decoder_forward_pre_hook_counts'];assert len(calls)==flags['decoder_count'] and all(n==4 for n in calls.values())
  allocation=row['allocation'];assert allocation['job_id']==job and allocation['node_short']=='h100-10' and allocation['slurm_job_nodelist']=='h100-10'
  import re
  assert re.search(r'job_'+re.escape(job)+r'(?:/|$)',allocation['own_job_cgroup'])
 return selected[0]

def verify_micro_evidence(review,pilot,job,commit,micro):
 assert review['picked_local_micro']==micro and review['picked_global_micro']==micro*8
 evidence=review['auxiliary_bound_artifacts'];assert set(evidence)=={'capacity_summary','capacity_candidate','pilot_config'}
 paths={}
 for role,item in evidence.items():
  p=own_file(item['path'],pilot);assert digest(p)==item['sha256'];paths[role]=p
 summary=loads(paths['capacity_summary'].read_text());record=loads(paths['capacity_candidate'].read_text())
 selected=validate_capacity_record(summary,record,job,commit,micro)
 assert selected['aggregate_json']==str(paths['capacity_candidate']) and selected['aggregate_sha256']==digest(paths['capacity_candidate'])
 from omegaconf import OmegaConf
 config=OmegaConf.load(paths['pilot_config'])
 assert config.actor_rollout_ref.actor.ppo_micro_batch_size==micro*8
 assert config.trainer.total_training_steps==6 and config.trainer.smoke_only is True
 assert config.trainer.val_before_train is True and config.trainer.test_freq==5 and config.data.val_batch_size==64
 assert config.data.train_batch_size==512 and config.actor_rollout_ref.actor.ppo_mini_batch_size==256
 assert config.actor_rollout_ref.model.path==config.baselines.distillation.teacher_path
 assert config.actor_rollout_ref.model.path.endswith('/assets/qwen2.5-3b')
 assert config.baselines.distillation.teacher_device=='cuda' and config.baselines.distillation.loss_checkpointing is True
 assert config.baselines.local_runtime_only is True and config.baselines.local_runtime_host=='h100-10'
 assert config.retriever.url=='http://127.0.0.1:18084/retrieve'
 return True
